import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
from BRIDGE.models import LogReg
from BRIDGE.model import PrePrompt, pca_compression
from BRIDGE.model import PrePrompt as preprompt
from BRIDGE.utils import process
import pdb
import tqdm
import argparse
from BRIDGE.model import *
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser("BRIDGE")
import torch.nn.functional as F
from BRIDGE.config import get_args
from BRIDGE.utils.data_util import get_loader_pretrain_data, get_loader_down_data
from torch_geometric.datasets import TUDataset, Planetoid, Amazon, Coauthor, Reddit
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import torch.nn as nn
import wandb
from sklearn.decomposition import TruncatedSVD


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_laplacian_evd(adj):
    adj = adj.copy()
    adj.setdiag(-adj.sum(axis=1))
    adj = -adj
    svd = TruncatedSVD(n_components=100, n_iter=20, random_state=42)
    svd.fit(adj)
    eival = torch.tensor(svd.explained_variance_**0.5, dtype=torch.float32).to("cuda")
    eivec = torch.tensor(svd.components_, dtype=torch.float32).to("cuda")
    return eival, eivec


def train_model(
    unify_dim,
    reg_weight,
    reg_thres,
    is_Reddit,
    sparse,
    num_tokens,
    hid_units,
    nonlinearity,
    lr,
    l2_coef,
    nb_epochs,
    patience,
    LP,
    lambda_entropy,
    n_samples=3,
    variance_weight=0.1,
    downstreamlr=0.001,
):
    model = PrePrompt(
        unify_dim,
        hid_units,
        nonlinearity,
        1,
        3,
        0.1,
        args.combinetype,
        variance_weight,
        num_tokens,
        n_samples,
    )
    print("#" * 50)
    print("Downastream dataset is ", args.dataset)
    testsetsize = 1000
    loader = get_loader_down_data(args.dataset)
    for data in loader:
        features, adj = process.process_tu(data, data.x.shape[1])
        eival, eivec = get_laplacian_evd(adj)
        features = pca_compression(features, k=unify_dim)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_adj = sp_adj.cuda()
        features = torch.FloatTensor(features).cuda()
        print(features.shape)
        idx_test = range(data.y.shape[0] - testsetsize, data.y.shape[0])
        labels = data.y
        data = np.array(data.y)
        np.unique(data)
        nb_classes = len(np.unique(data))
        print(nb_classes)
    neighboradj = adj.todense().A
    neighborslist = [[] for x in range(testsetsize)]
    neighbors_2hoplist = [[] for x in range(testsetsize)]
    testindex = [[] for x in range(testsetsize)]
    testlist = [[] for x in range(testsetsize)]

    for x, y in enumerate(idx_test):
        neighborslist[x], neighbors_2hoplist[x] = process.find_2hop_neighbors(
            neighboradj, y
        )
        testlist[x] = [y] + neighborslist[x] + neighbors_2hoplist[x]
        testindex[x] = [x] * len(testlist[x])

    neighborslist = sum(neighborslist, [])
    neighbors_2hoplist = sum(neighbors_2hoplist, [])
    testlist = sum(testlist, [])
    testindex = sum(testindex, [])
    testlist = torch.Tensor(testlist).type(torch.long).cuda()
    testindex = torch.Tensor(testindex).type(torch.long).cuda()
    print(len(list(set(testindex))))
    model = model.cuda()
    model.load_state_dict(
        torch.load(
            "/home/LAB/shijh25/BRIDGE_GRAPH_RULE/scripts/saved_model/{}".format(
                args.model_path
            )
        )
    )
    embeds, _ = model.embed(
        features, sp_adj if sparse else adj, sparse, None, LP
    )
    downstreamlrlist = [downstreamlr]
    acclist = torch.FloatTensor(
        100,
    ).cuda()
    xent = nn.CrossEntropyLoss()
    test_embs = embeds[0, testlist]
    config = wandb.config
    data_path = config.data_path

    for downstreamlr in downstreamlrlist:
        test_lbls = labels[idx_test].cuda()
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        print("-" * 100)
        for shotnum in range(args.shot_num, args.shot_num + 1):
            tot = torch.zeros(1)
            tot = tot.cuda()
            accs = []
            cnt_wait = 0
            best = 1e9
            best_t = 0
            print("shotnum", shotnum)
            for i in tqdm(range(args.prompt_times)):
                masks_logits = model.masks_logits
                soft_masks = torch.sigmoid(masks_logits)
                log = downprompt(
                    soft_masks,
                    hid_units,
                    nb_classes,
                    args.combinetype,
                    unify_dim,
                    num_tokens,
                ).cuda()
                idx_train = torch.load(
                    "{}/fewshot_{}_50_graph/{}-shot_{}_graph/{}/idx.pt".format(
                        data_path,
                        args.dataset.lower(),
                        shotnum,
                        args.dataset.lower(),
                        i,
                    )
                )
                train_batch = torch.load(
                    "{}/fewshot_{}_50_graph/{}-shot_{}_graph/{}/batch.pt".format(
                        data_path,
                        args.dataset.lower(),
                        shotnum,
                        args.dataset.lower(),
                        i,
                    )
                )
                train_batch = torch.Tensor(train_batch).type(torch.long).cuda()
                pretrain_embs = embeds[0, idx_train]
                train_lbls = (
                    torch.load(
                        "{}/fewshot_{}_50_graph/{}-shot_{}_graph/{}/labels.pt".format(
                            data_path,
                            args.dataset.lower(),
                            shotnum,
                            args.dataset.lower(),
                            i,
                        )
                    )
                    .type(torch.long)
                    .squeeze()
                    .cuda()
                )
                opt = torch.optim.Adam([{"params": log.parameters()}], lr=downstreamlr)
                log = log.cuda()
                best = 1e9
                pat_steps = 0
                best_acc = torch.zeros(1)
                best_acc = best_acc.cuda()
                best_acc = 0
                cnt_wait = 0
                for idx_temp in range(args.fw_epochs):
                    log.train()
                    opt.zero_grad()
                    logits, entropy_logits, reg_loss = log(
                        eivec,
                        eival,
                        reg_thres,
                        features,
                        sp_adj,
                        sparse,
                        model.gcn,
                        idx_train,
                        train_batch,
                        pretrain_embs,
                        train_lbls,
                        1,
                    )
                    entropy_loss_value = torch.mean(entropy_logits)
                    loss = xent(logits, train_lbls)
                    loss = (
                        loss
                        + lambda_entropy * entropy_loss_value
                        + reg_weight * reg_loss
                    )
                    logits_test, _, _ = log(
                        eivec,
                        eival,
                        reg_thres,
                        features,
                        sp_adj,
                        sparse,
                        model.gcn,
                        testlist,
                        testindex,
                        test_embs,
                    )
                    preds = torch.argmax(logits_test, dim=1).cuda()
                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    if acc > best_acc:
                        best_acc = acc
                        cnt_wait = 0
                    else:
                        cnt_wait += 1
                    if cnt_wait == patience:
                        print(
                            f"Early stopping at iteration {idx_temp + 1} with best accuracy: {best_acc:.4f}"
                        )
                        break
                    loss.backward(retain_graph=True)
                    opt.step()
                logits, _, _ = log(
                    eivec,
                    eival,
                    reg_thres,
                    features,
                    sp_adj,
                    sparse,
                    model.gcn,
                    testlist,
                    testindex,
                    test_embs,
                )
                preds = torch.argmax(logits, dim=1).cuda()
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                accs.append(acc * 100)
                tot += acc
            print("-" * 100)
            print("Average accuracy:[{:.4f}]".format(tot.item() / 100))
            accs = torch.stack(accs)
            mean_acc = accs.mean().item()
            std_acc = accs.std().item()
            print("Mean:[{:.4f}]".format(mean_acc))
            print("Std :[{:.4f}]".format(std_acc))
            print("-" * 100)
            wandb.log(
                {
                    "shot_num": shotnum,
                    "learning_rate": lr,
                    "downstream_learning_rate": downstreamlr,
                    "hidden_units": hid_units,
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                }
            )
            print("-" * 100)
            row = [
                shotnum,
                unify_dim,
                lr,
                downstreamlr,
                hid_units,
                accs.mean().item(),
                accs.std().item(),
            ]
            out = open(
                "{}/ICML25_{}_graph_fewshot.csv".format(
                    data_path, args.dataset.lower()
                ),
                "a",
                newline="",
            )
            csv_writer = csv.writer(out, dialect="excel")
            csv_writer.writerow(row)


if __name__ == "__main__":
    args = get_args()
    print("-" * 100)
    print(args)
    print("-" * 100)
    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda")
    print(device)
    unify_dim = args.unify_dim
    is_Reddit = args.is_Reddit
    sparse = args.sparse
    num_tokens = args.num_tokens
    hid_units = args.hid_units
    nonlinearity = args.nonlinearity
    lr = args.lr
    l2_coef = args.l2_coef
    nb_epochs = args.nb_epochs
    patience = args.patience
    LP = args.LP
    lambda_entropy = args.lambda_entropy
    n_samples = args.n_samples
    variance_weight = args.variance_weight
    downstreamlr = args.downstreamlr
    reg_weight = args.reg_weight
    reg_thres = args.reg_thres
    train_model(
        unify_dim,
        reg_weight,
        reg_thres,
        is_Reddit,
        sparse,
        num_tokens,
        hid_units,
        nonlinearity,
        lr,
        l2_coef,
        nb_epochs,
        patience,
        LP,
        lambda_entropy,
        n_samples,
        variance_weight,
        downstreamlr,
    )
