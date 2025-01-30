from torch_geometric.datasets import TUDataset, Planetoid, Amazon, Coauthor, Reddit
from torch_geometric.loader import DataLoader
import os
import wandb


def get_loader_pretrain_data(dataset):
    config = wandb.config
    data_path = config.data_path
    dataset1 = Planetoid(root=data_path, name="Cora")
    dataset2 = Planetoid(root=data_path, name="Citeseer")
    dataset3 = Planetoid(root=data_path, name="Pubmed")
    dataset4 = Amazon(root=data_path, name="Photo")
    dataset5 = Amazon(root=data_path, name="Computers")

    if dataset == "Cora":
        loader1 = DataLoader(dataset2)
        loader2 = DataLoader(dataset3)
        loader3 = DataLoader(dataset4)
        loader4 = DataLoader(dataset5)
    if dataset == "Citeseer":
        loader1 = DataLoader(dataset1)
        loader2 = DataLoader(dataset3)
        loader3 = DataLoader(dataset4)
        loader4 = DataLoader(dataset5)
    if dataset == "Pubmed":
        loader1 = DataLoader(dataset1)
        loader2 = DataLoader(dataset2)
        loader3 = DataLoader(dataset4)
        loader4 = DataLoader(dataset5)
    if dataset == "Photo":
        loader1 = DataLoader(dataset1)
        loader2 = DataLoader(dataset2)
        loader3 = DataLoader(dataset3)
        loader4 = DataLoader(dataset5)
    if dataset == "Computers":
        loader1 = DataLoader(dataset1)
        loader2 = DataLoader(dataset2)
        loader3 = DataLoader(dataset3)
        loader4 = DataLoader(dataset4)
    if dataset == "Reddit":
        num_tokens = 5
        loader1 = DataLoader(dataset1)
        loader2 = DataLoader(dataset2)
        loader3 = DataLoader(dataset3)
        loader4 = DataLoader(dataset4)
        loader5 = DataLoader(dataset5)
        return loader1, loader2, loader3, loader4, loader5
    return loader1, loader2, loader3, loader4


def get_loader_down_data(dataset):
    config = wandb.config
    data_path = config.data_path
    if dataset == "Cora":
        data_down = Planetoid(root=data_path, name="Cora")
        loader = DataLoader(data_down)

    elif dataset == "Citeseer":
        data_down = Planetoid(root=data_path, name="Citeseer")
        loader = DataLoader(data_down)

    elif dataset == "Pubmed":
        data_down = Planetoid(root=data_path, name="Pubmed")
        loader = DataLoader(data_down)

    elif dataset == "Photo":
        data_down = Amazon(root=data_path, name="Photo")
        loader = DataLoader(data_down)

    elif dataset == "Computers":
        data_down = Amazon(root=data_path, name="Computers")
        loader = DataLoader(data_down)

    elif dataset == "Reddit":
        data_path1 = data_path
        data_path1 = os.path.join(data_path1, "Reddit")
        data_down = Reddit(root=data_path1)
        loader = DataLoader(data_down)

    else:
        raise ValueError(f"Dataset {dataset} is not recognized.")
    return loader
