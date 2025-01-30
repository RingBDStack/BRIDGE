import argparse
from datetime import datetime
import os
import wandb


def get_args():
    parser = argparse.ArgumentParser("BRIDGE")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--dataset", type=str, default="Cora", help="data")
    parser.add_argument("--seed", type=int, default=39, help="seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")  
    parser.add_argument("--l2_coef", type=float, default=0.0, help="pre_weight_decay")
    parser.add_argument(
        "--hid_units", type=int, default=256, help="GCN output dimension"
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=0.20401015296835048,
        help="entorpy_loss_weight",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1913510180577923,
        help="routingnetwork_Drop",
    )
    parser.add_argument(
        "--variance_weight",
        type=float,
        default=1521434.9368374627,
        help="variance_loss_weight",
    )
    parser.add_argument(
        "--n_samples", type=float, default=2, help="nums of reinforce randowm feature"
    )
    parser.add_argument(
        "--downstreamlr", type=float, default=0.000962404050084371, help="downstreamlr"
    )
    parser.add_argument(
        "--combinetype", type=str, default="mul", help="the type of text combining"
    )
    parser.add_argument("--reg_weight", type=float, default="1", help="reg_weight")
    parser.add_argument("--reg_thres", type=float, default="0.4", help="reg_thres")
    parser.add_argument(
        "--model_path", type=str, default="unwork", help="be helpful in down only"
    )
    parser.add_argument(
        "--nb_epochs", type=int, default="10000", help="pretrain_num_epochs"
    )
    parser.add_argument("--shot_num", type=int, default="1", help="fewshotnum")
    parser.add_argument(
        "--fw_epochs", type=int, default="400", help="fewshot_num_epochs"
    )
    parser.add_argument("--prompt_times", type=int, default="100", help="total_avg")
    args = parser.parse_args()
    args.patience = 50
    args.sparse = True
    args.LP = False
    args.nonlinearity = "prelu"
    args.is_Reddit = False
    args.num_tokens = 4
    args.unify_dim = 50
    if args.dataset == "Reddit":
        args.is_Reddit = True
        args.num_tokens = 5
    save_dir = "./save_model"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.dataset}_{current_time}.pkl"
    save_path = os.path.join(save_dir, save_name)
    args.save_name = save_path
    wandb.init(
        settings=wandb.Settings(init_timeout=240),
        project="our-watch-acc",
        config={
            "dataset": args.dataset,
            "seed": args.seed,
            "gpu": args.gpu,
            "combinetype": args.combinetype,
            "patience": args.patience,
            "lr": args.lr,
            "l2_coef": args.l2_coef,
            "hid_units": args.hid_units,
            "sparse": args.sparse,
            "LP": args.LP,
            "nonlinearity": args.nonlinearity,
            "is_Reddit": args.is_Reddit,
            "num_tokens": args.num_tokens,
            "unify_dim": args.unify_dim,
            "save_name": args.save_name,
            "lambda_entropy": args.lambda_entropy,
            "dropout_rate": args.dropout_rate,
            "data_path": "../../data",
            "variance_weight": args.variance_weight,
            "n_samples": args.n_samples,
            "downstreamlr": args.downstreamlr,
            "reg_weight": args.reg_weight,
            "reg_thres": args.reg_thres,
            "model_path": args.model_path,
            "nb_epochs": args.nb_epochs,
            "shot_num": args.shot_num,
            "fw_epochs": args.fw_epochs,
            "prompt_times": args.prompt_times,
        },
    )
    config = wandb.config
    return config
