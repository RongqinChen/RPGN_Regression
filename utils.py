"""
Utils file for training.
"""

import argparse
import os
import time
import yaml


def args_setup():
    r"""Setup argparser.
    """
    parser = argparse.ArgumentParser("arguments for training and testing")
    # common args
    parser.add_argument("--save_dir", type=str, default="./results", help="Base directory for saving information.")
    parser.add_argument("--seed", type=int, default=234, help="Random seed for reproducibility.")

    # training args
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument("--l2_wd", type=float, default=0., help="L2 weight decay.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--num_warmup_epochs", type=int, default=500, help="Number of warmup epochs.")
    parser.add_argument("--test_eval_interval", type=int, default=10,
                        help="Interval between validation on test dataset.")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Factor in the ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience in the ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument("--offline", action="store_true", help="If true, save the wandb log offline. "
                                                               "Mainly use for debug.")

    # data args
    parser.add_argument("--pe_method", type=str, default="bern_mixed_sym2", help="Positional encoding computation method.")
    parser.add_argument("--pe_power", type=int, default="8", help="Positional encoding power.")

    # model args
    parser.add_argument("--emb_channels", type=int, default=16, help="Embedding size.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden size of the model.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layer for GNN.")
    parser.add_argument("--mlp_depth", type=int, default=6, help="Number of MLP layer of RPGN.")
    parser.add_argument("--norm_type", type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
                        help="Normalization method in model.")
    parser.add_argument("--drop_prob", type=float, default=0.0,
                        help="Probability of zeroing an activation in dropout models.")
    parser.add_argument("--graph_pool", type=str, default="mean", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph level tasks.")
    parser.add_argument("--jumping_knowledge", type=str, default="last",
                        choices=("last", "concat"), help="Jumping knowledge method.")
    return parser


def update_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    r"""Update argparser given config file.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    if args.config_file is not None:
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if isinstance(value, list):
                for v in value:
                    getattr(args, key, []).append(v)
            else:
                setattr(args, key, value)

    arg_list = [
        str(args.num_layers),
        str(args.hidden_channels),
        str(args.mlp_depth),
        str(args.pe_method),
        str(args.pe_power),
    ]
    model_cfg = ".".join(arg_list)
    args.project_name = f"{args.dataset_name}_{model_cfg}"
    args.save_dir = args.save_dir + "/" + args.dataset_name
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed
