"""
script to run on ZINC task.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch_geometric.datasets import GNNBenchmarkDataset

import utils
from interfaces.pl_data import PlPyGDataTestonValModule
from interfaces.pl_model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    # parser.add_argument("--dataset_name", type=str, default="PATTERN", help="Name of dataset.")
    parser.add_argument("--config_file", type=str, default="configs/pattern.yaml",
                        help="Additional configuration file for different dataset and models.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeat run.")
    args = parser.parse_args()

    args = utils.update_args(args)
    args.full = False
    if args.full:
        args.project_name = "full_" + args.project_name

    train_dataset = GNNBenchmarkDataset("data", args.dataset_name, "train")
    val_dataset = GNNBenchmarkDataset("data", args.dataset_name, "val")
    test_dataset = GNNBenchmarkDataset("data", args.dataset_name, "test")

    # pre-compute Positional encoding
    print("Computing positional encoding ...")
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power)
    args.pe_len = pe_computation.pe_len
    train_dataset._data_list = [pe_computation(data) for data in train_dataset]
    val_dataset._data_list = [pe_computation(data) for data in val_dataset]
    test_dataset._data_list = [pe_computation(data) for data in test_dataset]
    pe_elapsed = time.perf_counter() - time_start
    pe_elapsed = time.strftime("%H:%M:%S", time.gmtime(pe_elapsed)) + f"{pe_elapsed:.2f}"[-3:]
    print(f"Took {pe_elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    MACHINE = os.environ.get("MACHINE", "") + "_"
    for i in range(1, args.runs + 1):
        logger = WandbLogger(f"run_{str(i)}", args.save_dir, offline=args.offline, project=MACHINE + args.project_name)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # loss_criterion = nn.L1Loss()
        # evaluator = torchmetrics.MeanAbsoluteError()
        in_channels = train_dataset._data.x.size(1)
        node_encoder = NodeEncoder(in_channels, args.emb_channels)
        edge_encoder = EdgeEncoder(None, 0)

        modelmodule = PlGNNTestonValModule(
            loss_criterion=weighted_cross_entropy, evaluator=accuracy_SBM,
            args=args, node_encoder=node_encoder, edge_encoder=edge_encoder
        )

        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=args.num_epochs,
            enable_checkpointing=True,
            enable_progress_bar=True,
            logger=logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor="val/metric", mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
                timer
            ]
        )

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {
            "final/best_val_metric": val_result["val/metric"],
            "final/best_test_metric": test_result["test/metric"],
            "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
        }
        print("Positional encoding:", f"({args.pe_method}, {args.pe_power})")
        print("PE computation time:", pe_elapsed)
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
        logger.log_metrics(results)
        wandb.finish()

    return


class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding_idx=0):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_node_attr = batch["batch_node_attr"]
        B, N, _ = batch_node_attr.size()
        node_h: Tensor = self.linear(batch_node_attr[:, :, 0].flatten())
        batch_node_h = node_h.reshape((B, N, -1))  # B, N, H
        batch_node_h = batch_node_h.permute((0, 2, 1))  # B, H, N
        batch_full_node_h = torch.diag_embed(batch_node_h)  # B, H, N, N
        return batch_full_node_h


class EdgeEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding_idx=0):
        super().__init__()
        self.out_channels = 0

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_node_attr: Tensor = batch["batch_node_attr"]
        B, N, _ = batch_node_attr.size()
        batch_full_edge_h = batch_node_attr.new_empty((B, 0, N, N))
        return batch_full_edge_h


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(), weight=weight[true])
        return loss, torch.sigmoid(pred)


if __name__ == "__main__":
    main()
