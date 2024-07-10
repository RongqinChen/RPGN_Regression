"""
script to train on ZINC task.
"""
import time

import torch
import torchmetrics
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch_geometric.datasets import ZINC

import utils
from interfaces.pl_data import PlPyGDataTestonValModule
from interfaces.pl_model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_float32_matmul_precision('high')


def main():
    parser = utils.args_setup()
    parser.add_argument("--dataset_name", type=str, default="ZINC", help="Name of dataset.")
    parser.add_argument("--config_file", type=str, default="configs/zinc.yaml",
                        help="Additional configuration file for different dataset and models.")
    parser.add_argument("--task_type", type=str, default="graph_regression", help="Task type.")
    parser.add_argument("--num_task", type=int, default=1, help="The number of tasks.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeat run.")
    # parser.add_argument("--full", action="store_true", help="If true, run ZINC full.")
    args = parser.parse_args()

    args = utils.update_args(args)
    args.full = False
    if args.full:
        args.exp_name = "full_" + args.exp_name

    path = "data/ZINC"
    train_dataset = ZINC(path, not args.full, "train")
    val_dataset = ZINC(path, not args.full, "val")
    test_dataset = ZINC(path, not args.full, "test")

    # pre-compute Positional encoding
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power)
    args.pe_len = pe_computation.pe_len
    train_dataset._data_list = [pe_computation(data) for data in train_dataset]
    val_dataset._data_list = [pe_computation(data) for data in val_dataset]
    test_dataset._data_list = [pe_computation(data) for data in test_dataset]

    elapsed = time.perf_counter() - time_start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) + f"{elapsed:.2f}"[-3:]
    print("running time", f"Took {elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f"run_{str(i)}", project=args.project_name, save_dir=args.save_dir, offline=args.offline)
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
        loss_criterion = nn.L1Loss()
        evaluator = torchmetrics.MeanAbsoluteError()
        args.mode = "min"
        init_encoder = NodeEncoder(21, args.emb_channels)
        edge_encoder = EdgeEncoder(4, args.emb_channels)

        modelmodule = PlGNNTestonValModule(
            loss_criterion=loss_criterion,
            evaluator=evaluator,
            args=args,
            init_encoder=init_encoder,
            edge_encoder=edge_encoder
        )
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          max_epochs=args.num_epochs,
                          enable_checkpointing=False,
                          enable_progress_bar=True,
                          logger=logger,
                          callbacks=[TQDMProgressBar(refresh_rate=20),
                                     ModelCheckpoint(monitor="val/metric", mode=args.mode),
                                     LearningRateMonitor(logging_interval="epoch"), timer])

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")
        results = {
            "final/best_val_metric": val_result["val/metric"],
            "final/best_test_metric": test_result["test/metric"],
            "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
        }
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
        logger.log_metrics(results)
        wandb.finish()

    return


class NodeEncoder(torch.nn.Module):
    def __init__(self, num_types, out_channels, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_types + 1, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_node_attr = batch["batch_node_attr"]
        N = 37
        B, _, _ = batch_node_attr.size()
        node_h: Tensor = self.emb(batch_node_attr[:, :, 0].flatten())
        batch_node_h = node_h.reshape((B, N, -1))  # B, N, H
        batch_node_h = batch_node_h.permute((0, 2, 1))  # B, H, N
        batch_full_node_h = torch.diag_embed(batch_node_h)  # B, H, N, N
        return batch_full_node_h


class EdgeEncoder(torch.nn.Module):
    def __init__(self, num_types, out_channels, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_types + 1, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_full_edge_attr = batch["batch_full_edge_attr"]
        B, N, N, _ = batch_full_edge_attr.size()
        edge_h: Tensor = self.emb(batch_full_edge_attr[:, :, :, 0].flatten())
        batch_full_edge_h = edge_h.reshape((B, N, N, -1))
        batch_full_edge_h = batch_full_edge_h.permute((0, 3, 1, 2)).contiguous()
        return batch_full_edge_h


if __name__ == "__main__":
    main()
