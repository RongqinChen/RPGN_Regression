"""
script to train on QM9 targets.
"""

import os
import time

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch_geometric.data import Data
from torchmetrics import MeanAbsoluteError
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute
from tqdm import tqdm

import utils
from datasets.QM9Dataset import QM9, conversion
from interfaces.pl_data import PlPyGDataTestonValModule
from interfaces.pl_model import PlGNNTestonValModule
from positional_encoding import PositionalEncodingComputation

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')


class SetY(object):
    def __init__(self, target, mean, std):
        super().__init__()
        self.target = target
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        data.y = (data.label[:, self.target] - self.mean) / self.std
        return data


class InputTransform(object):
    """QM9 input feature transformation. Concatenate x and z together.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data.x = torch.cat([data.z.unsqueeze(-1), data.x], dim=-1)
        data.edge_attr = torch.where(data.edge_attr == 1)[-1]
        data.label = data.y.clone()
        del data.y
        return data


class MeanAbsoluteErrorQM9(MeanAbsoluteError):
    def __init__(self, std, conversion, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.conversion = conversion

    def compute(self) -> Tensor:
        return (_mean_absolute_error_compute(self.sum_abs_error, self.total) * self.std) / self.conversion


def main():
    parser = utils.args_setup()
    parser.add_argument("--dataset_name", type=str, default="QM9", help="Name of dataset.")
    parser.add_argument("--config_file", type=str, default="configs/qm9.yaml",
                        help="Additional configuration file for different dataset and models.")
    parser.add_argument("--task", type=int, default=-1, choices=list(range(19)), help="Train target. -1 for all first 12 targets.")

    args = parser.parse_args()
    args = utils.update_args(args)

    dataset = QM9("data/QM9")
    # pre-compute Positional encoding
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power)
    args.pe_len = pe_computation.pe_len
    input_transform = InputTransform()
    dataset._data_list = [pe_computation(input_transform(data)) for data in tqdm(dataset, 'Computing PE ..')]
    pe_elapsed = time.perf_counter() - time_start
    pe_elapsed = time.strftime("%H:%M:%S", time.gmtime(pe_elapsed)) + f"{pe_elapsed:.2f}"[-3:]
    print(f"Took {pe_elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")

    if args.task == -1:
        targets = list(range(12))
    else:
        targets = [args.task]

    for target in targets:
        args.task = target
        tenprecent = int(len(dataset) * 0.1)
        dataset = dataset.shuffle()
        train_dataset = dataset[2 * tenprecent:]
        test_dataset = dataset[:tenprecent]
        val_dataset = dataset[tenprecent:2 * tenprecent]

        y_list = [data.label[0, target] for data in train_dataset]
        y_train = torch.stack(y_list, 0)
        mean, std = y_train.mean(), y_train.std()

        del train_dataset._data
        del test_dataset._data
        del val_dataset._data
        set_y_fn = SetY(target, mean, std)
        dataset._data_list = [set_y_fn(data) for data in dataset]
        train_dataset._data_list = dataset._data_list
        test_dataset._data_list = dataset._data_list
        val_dataset._data_list = dataset._data_list

        MACHINE = os.environ.get("MACHINE", "") + "_"
        logger = WandbLogger(f"target_{str(target + 1)}", args.save_dir, offline=args.offline, project=MACHINE + args.project_name)
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
            num_workers=args.num_workers
        )

        loss_criterion = nn.MSELoss()
        evaluator = MeanAbsoluteErrorQM9(std.item(), conversion[args.task].item())
        init_encoder = QM9NodeEncoder(out_channels=args.emb_channels)
        edge_encoder = EdgeEncoder(out_channels=args.emb_channels)
        modelmodule = PlGNNTestonValModule(
            loss_criterion=loss_criterion, evaluator=evaluator,
            args=args, init_encoder=init_encoder, edge_encoder=edge_encoder
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


class QM9NodeEncoder(nn.Module):
    def __init__(self, num_types=10, out_channels=8, use_pos=False, padding_idx=0):
        super().__init__()
        self.use_pos = use_pos
        if use_pos:
            in_channels = 22
        else:
            in_channels = 19
        self.z_embedding = nn.Embedding(num_types + 1, 8, padding_idx)
        self.init_proj = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.z_embedding.reset_parameters()
        self.init_proj.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        x: Tensor = batch["batch_node_attr"]
        B, N, _ = x.size()
        z = x[:, :, 0].flatten().long()
        x = x[:, :, 1:].flatten(0, 1) - 1.
        h1 = self.z_embedding(z)
        x2 = torch.cat([h1, x], -1)
        x3 = self.init_proj(x2)
        batch_node_h = x3.reshape((B, N, -1))  # B, N, H
        batch_node_h = batch_node_h.permute((0, 2, 1))  # B, H, N
        batch_full_node_h = torch.diag_embed(batch_node_h)  # B, H, N, N
        return batch_full_node_h


class EdgeEncoder(nn.Module):
    def __init__(self, num_types=4, out_channels=8, padding_idx=0):
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
