"""
script to train on QM9 targets.
"""

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

import utils
from datasets.QM9Dataset import QM9, conversion
from interfaces.pl_data import PlPyGDataTestonValModule
from interfaces.pl_model import PlGNNTestonValModule
from models.input_encoder import EmbeddingEncoder, QM9InputEncoder
from positional_encoding import PositionalEncodingComputation


class StoreLabel(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data.label = data.y.clone()
        return data


class SetY(object):
    def __init__(self, target, mean, std):
        super().__init__()
        self.target = target
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        data.y = (data.label[0, self.target] - self.mean) / self.std
        return data


class InputTransform(object):
    """QM9 input feature transformation. Concatenate x and z together.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:
        x = data.x
        z = data.z
        data.x = torch.cat([z.unsqueeze(-1), x], dim=-1)
        data.edge_attr = torch.where(data.edge_attr == 1)[-1]
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
    parser.add_argument("--task", type=int, default=11, choices=list(range(19)), help="Train target.")
    parser.add_argument("--search", action="store_true", help="If true, run all first 12 targets.")

    args = parser.parse_args()
    args = utils.update_args(args)

    dataset = QM9("data/QM9")
    # pre-compute Positional encoding
    time_start = time.perf_counter()
    pe_computation = PositionalEncodingComputation(args.pe_method, args.pe_power)
    args.pe_len = pe_computation.pe_len
    input_transform = InputTransform()
    store_label_fn = StoreLabel()
    dataset._data_list = [
        pe_computation(input_transform(store_label_fn(data)))
        for data in dataset
    ]
    elapsed = time.perf_counter() - time_start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) + f"{elapsed:.2f}"[-3:]
    print("running time", f"Took {elapsed} to compute positional encoding ({args.pe_method}, {args.pe_power}).")
    args.precomputation_time = elapsed

    if args.search:
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
        y_train = torch.cat(y_list, 0)
        mean, std = y_train.mean(), y_train.std()

        set_y_fn = SetY(target, mean, std)
        dataset._data_list = [set_y_fn(data) for data in dataset]

        logger = WandbLogger(name=f"target_{str(args.task + 1)}", project=args.exp_name, save_dir=args.save_dir, offline=args.offline)
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
        evaluator = MeanAbsoluteErrorQM9(std[args.task].item(), conversion[args.task].item())
        init_encoder = QM9InputEncoder(args.hidden_channels)
        edge_encoder = EmbeddingEncoder(4, args.inner_channels)
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
        logger.log_metrics(results)
        wandb.finish()

    return


class QM9InputEncoder(nn.Module):
    r"""Input encoder for QM9 dataset.
    Args:
        hidden_channels (int): Hidden size.
        use_pos (bool, optional): If True, add position feature to embedding.
    """

    def __init__(self,
                 hidden_channels: int,
                 use_pos: Optional[bool] = False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            in_channels = 22
        else:
            in_channels = 19
        self.init_proj = Linear(in_channels, hidden_channels)
        self.z_embedding = nn.Embedding(10, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        z = x[:, 0].squeeze().long()
        x = x[:, 1:]
        z_emb = self.z_embedding(z)
        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)
        x = self.init_proj(x)

        return x


class QM9NodeEncoder(torch.nn.Module):
    def __init__(self, num_types, out_channels, use_pos, padding_idx=0):
        super().__init__()
        self.use_pos = use_pos
        if use_pos:
            in_channels = 22
        else:
            in_channels = 19
        self.init_proj = nn.Linear(in_channels, out_channels)
        self.z_embedding = nn.Embedding(num_types + 1, out_channels, padding_idx)
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, batch: dict):
        # Encode just the first dimension if more exist
        batch_node_attr = batch["batch_node_attr"]
        B, N, _ = batch_node_attr.size()
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
