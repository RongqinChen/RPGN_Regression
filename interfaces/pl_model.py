"""
Pytorch lightning model module for PyG model.
"""
import math
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict, Optional
import torch.optim as optim
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from models.model_construction import make_model


class PlGNNModule(LightningModule):
    r"""Basic pytorch lighting module for GNNs.
    Args:
        loss_criterion (nn.Module) : Loss compute module.
        evaluator (Metric): Evaluator for evaluating model performance.
        args (ArgumentParser): Arguments dict from argparser.
        init_encoder (nn.Module, optional): Node feature initial encoder.
        edge_encoder (nn.Module, optional): Edge feature encoder.
    """

    def __init__(
        self, loss_criterion: nn.Module, evaluator: Metric,
        args: ArgumentParser, model: nn.Module
    ):
        super(PlGNNModule, self).__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.train_evaluator = deepcopy(evaluator)
        self.val_evaluator = deepcopy(evaluator)
        self.test_evaluator = deepcopy(evaluator)
        self.args = args

    def forward(self, data: Data) -> Tensor:
        return self.model(data)

    def training_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = batch["y"].squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("train/loss", loss, prog_bar=True, batch_size=self.args.batch_size)
        self.train_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_train_epoch_end(self):
        self.log("train/metric", self.train_evaluator.compute(), prog_bar=False)
        self.train_evaluator.reset()

    def validation_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = batch["y"].squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("val/loss", loss, prog_bar=False, batch_size=self.args.batch_size)
        self.val_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_validation_epoch_end(self):
        self.log("val/metric", self.val_evaluator.compute(), prog_bar=True)
        self.val_evaluator.reset()

    def test_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = batch["y"].squeeze()
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("test/loss", loss, prog_bar=False, batch_size=self.args.batch_size)
        self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_test_epoch_end(self) -> None:
        self.log("test/metric", self.test_evaluator.compute(), prog_bar=True)
        self.test_evaluator.reset()

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=eval(self.args.lr),
            weight_decay=eval(self.args.l2_wd),
        )

        num_warmup_epochs = self.args.num_warmup_epochs

        def lr_lambda(current_step, num_cycles: float = 0.5):
            if current_step < num_warmup_epochs:
                return max(1e-6, float(current_step) / float(max(1, num_warmup_epochs)))

            progress = float(current_step - num_warmup_epochs) / float(
                max(1, self.args.num_epochs - num_warmup_epochs)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict:
        r"""Remove 'v_num' in progress bar for clarity"""
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class PlGNNTestonValModule(PlGNNModule):
    r"""Given a preset evaluation interval, run test dataset during validation
        to have a snoop on test performance every args.test_eval_interval epochs during training.
    """

    def __init__(
        self, loss_criterion: nn.Module, evaluator: Metric, args: ArgumentParser,
        node_encoder: Optional[nn.Module] = None,
        edge_encoder: Optional[nn.Module] = None,
    ):
        model = make_model(args, node_encoder, edge_encoder)
        super(PlGNNTestonValModule, self).__init__(loss_criterion, evaluator, args, model)
        self.test_eval_still = self.args.test_eval_interval

    def validation_step(self, batch: Data, batch_idx: Tensor, dataloader_idx: int) -> Dict:

        if dataloader_idx == 0:
            y = batch["y"].squeeze()
            out = self.model(batch).squeeze()
            loss = self.loss_criterion(out, y)
            self.log("val/loss", loss, prog_bar=False, batch_size=self.args.batch_size, add_dataloader_idx=False)
            self.val_evaluator.update(out, y)
        else:
            if self.test_eval_still != 0:
                return {'loader_idx': dataloader_idx}
            # only do validation on test set when reaching the predefined epoch.
            else:
                y = batch["y"].squeeze()
                out = self.model(batch).squeeze()
                loss = self.loss_criterion(out, y)
                self.log("test/loss", loss, prog_bar=False, batch_size=self.args.batch_size, add_dataloader_idx=False)
                self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y, 'loader_idx': dataloader_idx}

    def on_validation_epoch_end(self):
        self.log("val/metric", self.val_evaluator.compute(), prog_bar=True, add_dataloader_idx=False)
        self.val_evaluator.reset()
        if self.test_eval_still == 0:
            self.log("test/metric", self.test_evaluator.compute(), prog_bar=True, add_dataloader_idx=False)
            self.test_evaluator.reset()
            self.test_eval_still = self.args.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1

    def set_test_eval_still(self):
        # set test validation interval to zero to performance test dataset validation.
        self.test_eval_still = 0

    def on_test_epoch_start(self):
        self.set_test_eval_still()

    def test_step(self, batch: Data, batch_idx: Tensor, dataloader_idx: int) -> Dict:
        results = self.validation_step(batch, batch_idx, dataloader_idx)
        return results

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
