"""
Pytorch lightning data module for PyG dataset.
"""

from typing import Tuple, Optional, List
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from .pl_loader_fill2same import Fill2SameDataLoader
from .pl_loader_groupsame import GroupSameDataLoader


class PlPyGDataModule(LightningDataModule):
    r"""Pytorch lightning data module for PyG dataset.
    Args:
        train_dataset (Dataset): Train PyG dataset.
        val_dataset (Dataset): Validation PyG dataset.
        test_dataset (Dataset): Test PyG dataset.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of process for data loader.
        follow_batch (list, optional): A list of key that will create a corresponding batch key in data loader.
        drop_last (bool, optional): If true, drop the last batch during the training to avoid loss/metric inconsistence.
        fill2same (bool, optional): If flag is true, fill shapes so that all graphs are the same size, otherwise group graphs of the same size.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 0,
        follow_batch: Optional[List[str]] = [],
        drop_last: Optional[bool] = False,
        fill2same: Optional[bool] = True,
    ):

        super(PlPyGDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.follow_batch = follow_batch
        self.drop_last = drop_last
        if fill2same:
            max_num_nodes = max([data.num_nodes for data in train_dataset + val_dataset + test_dataset])
            self.LoaderModule = lambda dataset, batch_size, shuffle, drop_last: \
                Fill2SameDataLoader(dataset, max_num_nodes, batch_size, shuffle, drop_last)
        else:
            self.LoaderModule = GroupSameDataLoader

    def train_dataloader(self) -> DataLoader:
        return self.LoaderModule(self.train_dataset, self.batch_size, True, self.drop_last,
                                 self.follow_batch, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return self.LoaderModule(self.val_dataset, self.batch_size, False, self.drop_last,
                                 self.follow_batch, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return self.LoaderModule(self.test_dataset, self.batch_size, False, self.drop_last,
                                 self.follow_batch, num_workers=self.num_workers)


class PlPyGDataTestonValModule(PlPyGDataModule):
    r"""In validation mode, return both validation and test set for validation.
        Should use with PlGNNTestonValModule .
    """

    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return (DataLoader(self.val_dataset, self.batch_size, False, self.drop_last,
                           self.follow_batch, num_workers=self.num_workers),
                DataLoader(self.test_dataset, self.batch_size, False, self.drop_last,
                           self.follow_batch, num_workers=self.num_workers))

    def test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return self.val_dataloader()
