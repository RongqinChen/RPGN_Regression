import random
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Sequence, Union

import torch.utils.data
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


class Collater:
    def __init__(
        self,
        dataset: Dataset,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        pe_list = [data["pe"] for data in batch]
        pe_batch = torch.stack(pe_list)
        batch = Batch.from_data_list(
            batch,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys + ["pe"] if self.exclude_keys is not None else ["pe"],
        )
        batch["pe"] = pe_batch
        return batch


class BatchSampler():
    r"""Batch sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don"t do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sample_groups = self._group_same_size()

    def _group_same_size(self):
        sample_groups = defaultdict(list)
        for idx, data in enumerate(self.dataset):
            sample_groups[data.num_nodes].append(idx)
        return sample_groups

    def _split_to_batches(self):
        indices_list = []
        for group_indices in self.sample_groups.values():
            if self.shuffle:
                random.shuffle(group_indices)
            if self.drop_last and len(self.dataset) % self.batch_size < self.batch_size:
                length = len(group_indices) // self.batch_size * self.batch_size
            else:
                length = len(group_indices)

            if length > 0:
                batch_indices_list = [
                    group_indices[idx: idx + self.batch_size]
                    for idx in range(0, length - self.batch_size + 1, self.batch_size)
                ]
                indices_list.extend(batch_indices_list)

        if self.shuffle:
            random.shuffle(indices_list)
        return indices_list

    def __iter__(self) -> Iterator[List[int]]:
        indices_list = self._split_to_batches()
        return iter(indices_list)

    def __len__(self) -> int:
        # shuffle operation will not change the result size
        indices_list = self._split_to_batches()
        return len(indices_list)


class GroupSameDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        batch_sampler = BatchSampler(dataset, batch_size, shuffle, drop_last)
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
