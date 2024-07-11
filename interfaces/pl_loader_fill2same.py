from typing import List, Optional
import torch.utils.data
from torch_geometric.data import Data, Dataset


class Collater:
    def __init__(
        self,
        dataset: Dataset,
        max_num_nodes: int,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.max_num_nodes = max_num_nodes
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]) -> dict:
        pe_list = [data["pe"] for data in batch]
        num_nodes = [data.num_nodes for data in batch]
        maxN = self.max_num_nodes

        batch_node_attr = torch.nn.utils.rnn.pad_sequence(
            [data.x + 1 for data in batch],
            batch_first=True
        )  # B, N, *
        B, N, H = batch_node_attr.size()
        batch_node_attr = torch.cat((batch_node_attr, torch.zeros((B, maxN - N, H), dtype=torch.long)), 1)
        batch_node_mask = torch.tensor([
            [1] * n + [0] * (maxN - n)
            for n in num_nodes
        ])  # B, N

        edge_attr_size = 1 if batch[0].edge_attr.ndim == 1 else batch[0].edge_attr.size(1)
        if edge_attr_size == 1:
            edge_attr_list = [data.edge_attr.view((-1, 1)) for data in batch]
        else:
            edge_attr_list = [data.edge_attr for data in batch]

        full_edge_attr_list = []
        full_edge_mask_list = []
        for idx, data in enumerate(batch):
            full_edge_attr = torch.zeros((maxN, maxN, edge_attr_size), dtype=torch.long)
            full_edge_mask = torch.zeros((maxN, maxN))

            dst, src = data.edge_index
            full_edge_attr[dst, src, :] = (edge_attr_list[idx] + 1)
            full_edge_mask[dst, src] = 1.
            full_edge_attr_list.append(full_edge_attr)
            full_edge_mask_list.append(full_edge_mask)

        batch_full_edge_attr = torch.stack(full_edge_attr_list)
        batch_full_edge_mask = torch.stack(full_edge_mask_list)  # B, N, N

        pe_len = pe_list[0].shape[0]
        batch_full_pe = torch.zeros((len(batch), pe_len, maxN, maxN))
        for idx, pe in enumerate(pe_list):
            n = num_nodes[idx]
            batch_full_pe[idx, :, :n, :n] = pe

        batch_num_nodes = torch.tensor(num_nodes)

        y = torch.stack([data.y for data in batch], 0)
        batch = {
            "batch_node_attr": batch_node_attr,
            "batch_node_mask": batch_node_mask,
            "batch_full_edge_attr": batch_full_edge_attr,
            "batch_full_edge_mask": batch_full_edge_mask,
            "batch_full_pe": batch_full_pe,
            "batch_num_nodes": batch_num_nodes,
            "y": y,
        }
        return batch


class Fill2SameDataLoader(torch.utils.data.DataLoader):
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
        dataset: Dataset,
        max_num_nodes: int,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
        num_workers: int = 1,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            collate_fn=Collater(dataset, max_num_nodes, follow_batch, exclude_keys),
            drop_last=drop_last,
            **kwargs,
        )
