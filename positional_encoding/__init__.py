import torch
from torch_geometric.data import Data

from .adj_powers import compute_adjacency_power_series
from .rrwp import compute_rrwp
from .bernstein import compute_bernstein_polynomials
from .bern_mixed_sym2 import compute_bern_mixed_sym2
from .bern_mixed_sym3 import compute_bern_mixed_sym3
from .bern_mixed_smooth import compute_bern_mixed_smooth


pe_computer_dict = {
    "adj_powers": compute_adjacency_power_series,       # output_len: K+1
    "rrwp": compute_rrwp,                               # output_len: K+1
    "bernstein": compute_bernstein_polynomials,         # output_len: K+2
    "bern_mixed_sym2": compute_bern_mixed_sym2,         # output_len: K*2+1
    "bern_mixed_sym3": compute_bern_mixed_sym3,         # output_len: K//2*5+1
    "bern_mixed_smooth": compute_bern_mixed_smooth,     # output_len: K+1
}

pe_len_dict = {
    "adj_powers": lambda K: K + 1,
    "rrwp": lambda K: K + 1,
    "bernstein": lambda K: K + 2,
    "bern_mixed_sym2": lambda K: K * 2 + 1,
    "bern_mixed_sym3": lambda K: K // 2 * 5 + 1,
    "bern_mixed_smooth": lambda K: K + 1,
}


class PositionalEncodingComputation(object):
    r"""Positional encoding computation.
    Args:
        pe_method (str): the method computes positional encoding.
        task (int): Specify the task in dataset if it has multiple targets.
    """

    def __init__(self, pe_method: str, pe_power: int, task: int = None):
        self.compute_pe = pe_computer_dict[pe_method]
        self.pe_power = pe_power
        self.pe_len = pe_len_dict[pe_method](pe_power)
        self.task = task

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        if "x" not in data:
            data["x"] = torch.zeros([N, 1]).long()
        if "edge_attr" not in data:
            data["edge_attr"] = torch.zeros([data.num_edges, 1]).long()

        adj = torch.zeros((N, N))
        edge_index = data["edge_index"]
        adj[edge_index[0], edge_index[1]] = torch.ones((edge_index.size(1),))
        data["pe"] = self.compute_pe(adj, self.pe_power)

        if self.task is not None:
            data.y = data.y[:, self.task]
        return data
