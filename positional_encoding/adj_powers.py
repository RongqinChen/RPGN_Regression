import torch


def compute_adjacency_power_series(adj: torch.Tensor, pe_power: int):
    K = pe_power
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    pe_list = [torch.eye(adj.size(0))]
    out = norm_adj
    pe_list.append(out)
    while len(pe_list) <= K:
        out = out @ norm_adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=0)  # (K+1) x n x n
    return pe
