import torch


def compute_rrwp(adj: torch.Tensor, pe_power: int):
    K = pe_power
    deg = adj.sum(dim=0)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = deg_inv.reshape((-1, 1)) * adj
    pe_list = [torch.eye(adj.size(0))]
    out = adj
    pe_list.append(out)
    while len(pe_list) <= K:
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=0)  # (K+1) x n x n
    return pe
