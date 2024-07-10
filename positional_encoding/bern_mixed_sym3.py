import torch
from scipy.special import comb


def compute_bern_mixed_sym3(adj: torch.Tensor, pe_power: int):
    K = pe_power
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj0 = eye + norm_adj
    adj1 = eye - norm_adj

    num_nodes = adj.size(0)
    eye = torch.eye(num_nodes)
    basis_2ddict = {
        1: {0: adj0, 1: adj1},
        2: {0: adj0 @ adj0, 1: adj0 @ adj1, 2: adj1 @ adj1}
    }
    for k in range(4, K + 1, 2):
        a_idx = ((k // 2) + 1) // 2 * 2
        b_idx = k - a_idx
        basis_dict = {
            (k // 2 - 1): basis_2ddict[a_idx][a_idx // 2] @ basis_2ddict[b_idx][b_idx // 2 - 1],
            (k // 2 + 0): basis_2ddict[a_idx][a_idx // 2] @ basis_2ddict[b_idx][b_idx // 2 + 0],
            (k // 2 + 1): basis_2ddict[a_idx][a_idx // 2] @ basis_2ddict[b_idx][b_idx // 2 + 1],
        }
        basis_2ddict[k] = basis_dict

    for k in range(3, K + 1, 2):
        a_idx = ((k // 2) + 1) // 2 * 2
        b_idx = k - a_idx
        basis_dict = {
            (k // 2 + 0): basis_2ddict[a_idx][a_idx // 2] @ basis_2ddict[b_idx][b_idx // 2 + 0],
            (k // 2 + 1): basis_2ddict[a_idx][a_idx // 2] @ basis_2ddict[b_idx][b_idx // 2 + 1],
        }
        basis_2ddict[k] = basis_dict

    #   k   ()//2    ()+1    ()//2   ()*2   a_idx   b_idx   k//2   a_idx // 2  b_idx // 2
    #   4     2        3       1       2      2       2       2       1           1
    #   6     3        4       2       4      4       2       3       2           1
    #   8     4        5       2       4      4       4       4       2           2
    for k in range(2, K + 1, 2):
        basis_2ddict[k][(k // 2 - 1)] *= ((2 ** -k) * comb(k, k // 2 - 1))
        basis_2ddict[k][(k // 2 + 0)] *= ((2 ** -k) * comb(k, k // 2 + 0))
        basis_2ddict[k][(k // 2 + 1)] *= ((2 ** -k) * comb(k, k // 2 + 1))

    #   k   ()//2    ()+1    ()//2   ()*2   a_idx   b_idx   k//2   a_idx // 2  b_idx // 2
    #   3     1        2       1       2      2       1       1       1           0
    #   5     2        3       1       2      2       3       2       1           1
    #   7     3        4       2       4      4       3       3       2           1
    for k in range(1, K + 1, 2):
        basis_2ddict[k][(k // 2 + 0)] *= ((2 ** -k) * comb(k, k // 2 + 0))
        basis_2ddict[k][(k // 2 + 1)] *= ((2 ** -k) * comb(k, k // 2 + 1))

    polys = [eye] + [
        base
        for kl in range(1, K + 1)
        for base in basis_2ddict[kl].values()
    ]
    pe = torch.stack(polys, dim=0)  # (K//2*5+1) x n x n
    return pe
