import torch
from scipy.special import comb


def compute_bernstein_polynomials(adj: torch.Tensor, pe_power: int):
    K = pe_power
    deg = adj.sum(dim=0)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm_adj = deg_inv_sqrt.view((-1, 1)) * adj * deg_inv_sqrt.view((1, -1))
    eye = torch.eye(adj.size(0))
    adj0 = eye + norm_adj
    adj1 = eye - norm_adj

    base1_list = [eye, adj0 / 2.] + [None] * (K - 1)
    base2_list = [eye, adj1 / 2.] + [None] * (K - 1)
    for k in range(2, K + 1):
        lidx, ridx = k // 2, k - k // 2
        base1_list[k] = base1_list[lidx] @ base1_list[ridx]
        base2_list[k] = base2_list[lidx] @ base2_list[ridx]

    bp_base_list = [base1_list[K - k] @ base2_list[k] for k in range(K + 1)]
    bp_coef_list = [comb(K, k) for k in range(K + 1)]
    basis = [bp_base_list[k] * bp_coef_list[k] for k in range(K + 1)]
    basis = [eye] + basis
    pe = torch.stack(basis, dim=0)  # (K+2) x n x n
    return pe
