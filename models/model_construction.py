"""
Model construction.
"""

from argparse import ArgumentParser
from torch import nn


from models.network import SpecDistGNN


def make_model(
    args: ArgumentParser,
    node_encoder: nn.Module,
    edge_encoder: nn.Module,
) -> nn.Module:
    r"""Make GNN model given input parameters.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        node_encoder (nn.Module): Node feature input encoder.
        edge_encoder (nn.Module): Edge feature input encoder.
    """

    gnn = SpecDistGNN(
        args.pe_len,
        node_encoder, edge_encoder,
        args.hidden_channels,
        args.num_layers,
        args.mlp_depth,
        args.norm_type,
        args.graph_pool,
        args.drop_prob,
        args.jumping_knowledge,
        args.task_type,
        args.num_task,
    )

    return gnn
