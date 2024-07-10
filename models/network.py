"""
SpecDistGNN framework.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from .mlp import MLP
from .norms import Normalization
from .block import RP_RegularBlock
from .diagonal_pool import diag_offdiag_avgpool, diag_offdiag_sumpool
from .output_decoder import GraphClassification, GraphRegression, NodeClassification


class SpecDistGNN(nn.Module):
    r"""An implementation of SpecDistGNN.
    Args:
        pe_len (int): the length of positional embedding.
        node_encoder (nn.Module): initial node feature encoding.
        edge_encoder (nn.Module): initial edge feature encoding.
        hidden_channels (int): hidden_channels
        num_layers (int): the number of layers
        mlp_depth (int): the number of layers in each MLP in RPGN
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        drop_prob (float, optional): dropout rate.
        graph_pool (str): Method of graph pooling, last,concat,max or sum.
        jumping_knowledge (str, optional): Method of jumping knowledge, last,concat,max or sum.
        task_type (str): Task type, graph_classification, graph_regression, node_classification.
    """

    def __init__(
        self,
        pe_len: int,
        node_encoder: nn.Module,
        edge_encoder: nn.Module,
        hidden_channels: int,
        num_layers: int,
        mlp_depth: int,
        norm_type: Optional[str] = "Batch",
        graph_pool: nn.Module = None,
        drop_prob: Optional[float] = 0.0,
        jumping_knowledge: bool = False,
        task_type: str = "graph_classfication",
        num_tasks: int = 1,
    ):

        super(SpecDistGNN, self).__init__()
        self.norm_type = norm_type
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(drop_prob)
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder

        # First part - sequential mlp blocks
        block_channels = [hidden_channels] * num_layers
        block_channels.insert(0, pe_len + node_encoder.out_channels + edge_encoder.out_channels)
        self.block_channels = block_channels

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for in_channels, out_channels in zip(block_channels[: -1], block_channels[1:]):
            mlp_block = RP_RegularBlock(in_channels, out_channels, mlp_depth)
            self.blocks.append(mlp_block)
            self.norms.append(Normalization(out_channels, "Batch2d"))

        self.graph_pool = diag_offdiag_avgpool if graph_pool == "mean" else diag_offdiag_sumpool

        out_channels = [channels * 2 for channels in block_channels[1:]]
        self.jk_mlp = MLP(sum(out_channels), out_channels[-1]) if jumping_knowledge == "concat" else None
        self.act = nn.ReLU()
        if task_type == "graph_classification":
            self.out_decoder = GraphClassification(out_channels[-1], num_tasks)
        elif task_type == "graph_regression":
            self.out_decoder = GraphRegression(out_channels[-1], num_tasks)
        elif task_type == "node_classification":
            self.out_decoder = NodeClassification(out_channels[-1], num_tasks)
        else:
            raise NotImplementedError()

        self.reset_parameters()

    def weights_init(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.blocks.apply(self.weights_init)
        self.norms.apply(self.weights_init)
        if self.jk_mlp is not None:
            self.jk_mlp.apply(self.weights_init)
        self.out_decoder.apply(self.weights_init)

    def forward(self, data: Data) -> Tensor:
        hs = [data["batch_full_pe"]]
        hs += [self.node_encoder(data)] if self.node_encoder is not None else []
        hs += [self.edge_encoder(data)] if self.edge_encoder is not None else []

        h = torch.cat(hs, 1)
        h_list = []
        for block, norm in zip(self.blocks, self.norms):
            h = block(h)
            h = norm(h)
            h = self.dropout(h)
            h = self.act(h)
            h_list.append(h)

        if self.jk_mlp is not None:
            z_list = [self.graph_pool(h) for h in h_list]
            z = self.jk_mlp(torch.cat(z_list, 1))
        else:
            z = self.graph_pool(h_list[-1])

        out = self.out_decoder(z)
        return out
