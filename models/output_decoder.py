"""
Different output decoders for different datasets/tasks.
"""

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense import Linear


class GraphClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphClassification, self).__init__()
        self.classifier = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.ELU(),
            Linear(in_channels // 2, out_channels)
        )
        self.reset_parameters()

    def weights_init(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.classifier.apply(self.weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class GraphRegression(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphRegression, self).__init__()
        self.regressor = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.ReLU(),
            Linear(in_channels // 2, in_channels // 4), nn.ReLU(),
            Linear(in_channels // 4, out_channels)
        )
        self.reset_parameters()

    def weights_init(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.regressor.apply(self.weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.regressor(x)


class NodeClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(NodeClassification, self).__init__()
        self.classifier = nn.Sequential(
            Linear(in_channels, in_channels // 2), nn.ELU(),
            Linear(in_channels // 2, out_channels)
        )
        self.reset_parameters()

    def weights_init(self, m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.classifier.apply(self.weights_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)
