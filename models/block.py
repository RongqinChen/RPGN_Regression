import torch
import torch.nn as nn


class RP_RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """

    def __init__(self, in_channels, out_channels, mlp_depth):
        super().__init__()
        self.mlp1 = MlpBlock(in_channels, out_channels, mlp_depth)
        self.mlp2 = MlpBlock(in_channels, out_channels, mlp_depth)
        self.skip = SkipConnection(in_channels + out_channels, out_channels)
        self.update = MlpBlock(out_channels, out_channels, 2)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        mult = torch.matmul(mlp1, mlp2) / inputs.shape[-1]
        mult = torch.sqrt(torch.relu(mult)) - torch.sqrt(torch.relu(-mult))
        skip = self.skip(in1=inputs, in2=mult)
        out = self.update(skip) + skip
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """

    def __init__(self, in_channels, out_channels, mlp_depth, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for _ in range(mlp_depth):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_channels = out_channels

    def forward(self, inputs):
        out = self.convs[0](inputs)
        for idx in range(1, len(self.convs)):
            out = self.activation(out)
            out = self.convs[idx](out)

        return out


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_channels: d1+d2
    :param out_channels: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
