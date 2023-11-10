# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.registry import MODELS

DEFAULT_ACT = nn.GELU


class LayerNorm2dCFirst(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        _, c, _, _ = x.shape
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, c, 1, 1) * x + self.bias.view(1, c, 1, 1)


class ConvBNAct(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvLNAct(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.GELU()):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.ln = LayerNorm2dCFirst(out_channels)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.ln(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-
    mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance
        variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Applies forward pass using activation on convolutions of the input,
        optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


@MODELS.register_module(force=True)
class ConvSRNet(BaseModule):

    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 3,
                 base_channels: int = 16,
                 num_layers: int = 8):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels

        # 3 -> 16 -> 32 -> 64
        self.stem = nn.Sequential(
            ConvLNAct(input_channels, base_channels, 3, 1, 1),
            ConvLNAct(base_channels, base_channels * 2, 3, 1, 1),
            ConvLNAct(base_channels * 2, base_channels * 4, 3, 2, 1),
            ConvLNAct(base_channels * 4, base_channels * 8, 3, 1, 1),
            ConvLNAct(base_channels * 8, base_channels * 16, 3, 2, 1),
        )

        self.cnns = nn.Sequential(*[
            ConvLNAct(base_channels * 16, base_channels * 16, 3, 1, 1)
            for _ in range(num_layers)
        ])

        self.tail = nn.Sequential(
            ConvLNAct(base_channels * 16, output_channels * 16, 1, 1, 0),
            nn.PixelShuffle(4))

    def forward(self, x):
        stem = self.stem(x)
        y = self.cnns(stem)
        y = self.tail(y + stem)
        return x + y


if __name__ == '__main__':
    net = ConvSRNet(base_channels=8, num_layers=8)
    x = torch.randn(1, 3, 1024, 1024)
    y = net(x)
    print(y.shape)

    mb = sum([p.numel() * p.element_size()
              for p in net.parameters()]) / 1024 / 1024
    print(mb)

    torch.onnx.export(
        net,
        x,
        'conv_sr_net.onnx',
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
