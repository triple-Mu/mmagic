# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.models.utils import fuse_conv_bn, generation_init_weights
from mmagic.registry import MODELS

DEFAULT_ACT = nn.SiLU


def conv_bn(in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            groups: int = 1) -> nn.Module:
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


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


class RepVGGBlock(nn.Module):
    default_act = nn.SiLU()

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 deploy: bool = False,
                 use_se: bool = False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = self.default_act

        if use_se:
            self.se = ChannelAttention(out_channels)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3),
                                           dtype=torch.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


@MODELS.register_module(force=True)
class ReconstructiveSubNetworkRepVGG(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_width: int = 32) -> None:
        super(ReconstructiveSubNetworkRepVGG, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(
            base_width, out_channels=out_channels)
        generation_init_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        b5, b4, b3, b2, b1, b0, b = self.encoder(x)
        output = self.decoder(b5, b4, b3, b2, b1, b0, b)
        output = x + output
        return output


class EncoderReconstructive(nn.Module):

    def __init__(self, in_channels: int, base_width: int) -> None:
        super(EncoderReconstructive, self).__init__()

        self.stem1 = RepVGGBlock(
            in_channels, base_width, kernel_size=3, padding=1)

        self.stem2 = RepVGGBlock(
            base_width, base_width, kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            RepVGGBlock(
                base_width * 1, base_width * 1, kernel_size=3, padding=1))

        self.mp1 = nn.Sequential(
            RepVGGBlock(
                base_width * 1,
                base_width * 2,
                kernel_size=3,
                stride=2,
                padding=1))

        self.block2 = nn.Sequential(
            RepVGGBlock(
                base_width * 2, base_width * 2, kernel_size=3, padding=1))

        self.mp2 = nn.Sequential(
            RepVGGBlock(
                base_width * 2,
                base_width * 4,
                kernel_size=3,
                stride=2,
                padding=1))

        self.block3 = nn.Sequential(
            RepVGGBlock(
                base_width * 4, base_width * 4, kernel_size=3, padding=1))

        self.mp3 = nn.Sequential(
            RepVGGBlock(
                base_width * 4,
                base_width * 8,
                kernel_size=3,
                stride=2,
                padding=1))

        self.block4 = nn.Sequential(
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1))

        self.mp4 = nn.Sequential(
            RepVGGBlock(
                base_width * 8,
                base_width * 8,
                kernel_size=3,
                stride=2,
                padding=1))

        self.block5 = nn.Sequential(
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.Conv2d(
                base_width * 8,
                base_width * 16,
                kernel_size=1,
                padding=0,
                bias=False), nn.BatchNorm2d(base_width * 16),
            DEFAULT_ACT(inplace=True))

    def forward(self, x: Tensor) -> Tuple:
        b = self.stem1(x)
        b0 = self.stem2(b)
        b1 = self.block1(b0)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5, b4, b3, b2, b1, b0, b


class DecoderReconstructive(nn.Module):

    def __init__(self, base_width: int, out_channels: int = 3) -> None:
        super(DecoderReconstructive, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                base_width * 16,
                base_width * 8,
                kernel_size=1,
                padding=0,
                bias=False), nn.BatchNorm2d(base_width * 8),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1))

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 8,
                base_width * 8,
                kernel_size=2,
                stride=2,
                bias=False), nn.BatchNorm2d(base_width * 8),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1))

        self.block2 = nn.Sequential(
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1))

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 8,
                base_width * 4,
                kernel_size=2,
                stride=2,
                bias=False), nn.BatchNorm2d(base_width * 4),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 4, base_width * 4, kernel_size=3, padding=1))

        self.block3 = nn.Sequential(
            RepVGGBlock(
                base_width * 4, base_width * 4, kernel_size=3, padding=1))

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 4,
                base_width * 2,
                kernel_size=2,
                stride=2,
                bias=False), nn.BatchNorm2d(base_width * 2),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 2, base_width * 2, kernel_size=3, padding=1))

        self.block4 = nn.Sequential(
            RepVGGBlock(
                base_width * 2, base_width * 2, kernel_size=3, padding=1))

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 2,
                base_width * 1,
                kernel_size=2,
                stride=2,
                bias=False), nn.BatchNorm2d(base_width * 1),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 1, base_width * 1, kernel_size=3, padding=1))

        self.block5 = nn.Sequential(
            RepVGGBlock(
                base_width * 1, base_width * 1, kernel_size=3, padding=1))

        self.suffix = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 2,
                base_width * 1,
                kernel_size=2,
                stride=2,
                bias=False), nn.BatchNorm2d(base_width * 1),
            DEFAULT_ACT(inplace=True),
            RepVGGBlock(
                base_width * 1, base_width * 1, kernel_size=3, padding=1))

        self.fin_out = nn.Sequential(
            nn.Conv2d(
                base_width * 1,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True), )

    def forward(self, b5: Tensor, b4: Tensor, b3: Tensor, b2: Tensor,
                b1: Tensor, b0: Tensor, b: Tensor) -> Tensor:
        # b5: 1,256,32,32
        # b4: 1,128,64,64
        # b3: 1,64,128,128
        # b2: 1,32,256,256
        # b1: 1,16,512,512

        block1 = self.block1(b5)
        up1 = self.up1(block1)

        block2 = self.block2(up1 + b4)
        up2 = self.up2(block2)

        block3 = self.block3(up2 + b3)
        up3 = self.up3(block3)

        block4 = self.block4(up3 + b2)
        up4 = self.up4(block4)

        block5 = self.block5(up4 + b1)
        concat = torch.cat([block5, b0], dim=1)

        out = self.suffix(concat)
        out = out + b

        out = self.fin_out(out)
        return out


if __name__ == '__main__':
    net = ReconstructiveSubNetworkRepVGG(
        in_channels=3, out_channels=3, base_width=16)
    net.eval()

    x = torch.rand(1, 3, 1024, 1024)
    y1 = net(x)
    for m in net.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()
    net = fuse_conv_bn(net)
    y2 = net(x)
    param_mb = sum(m.numel() * m.element_size()
                   for m in net.parameters()) / (1 << 20)
    print(f'Model size: {param_mb:.2f} MB')

    torch.onnx.export(
        net,
        x,
        'unet-repvgg-se.onnx',
        input_names=['image'],
        output_names=['new_image'])
    # se = SEBlock(256)
    #
    # x = torch.randn(1, 256, 100, 100)
    # y = se(x)
    # print(y.shape)
    #
    # param_mb = sum(m.numel() * m.element_size()
    #                for m in se.parameters()) / (1 << 20)
    # print(f'Model size: {param_mb * 1024 * 1024} B')
