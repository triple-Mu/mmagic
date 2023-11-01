# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.models.utils import fuse_conv_bn, generation_init_weights
from mmagic.registry import MODELS

DEFAULT_ACT = nn.SiLU


class Focus(nn.Module):

    @staticmethod
    def forward(x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        s1 = x[..., ::2, ::2]
        s2 = x[..., 1::2, ::2]
        s3 = x[..., ::2, 1::2]
        s4 = x[..., 1::2, 1::2]
        return torch.cat([s1, s2, s3, s4], 1)


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
    default_act = DEFAULT_ACT()

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
        super().__init__()
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
            self.rbr_dense = ConvBNAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.rbr_1x1 = ConvBNAct(
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

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1) -> Tensor:
        if kernel1x1 is None:
            device = next(self.parameters()).device
            return torch.tensor(0, device=device)
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self,
                        branch: Optional[nn.Module]) -> Tuple[Tensor, Tensor]:
        if branch is None:
            device = next(self.parameters()).device
            return torch.tensor(
                0, device=device), torch.tensor(
                    0, device=device)
        if isinstance(branch, ConvBNAct):
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
                                           dtype=torch.float32,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
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
class RepVGGUnetV2(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_width: int = 12) -> None:
        super(RepVGGUnetV2, self).__init__()
        self.encoder = Encoder(in_channels, base_width)
        self.decoder = Decoder(base_width, out_channels=out_channels)
        generation_init_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        b5, b4, b3, b2, b1, b0 = self.encoder(x)
        output = self.decoder(b5, b4, b3, b2, b1, b0)
        output = x + output
        return output


class Encoder(nn.Module):

    def __init__(self, in_channels: int = 3, base_width: int = 16) -> None:
        super().__init__()

        # [1,3,1024,1024] -> [1,16,512,512]
        self.block0 = RepVGGBlock(
            in_channels, base_width, kernel_size=3, stride=2, padding=1)

        # [1,16,512,512] -> [1,16,512,512]
        self.block1 = RepVGGBlock(
            base_width, base_width, kernel_size=3, padding=1)

        # [1,16,512,512] -> [1,32,256,256]
        self.down1 = RepVGGBlock(
            base_width, base_width * 2, kernel_size=3, stride=2, padding=1)

        # [1,32,256,256] -> [1,32,256,256]
        self.block2 = RepVGGBlock(
            base_width * 2, base_width * 2, kernel_size=3, padding=1)

        # [1,32,256,256] -> [1,64,128,128]
        self.down2 = RepVGGBlock(
            base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1)

        # [1,64,128,128] -> [1,64,128,128]
        self.block3 = RepVGGBlock(
            base_width * 4, base_width * 4, kernel_size=3, padding=1)

        # [1,64,128,128] -> [1,128,64,64]
        self.down3 = RepVGGBlock(
            base_width * 4, base_width * 8, kernel_size=3, stride=2, padding=1)

        # [1,128,64,64] -> [1,128,64,64]
        self.block4 = RepVGGBlock(
            base_width * 8, base_width * 8, kernel_size=3, padding=1)

        # [1,128,64,64] -> [1,256,32,32]
        self.down4 = RepVGGBlock(
            base_width * 8,
            base_width * 16,
            kernel_size=3,
            stride=2,
            padding=1)

        # [1,256,32,32] -> [1,256,32,32]
        self.block5 = RepVGGBlock(
            base_width * 16,
            base_width * 16,
            kernel_size=3,
            padding=1,
            use_se=True)

    def forward(self, x: Tensor) -> Tuple:
        block0 = self.block0(x)

        block1 = self.block1(block0)
        down1 = self.down1(block1)

        block2 = self.block2(down1)
        down2 = self.down2(block2)

        block3 = self.block3(down2)
        down3 = self.down3(block3)

        block4 = self.block4(down3)
        down4 = self.down4(block4)

        block5 = self.block5(down4)

        return block5, block4, block3, block2, block1, block0


class Decoder(nn.Module):

    def __init__(self, base_width: int = 16, out_channels: int = 3) -> None:
        super().__init__()

        # [1,192,32,32]
        # [1,96,64,64]
        # [1,48,128,128]
        # [1,24,256,256]
        # [1,12,512,512]
        # [1,12,512,512]

        # [1,256,32,32] -> [1,256,32,32]
        self.block1 = RepVGGBlock(
            base_width * 16,
            base_width * 16,
            kernel_size=3,
            padding=1,
            use_se=True)

        # [1,512,32,32] -> [1,128,64,64] -> [1,128,64,64]
        self.up1 = nn.Sequential(
            nn.PixelShuffle(2),
            RepVGGBlock(
                base_width * 8, base_width * 8, kernel_size=3, padding=1))

        # [1,128,64,64] -> [1,128,64,64]
        self.block2 = RepVGGBlock(
            base_width * 8, base_width * 8, kernel_size=3, padding=1)

        # [1,256,64,64] -> [1,64,128,128] -> [1,64,128,128]
        self.up2 = nn.Sequential(
            nn.PixelShuffle(2),
            RepVGGBlock(
                base_width * 4, base_width * 4, kernel_size=3, padding=1))

        # [1,64,128,128] -> [1,64,128,128]
        self.block3 = RepVGGBlock(
            base_width * 4, base_width * 4, kernel_size=3, padding=1)

        # [1,128,128,128] -> [1,32,256,256] -> [1,32,256,256]
        self.up3 = nn.Sequential(
            nn.PixelShuffle(2),
            RepVGGBlock(
                base_width * 2, base_width * 2, kernel_size=3, padding=1))

        # [1,32,256,256] -> [1,32,256,256]
        self.block4 = nn.Sequential(
            RepVGGBlock(
                base_width * 2, base_width * 2, kernel_size=3, padding=1))

        # [1,64,256,256] -> [1,16,512,512] -> [1,16,512,512]
        self.up4 = nn.Sequential(
            nn.PixelShuffle(2),
            RepVGGBlock(base_width, base_width, kernel_size=3, padding=1))

        # [1,16,512,512] -> [1,16,512,512]
        self.block5 = nn.Sequential(
            RepVGGBlock(base_width, base_width, kernel_size=3, padding=1))

        # [1,32,512,512] -> [1,8,1024,1024] -> [1,8,1024,1024]
        self.up5 = nn.Sequential(
            nn.PixelShuffle(2),
            RepVGGBlock(
                base_width // 2, base_width // 2, kernel_size=3, padding=1))

        self.up6 = nn.PixelShuffle(2)
        self.end = nn.Sequential(
            nn.Conv2d(
                base_width // 2,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0), nn.BatchNorm2d(out_channels),
            DEFAULT_ACT(inplace=True))

    def forward(self, b5: Tensor, b4: Tensor, b3: Tensor, b2: Tensor,
                b1: Tensor, b0: Tensor) -> Tensor:
        # b5: [1,192,32,32]
        # b4: [1,96,64,64]
        # b3: [1,48,128,128]
        # b2: [1,24,256,256]
        # b1: [1,12,512,512]
        # b0: [1,12,512,512]

        block1 = self.block1(b5)
        up1 = self.up1(torch.cat([b5, block1], dim=1))

        block2 = self.block2(up1)
        up2 = self.up2(torch.cat([b4, block2], dim=1))

        block3 = self.block3(up2)
        up3 = self.up3(torch.cat([b3, block3], dim=1))

        block4 = self.block4(up3)
        up4 = self.up4(torch.cat([b2, block4], dim=1))

        block5 = self.block5(up4)
        up5 = self.up5(torch.cat([b1, block5], dim=1))

        up6 = self.up6(b0)
        end = self.end(up5)
        return end + up6


if __name__ == '__main__':
    net = RepVGGUnetV2(in_channels=3, out_channels=3, base_width=12)
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

    # exit(1)
    torch.onnx.export(
        net,
        x,
        'unet-repvgg-se.onnx',
        input_names=['image'],
        output_names=['new_image'])
