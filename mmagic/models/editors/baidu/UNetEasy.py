# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa F401
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.registry import MODELS


@MODELS.register_module(force=True)
class ReconstructiveSubNetwork(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_width: int = 16) -> None:
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(
            base_width, out_channels=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        b5, b4, b3, b2, b1 = self.encoder(x)
        output = self.decoder(b5, b4, b3, b2, b1)
        output = x + output
        return output


class EncoderReconstructive(nn.Module):

    def __init__(self, in_channels: int, base_width: int) -> None:
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(
            nn.Conv2d(
                base_width, base_width, kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(
            nn.Conv2d(
                base_width * 2,
                base_width * 2,
                kernel_size=3,
                stride=2,
                padding=1))
        self.block3 = nn.Sequential(
            nn.Conv2d(
                base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(
            nn.Conv2d(
                base_width * 4,
                base_width * 4,
                kernel_size=3,
                stride=2,
                padding=1))
        self.block4 = nn.Sequential(
            nn.Conv2d(
                base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(
            nn.Conv2d(
                base_width * 8,
                base_width * 8,
                kernel_size=3,
                stride=2,
                padding=1))
        self.block5 = nn.Sequential(
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tuple:
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5, b4, b3, b2, b1


class DecoderReconstructive(nn.Module):

    def __init__(self, base_width: int, out_channels: int = 3) -> None:
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 8, base_width * 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 16), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 16,
                base_width * 8,
                kernel_size=1,
                stride=1,
                padding=0), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 4, base_width * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8,
                base_width * 4,
                kernel_size=1,
                stride=1,
                padding=0), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(
                base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width * 2, base_width * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4,
                base_width * 2,
                kernel_size=1,
                stride=1,
                padding=0), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(
                base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 2, base_width * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 1), nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                base_width, base_width * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.fin_out = nn.Sequential(
            nn.Conv2d(
                base_width, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, b5: Tensor, b4: Tensor, b3: Tensor, b2: Tensor,
                b1: Tensor) -> Tensor:
        up1 = self.up1(b5)
        db1 = self.db1(up1 + b4)

        up2 = self.up2(db1)
        db2 = self.db2(up2 + b3)

        up3 = self.up3(db2)
        db3 = self.db3(up3 + b2)

        up4 = self.up4(db3)
        db4 = self.db4(up4 + b1)

        out = self.fin_out(db4)
        return out


if __name__ == '__main__':
    net = ReconstructiveSubNetwork(
        in_channels=3, out_channels=3, base_width=16)
    net.eval()

    x = torch.rand(1, 3, 1024, 1024)
    y = net(x)
    param_mb = sum(m.numel() * m.element_size()
                   for m in net.parameters()) / (1 << 20)
    print(f'Model size: {param_mb:.2f} MB')
    torch.onnx.export(
        net, x, 'unet.onnx', input_names=['image'], output_names=['new_image'])
