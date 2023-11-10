# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn

from mmagic.registry import MODELS


class AddOp(nn.Module):

    @staticmethod
    def forward(x1, x2):
        return x1 + x2


class AnchorOp(nn.Module):

    def __init__(self,
                 scaling_factor,
                 in_channels=3,
                 init_weights=True,
                 freeze_weights=True,
                 kernel_size=1,
                 **kwargs):
        super().__init__()

        self.net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(in_channels * scaling_factor**2),
            kernel_size=kernel_size,
            **kwargs)

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2,
                                 num_channels_per_group, kernel_size,
                                 kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2:(ii + 1) * scaling_factor**2,
                       ii % num_channels_per_group, kernel_size // 2,
                       kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)


class QuickSRNetBase(nn.Module):
    # default_act = nn.Hardtanh(min_val=0., max_val=1.)
    default_act = nn.ReLU

    def __init__(self,
                 scaling_factor,
                 num_channels,
                 num_intermediate_layers,
                 use_ito_connection,
                 in_channels=3,
                 out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)

        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor

        else:
            raise NotImplementedError(
                f'1.5 is the only supported non-integer scaling factor. '
                f'Received {scaling_factor}.')

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(3, 3),
                    padding=1),
                self.default_act()
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=(3, 3),
                padding=1),
            self.default_act(),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2**2)
            cl_out_channels = out_channels * (3**2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor**2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(
            in_channels=cl_in_channels,
            out_channels=cl_out_channels,
            kernel_size=cl_kernel_size,
            padding=cl_padding)

        if use_ito_connection:
            self.add_op = AddOp()

            if scaling_factor == 1.5:
                self.anchor = AnchorOp(
                    scaling_factor=3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    freeze_weights=False)
            else:
                self.anchor = AnchorOp(
                    scaling_factor=self.scaling_factor, freeze_weights=False)

        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = self.default_act()

        self.initialize()

    def forward(self, input):
        x = self.cnn(input)

        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)

        x = self.clip_output(x)

        return self.depth_to_space(x)

    def initialize(self):
        for conv_layer in self.cnn:
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels,
                                            conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.

        if not self._use_ito_connection:
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squarred = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out %
                              out_channels) // scaling_factor_squarred
                    self.conv_last.weight[idx_out, idx_in, middle,
                                          middle] += 1.


@MODELS.register_module()
class QuickSRNetSmall(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=2,
            use_ito_connection=False,
            **kwargs)


@MODELS.register_module()
class QuickSRNetMedium(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=5,
            use_ito_connection=False,
            **kwargs)


@MODELS.register_module()
class QuickSRNetLarge(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=64,
            num_intermediate_layers=11,
            use_ito_connection=True,
            **kwargs)


if __name__ == '__main__':
    net = QuickSRNetLarge(4)
    print(net)

    x = torch.randn(1, 3, 256, 256)
    y = net(x)

    # torch.onnx.export(net, x, "quicksrnet_large.onnx", opset_version=11)
