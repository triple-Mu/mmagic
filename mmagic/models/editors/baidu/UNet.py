# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import DefaultScope
from mmengine.model import BaseModule
from mmengine.runner import Runner

from mmagic.models.utils import generation_init_weights
from mmagic.registry import MODELS
# from ...utils import generation_init_weights
# generation_init_weights

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class UnetSkipConnectionBlock(nn.Module):
    """Construct a Unet submodule with skip connections, with the following.

    structure: downsampling - `submodule` - upsampling.

    Args:
        outer_channels (int): Number of channels at the outer conv layer.
        inner_channels (int): Number of channels at the inner conv layer.
        in_channels (int): Number of channels in input images/features. If is
            None, equals to `outer_channels`. Default: None.
        submodule (UnetSkipConnectionBlock): Previously constructed submodule.
            Default: None.
        is_outermost (bool): Whether this module is the outermost module.
            Default: False.
        is_innermost (bool): Whether this module is the innermost module.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
    """

    def __init__(self,
                 outer_channels,
                 inner_channels,
                 in_channels=None,
                 submodule=None,
                 is_outermost=False,
                 is_innermost=False,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 use_shu=False):
        super().__init__()
        # cannot be both outermost and innermost
        assert not (is_outermost and is_innermost), (
            "'is_outermost' and 'is_innermost' cannot be True"
            'at the same time.')
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the unet skip connection block.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        k_conv = 3
        k_deconv = 2
        stride = 2
        padding = 1
        self.use_shu = use_shu
        if in_channels is None:
            in_channels = outer_channels
        down_conv_cfg = dict(type='Conv2d')
        down_norm_cfg = norm_cfg
        down_act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        up_conv_cfg = dict(type='Deconv')
        up_norm_cfg = norm_cfg
        up_act_cfg = dict(type='ReLU')
        up_in_channels = inner_channels * 2
        up_bias = use_bias
        middle = [submodule]
        upper = []

        if is_outermost:
            down_act_cfg = None
            down_norm_cfg = None
            up_bias = True
            up_norm_cfg = None
            upper = [nn.Tanh()]
        elif is_innermost:
            down_norm_cfg = None
            up_in_channels = inner_channels
            middle = []
        else:
            upper = [nn.Dropout(0.5)] if use_dropout else []

        down = [
            ConvModule(
                in_channels=in_channels,
                out_channels=inner_channels,
                kernel_size=k_conv,
                stride=stride,
                padding=padding,
                bias=use_bias,
                conv_cfg=down_conv_cfg,
                norm_cfg=down_norm_cfg,
                act_cfg=down_act_cfg,
                order=('act', 'conv', 'norm'))
        ]
        if self.use_shu:
            if is_outermost:
                up = [
                    nn.Sequential(
                        nn.Conv2d(up_in_channels, up_in_channels * 2, 1, bias=False),
                        nn.PixelShuffle(2),
                        ConvModule(in_channels=up_in_channels // 2, out_channels=outer_channels,
                                   kernel_size=3, stride=1, padding=1,
                                   bias=use_bias, conv_cfg=down_conv_cfg,
                                   norm_cfg=up_norm_cfg, act_cfg=up_act_cfg,
                                   order=('act', 'conv', 'norm')))
                ]
            else:
                up = [
                    nn.Sequential(
                        nn.Conv2d(up_in_channels, outer_channels * 2, 1, bias=False),
                        nn.PixelShuffle(2),
                        ConvModule(in_channels=outer_channels//2, out_channels=outer_channels,
                                   kernel_size=3, stride=1, padding=1,
                                   bias=use_bias, conv_cfg=down_conv_cfg,
                                   norm_cfg=up_norm_cfg, act_cfg=up_act_cfg,
                                   order=('act', 'conv', 'norm')))
                ]
        else:
            # print('@@@')
            up = [
                ConvModule(
                    in_channels=up_in_channels,
                    out_channels=outer_channels,
                    kernel_size=k_deconv,
                    stride=stride,
                    padding=0,
                    bias=up_bias,
                    conv_cfg=up_conv_cfg,
                    norm_cfg=up_norm_cfg,
                    act_cfg=up_act_cfg,
                    order=('act', 'conv', 'norm'))
            ]

        model = down + middle + up + upper

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # print('000', x.shape)
        if self.is_outermost:
            return self.model(x)

        # add skip connections
        return torch.cat([x, self.model(x)], 1)


@MODELS.register_module()
class UNetBaidu(BaseModule):
    """Construct the Unet-based generator from the innermost layer to the
    outermost layer, which is a recursive process.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        num_down (int): Number of downsamplings in Unet. If `num_down` is 8,
            the image with size 256x256 will become 1x1 at the bottleneck.
            Default: 8.
        base_channels (int): Number of channels at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_down=8,
                 base_channels=64,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 use_shu=False,
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__(init_cfg=init_cfg)
        self.use_shu = use_shu
        # We use norm layers in the unet generator.
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"

        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            base_channels * 8,
            base_channels * 8,
            in_channels=None,
            submodule=None,
            norm_cfg=norm_cfg,
            is_innermost=True,
            use_shu=use_shu)
        # add intermediate layers with base_channels * 8 filters
        for _ in range(num_down - 5):
            unet_block = UnetSkipConnectionBlock(
                base_channels * 8,
                base_channels * 8,
                in_channels=None,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout,
                use_shu=use_shu)
        # gradually reduce the number of filters
        # from base_channels * 8 to base_channels
        unet_block = UnetSkipConnectionBlock(
            base_channels * 4,
            base_channels * 8,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            use_shu=use_shu)
        unet_block = UnetSkipConnectionBlock(
            base_channels * 2,
            base_channels * 4,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            use_shu=use_shu)
        unet_block = UnetSkipConnectionBlock(
            base_channels,
            base_channels * 2,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            use_shu=use_shu)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(
            out_channels,
            base_channels,
            in_channels=in_channels,
            submodule=unet_block,
            is_outermost=True,
            norm_cfg=norm_cfg,
            use_shu=use_shu)

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return x + self.model(x)

    def init_weights(self):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        generation_init_weights(
            self, init_type=self.init_type, init_gain=self.init_gain)
        self._is_init = True


if __name__ == '__main__':

    # model = Runner.build_model(dict(
    #     type='UNetBaidu',
    #     in_channels=3,
    #     out_channels=3,
    #     num_down=8,
    #     base_channels=16,
    #     norm_cfg=dict(type='BN'),
    #     use_dropout=True,
    #     use_shu=True,
    # ))
    default_scope = DefaultScope.get_instance(
        '111', scope_name='mmagic')
    model = UNetBaidu(
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=16,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            use_shu=True,

    ).cuda()
    a = torch.zeros((1, 3, 1024, 1024)).cuda()
    res = model(a)
    print(res.shape)



    # model = nn.Sequential(
    #     nn.Conv2d(64, 64 * 2, 1, bias=False),
    #     nn.PixelShuffle(2),
    #     ConvModule(in_channels=64 // 2, out_channels=64 ,
    #                kernel_size=3, stride=1, padding=1,
    #                bias=False, conv_cfg=dict(type='Conv2d'),
    #                norm_cfg=None, act_cfg=None,
    #                order=('act', 'conv', 'norm')))
    # a = torch.zeros((1, 64, 1024, 1024))
    # print(model(a).shape)