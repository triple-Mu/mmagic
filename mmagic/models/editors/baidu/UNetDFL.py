# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmengine import DefaultScope

from mmagic.models.editors.baidu.UNet import UNetBaidu
from mmagic.registry import MODELS


@MODELS.register_module()
class UNetBaiduDFL(UNetBaidu):
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

    def __init__(
        self,
        in_channels,
        out_channels,
        *args,
        **kwargs,
        # in_channels,
        # out_channels,
        # num_down=8,
        # base_channels=64,
        # norm_cfg=dict(type='BN'),
        # use_dropout=False,
        # init_cfg=dict(type='normal', gain=0.02)
    ):
        proj_weight = np.array([
            -160., -80., -40., -20., -10., 0., 10., 20., 40., 80., 160.
        ]) / 255
        self.reg_max = len(proj_weight)
        out_channels = out_channels * self.reg_max
        super().__init__(in_channels, out_channels, *args, **kwargs)

        proj_weight = torch.from_numpy(proj_weight.reshape(1, -1, 1, 1)).to(
            torch.float32)
        self.d_conv = nn.Conv2d(self.reg_max, 1, 1, 1, 0, bias=False)
        self.d_conv.weight.data = proj_weight
        self.d_conv.weight.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        res = self.model(x)
        b, c, h, w = res.shape
        # print('!!!', res.shape)
        res = res.reshape(b, self.reg_max, h * 3, w).softmax(axis=1)
        res = self.d_conv(res).reshape(b, 3, h, w)
        return x + res

    def init_weights(self):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        pass


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
    default_scope = DefaultScope.get_instance('111', scope_name='mmagic')
    model = UNetBaiduDFL(
        in_channels=3,
        out_channels=3,
        num_down=8,
        base_channels=16,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        # use_shu=True,
    ).cuda()
    a = torch.zeros((1, 3, 1024, 1024)).cuda()
    res = model(a)
    print(res.shape)
