# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models.editors.pix2pix import UnetGenerator
from mmagic.registry import MODELS


@MODELS.register_module()
class UNetBaidu(UnetGenerator):

    def forward(self, x):
        return x + self.model(x)
