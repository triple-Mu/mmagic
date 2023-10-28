# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image
from torchvision.transforms.functional import (adjust_brightness,
                                               adjust_saturation, hflip,
                                               rotate, vflip)

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class NAFNetTransform(BaseTransform):

    def __init__(self, keys, use_rotate: bool = True):
        self.keys = keys if isinstance(keys, list) else [keys]
        self.use_rotate = use_rotate

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        aug = random.randint(0, 2)
        if aug == 1:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(adjust_brightness(v, 1))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(adjust_brightness(v, 1))

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(
                            adjust_saturation(v, sat_factor))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(adjust_saturation(v, sat_factor))

        aug = random.randint(0, 8)
        if self.use_rotate and aug == 1:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(vflip(v))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(vflip(v))

        elif aug == 2:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(hflip(v))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(hflip(v))

        elif self.use_rotate and aug == 3:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(rotate(v, 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(rotate(v, 90))

        elif self.use_rotate and aug == 4:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(rotate(v, 90 * 2))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(rotate(v, 90 * 2))

        elif self.use_rotate and aug == 5:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(rotate(v, 90 * 3))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(rotate(v, 90 * 3))

        elif self.use_rotate and aug == 6:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(rotate(vflip(v), 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(rotate(vflip(v), 90))

        elif self.use_rotate and aug == 7:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(rotate(hflip(v), 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(rotate(hflip(v), 90))
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}'

        return repr_str
