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

    def __init__(self, keys):
        self.keys = keys if isinstance(keys, list) else [keys]

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
                        results[key][i] = np.ascontiguousarray(
                            adjust_brightness(v, 1))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(
                        adjust_brightness(v, 1))

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(
                            adjust_saturation(v, sat_factor))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(
                        adjust_saturation(v, sat_factor))

        aug = random.randint(0, 8)
        if aug == 1:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(vflip(v))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(vflip(v))

        elif aug == 2:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(hflip(v))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(hflip(v))

        elif aug == 3:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(rotate(v, 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(rotate(v, 90))

        elif aug == 4:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(
                            rotate(v, 90 * 2))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(rotate(v, 90 * 2))

        elif aug == 5:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(
                            rotate(v, 90 * 3))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(rotate(v, 90 * 3))

        elif aug == 6:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(
                            rotate(vflip(v), 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(rotate(vflip(v), 90))

        elif aug == 7:
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.ascontiguousarray(
                            rotate(hflip(v), 90))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.ascontiguousarray(rotate(hflip(v), 90))
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}'

        return repr_str
