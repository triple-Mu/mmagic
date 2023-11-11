# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image
from torchvision.transforms.functional import (adjust_brightness,
                                               adjust_saturation,
                                               adjust_sharpness, hflip, rotate,
                                               vflip)

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LowUnetTransform(BaseTransform):

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

        aug = random.randint(0, 2)
        if aug == 1:
            sharpness_factor = random.randint(0, 5)
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(
                            adjust_sharpness(v, sharpness_factor))
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(
                        adjust_sharpness(v, sharpness_factor))

        aug = random.randint(0, 2)
        if aug == 1:
            done = False
            for key in self.keys:
                if isinstance(results[key], list):
                    for i, v in enumerate(results[key]):
                        if isinstance(v, np.ndarray):
                            v = Image.fromarray(v)
                        results[key][i] = np.array(hflip(v))
                        if not done:
                            mask = results.get('mask', None)
                            if mask is not None:
                                mask = Image.fromarray(mask)
                                results['mask'] = np.array(hflip(mask))
                                done = True
                else:
                    v = results[key]
                    if isinstance(v, np.ndarray):
                        v = Image.fromarray(v)
                    results[key] = np.array(hflip(v))
                    if not done:
                        mask = results.get('mask', None)
                        if mask is not None:
                            mask = Image.fromarray(mask)
                            results['mask'] = np.array(hflip(mask))
                            done = True

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}'

        return repr_str


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
        if aug == 1:
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

        elif aug == 3:
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

        elif aug == 4:
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

        elif aug == 5:
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

        elif aug == 6:
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

        elif aug == 7:
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
