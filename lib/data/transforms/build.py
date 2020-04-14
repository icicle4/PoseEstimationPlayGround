# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.get("data", "output_size"), (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    if is_train:
        max_rotation = cfg.get("data", "max_rotation")
        min_scale = cfg.get("data", "min_scale")
        max_scale = cfg.get("data", "max_scale")
        max_translate = cfg.get("data", "max_translate")
        input_size = cfg.get("data", "input_size")
        output_size = cfg.get("data", "output_size")
        flip = cfg.get("data", "flip")
        scale_type = cfg.get("data", "scale_type")
    else:
        scale_type = cfg.get("data", "scale_type")
        max_rotation = 0
        min_scale = 1
        max_scale = 1
        max_translate = 0
        input_size = 512
        output_size = [512]
        flip = 0

    # coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    # if cfg.DATASET.WITH_CENTER:
        # coco_flip_index.append(17)
    if cfg.get("data", "with_center"):
        coco_flip_index = FLIP_CONFIG['COCO_WITH_CENTER']
    else:
        coco_flip_index = FLIP_CONFIG['COCO']

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate,
                scale_aware_sigma= cfg.get("data", "scale_aware_sigma")
            ),
            T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
