# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator


def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    if cfg.get("data", "scale_aware_sigma"):
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size,
            cfg.get("data", "num_joints"),
            cfg.get("data", "sigma"),
        ) for output_size in cfg.get("data", "output_size")
    ]
    joints_generator = [
        JointsGenerator(
            cfg.get("data", "max_num_people"),
            cfg.get("data", "num_joints"),
            output_size,
            cfg.get("model", "tag_per_joint")
        ) for output_size in cfg.get("data", "output_size")
    ]

    dataset_name = cfg.get("data", "train") if is_train else cfg.get("data", "test")
    dataset = eval(cfg.get("data", "dataset"))(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        joints_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.get("train", "images_per_gpu")
        shuffle = True
    else:
        images_per_gpu = cfg.get("train", "images_per_gpu")
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)
    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.get("workers"),
        pin_memory=cfg.get("pin_memory"),
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None
    dataset = coco(
        cfg.get("data", "root"),
        cfg.get("data", "test"),
        cfg.get("data", "data_format"),
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    return data_loader, dataset
