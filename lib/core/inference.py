# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from lib.data.transforms import FLIP_CONFIG

def get_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    outputs = []
    heatmaps = []
    tags = []

    outputs.append(model(image))
    heatmaps.append(outputs[-1][:, :cfg.get("data", "num_joints")])
    tags.append(outputs[-1][:, cfg.get("data", "num_joints"):])

    if with_flip:
        outputs.append(model(torch.flip(image, [3])))
        outputs[-1] = torch.flip(outputs[-1], [3])
        heatmaps.append(outputs[-1][:, :cfg.get("data", "num_joints")])
        tags.append(outputs[-1][:, cfg.get("data", "num_joints"):])
        flip_index = FLIP_CONFIG['COCO_WITH_CENTER'] \
            if cfg.get("data", "with_center") else FLIP_CONFIG['COCO']
        heatmaps[-1] = heatmaps[-1][:, flip_index, :, :]
        if cfg.get("model", "tag_per_joint"):
            tags[-1] = tags[-1][:, flip_index, :, :]

    if cfg.get("data", "with_center") and cfg.get("test", "ignore_center"):
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    return outputs, heatmaps, tags


def get_multi_stage_outputs(
        cfg, model, image, project2image=False, size_projected=None
):
    # outputs = []
    tags = []

    output = model(image)
    offset_feat = cfg.get("data", "num_joints") if cfg.get("loss", "with_heatmaps_loss") else 0

    heatmap = output[:cfg.get("data", "num_joints")]
    tag = output[:, offset_feat:]

    if cfg.get("data", "with_center") and cfg.get("test", "ignore_center"):
        heatmap = heatmap[:, :-1]
        tags = tags[:, :-1]

    if project2image and size_projected:
        heatmap = torch.nn.functional.interpolate(
                heatmap,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )

        tag = torch.nn.functional.interpolate(
                tag,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False)

    return output, heatmap, tag


def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):
    if scale_factor == 1 or len(cfg.get("test", "scale_factor")) == 1:
        if final_heatmaps is not None and not cfg.get("test", "project2image"):
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.get("test", "project2image"):
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list
