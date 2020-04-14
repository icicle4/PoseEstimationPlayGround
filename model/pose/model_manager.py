#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Pose Model for pose detection.

from model.pose.nets.hr_pose import get_pose_net

from lib.core.loss import LossFactory


POSE_MODEL_DICT = {
    'hr_pose': get_pose_net
}

Loss_DICT = {
    'hr_pose': LossFactory
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_multi_pose_model(self):
        model = POSE_MODEL_DICT[self.configer.get("model", "name")]
        return model

    def get_pose_loss(self):
        return Loss_DICT[self.configer.get("model", "name")](self.configer)
