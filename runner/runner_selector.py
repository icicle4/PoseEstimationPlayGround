#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from runner.pose.pose_estimator import PoseEstimator
from runner.pose.pose_test import HrPoseTest

class RunnerSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def pose_runner(self):
        if self.configer.get('phase') == 'train':
            return PoseEstimator(self.configer)

        else:
            return HrPoseTest(self.configer)
