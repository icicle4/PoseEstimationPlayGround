#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


import time
import torch

from model.pose.model_manager import ModelManager
from tensorboardX import SummaryWriter
from collections import defaultdict
from lib.data import make_dataloader
from lib.utils import utils
import logging

import torch.optim as optim


class PoseEstimator(object):
    """
      The class for Pose Estimation. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = utils.AverageMeter()
        self.data_time = utils.AverageMeter()
        self.logger = logging.getLogger("Training")
        self.pose_model_manager = ModelManager(configer)
        self.runner_state = defaultdict(int)
        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.get_multi_pose_model()
        self.pose_net = self.pose_net(self.configer, is_train=True)
        torch.cuda.set_device(0)
        self.pose_net = self.pose_net.cuda(0)

        self.loss_module = self.pose_model_manager.get_pose_loss().cuda()

        self.writer_dict = {
            "writer": SummaryWriter(log_dir=self.configer.get("tb_log_dir")),
            "train_global_steps": 0,
            "valid_global_steps": 0,
        }
        dump_input = torch.rand((1, 3, self.configer.get("data", "input_size"),
                                 self.configer.get("data", "input_size")))
        self.writer_dict["writer"].add_graph(self.pose_net, (dump_input, ))

        self.optimizer = self.get_optimizer()

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.configer.get("train", "lr_step"),
                                                        gamma=self.configer.get("train", "lr_factor"))

        self.train_loader = make_dataloader(self.configer, is_train=True)

    def get_optimizer(self):
        optimizer = None
        if self.configer.get("train", "optimizer") == "sgd":
            optimizer = optim.SGD(
                self.pose_net.parameters(),
                lr = self.configer.get("train", "lr"),
                momentum=self.configer.get("train", "momentum"),
                weight_decay=self.configer.get("train", "wd"),
                nesterov=self.configer.get("train", "nesterov")
            )
        elif self.configer.get("train", "optimizer") == "adam":
            optimizer = optim.Adam(
                self.pose_net.parameters(),
                lr = self.configer.get("train", "lr")
            )
        return optimizer

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1
        self.scheduler.step(self.runner_state['epoch'])

        heatmaps_loss_meter = utils.AverageMeter()
        push_loss_meter = utils.AverageMeter()
        pull_loss_meter = utils.AverageMeter()

        for i, (images, heatmaps, masks, joints) in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)

            outputs = self.pose_net(images)

            heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
            masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
            joints = list(map(lambda x: x.cuda(non_blocking=True), joints))

            heatmaps_losses, push_losses, pull_losses = self.loss_module(
                outputs, heatmaps, masks, joints
                )
            loss = 0
            heatmaps_loss = heatmaps_losses[0].mean(dim=0)
            heatmaps_loss_meter.update(
                heatmaps_loss.item(), images.size(0)
            )
            loss = loss + heatmaps_loss
            push_loss = push_losses[0].mean(dim=0)
            push_loss_meter.update(
                push_loss.item(), images.size(0)
            )
            loss = loss + push_loss
            pull_loss = pull_losses[0].mean(dim=0)
            pull_loss_meter.update(
                pull_loss.item(), images.size(0)
            )
            loss = loss + pull_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.configer.get('solver', 'display_iter') == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{heatmaps_loss}{push_loss}{pull_loss}'.format(
                    self.runner_state['epoch'], i, len(self.train_loader),
                    batch_time=self.batch_time,
                    speed=images.size(0) / self.batch_time.val,
                    data_time=self.data_time,
                    heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                    push_loss=_get_loss_info(push_loss_meter, 'push'),
                    pull_loss=_get_loss_info(pull_loss_meter, 'pull')
                )

                self.logger.info(msg)

                self.batch_time.reset()
                self.data_time.reset()

                writer = self.writer_dict["writer"]
                global_steps = self.writer_dict["train_global_steps"]

                writer.add_scalar(
                    "train_heatmaps_loss",
                    heatmaps_loss_meter.val,
                    global_steps
                )

                writer.add_scalar(
                    "train_push_loss",
                    push_loss_meter.val,
                    global_steps
                )

                writer.add_scalar(
                    "train_pull_loss",
                    pull_loss_meter.val,
                    global_steps
                )

                self.writer_dict["train_global_steps"] = global_steps + 1


def _get_loss_info(loss_meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=loss_meter
    )


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
