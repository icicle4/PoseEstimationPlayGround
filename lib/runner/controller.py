#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some runner used by main runner.


import os
from lib.tools.util.logger import Logger as Log
import torch
from lib.utils.utils import save_checkpoint


class Controller(object):

    @staticmethod
    def init(runner):
        runner.runner_state['iters'] = 0
        runner.runner_state['last_iters'] = 0
        runner.runner_state['epoch'] = 0
        runner.runner_state['last_epoch'] = 0
        runner.runner_state['performance'] = 0
        runner.runner_state['val_loss'] = 0
        runner.runner_state['max_performance'] = 0
        runner.runner_state['min_val_loss'] = 0

    @staticmethod
    def train(runner):
        Log.info('Training start...')
        configer = runner.configer

        best_perf = -1
        best_model = False
        last_epoch = -1
        begin_epoch = configer.get("train", "begin_epoch")
        checkpoint_file = os.path.join(
            configer.get("final_output_dir"), 'checkpoint.pth.tar')
        if configer.get("auto_resume") and os.path.exists(checkpoint_file):
            Log.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            runner.pose_net.load_state_dict(checkpoint['state_dict'])
            runner.optimizer.load_state_dict(checkpoint['optimizer'])

            Log.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            runner.optimizer,
            configer.get("train", "lr_step"),
            configer.get("train", "lr_factor"),
            last_epoch=last_epoch
        )

        runner.scheduler = lr_scheduler
        for epoch in range(begin_epoch, configer.get("train", "end_epoch")):
            runner.train()
            perf_indicator = epoch
            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            Log.info("=> saving checkpoint to {}".format(configer.get("final_output_dir")))
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    "model": configer.get("model", "name"),
                    "state_dict": runner.pose_net.state_dict(),
                    'best_state_dict': runner.pose_net.module.state_dict(),
                    "perf": perf_indicator,
                    'optimizer': runner.optimizer.state_dict()
                }, best_model, configer.get("final_output_dir")
            )

        final_model_state_file = os.path.join(
            configer.get("final_output_dir"), 'final_state{}.pth.tar'.format(0)
        )
        Log.info("saving final model state to {}".format(final_model_state_file))
        torch.save(runner.pose_net.state_dict(), final_model_state_file)
        runner.writer_dict["writer"].close()
        Log.info('Training end...')

    @staticmethod
    def test(runner):
        Log.info('Testing start...')
        runner.test()
        Log.info('Testing end...')
