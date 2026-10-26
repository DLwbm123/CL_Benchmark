# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import csv
import os
import sys
from typing import Dict, Any
from argparse import Namespace
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path, get_timestamp
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']

#################Basic logger for record########
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

#############Tensorboard logger for visualization###########
import torch
import torchvision
import torch.nn as nn

class Visualization(nn.Module):
    def __init__(self, args: Namespace, savename = "/logs"):
        super().__init__()
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'summary'))

    def save_image(self, image, tag, epoch):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        self.writer.add_image(tag, grid, epoch)

    def save_scalar(self, scalar, tag, epoch):
        pass

    def forward(self, inputs, outputs, labels, others, epoch):
        pass

#############CSV logger for print CSV###########
class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, args: Namespace, savename = "/cl_results.csv") -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.bwt = None
        self.rmac = None  # Normalized RMA requires independent-task references.
        self.rma = None
        self.args = args
        #self.savepath = base_path() #self.set_path() #base_path() + "results/"
        self.savepath = base_path() + "results/"
        self.savename = savename

    def set_path(self):
        create_if_not_exists(base_path() + "results/")
        create_if_not_exists(base_path() + "results/" + self.dataset)
        create_if_not_exists(base_path() + "results/" +
                             "/" + self.dataset + "/" + self.model)
        return base_path() + "results/" + "/" + self.dataset + "/" + self.model


    def add_fwt(self, results, accs):
        self.fwt = forward_transfer(results, accs)

    def add_bwt(self, results):
        self.bwt = backward_transfer(results)

    def add_mean_dice(self, results):
        self.mean_dice = np.mean(results[-1])

    def add_rma(self, results):
        self.rma = restricted_modelability(results)

    def add_rmac(self, results1, results2):
        self.rmac = restricted_modelability_corrected(results1, results2)

    def add_ms(self, results):
        self.ms = model_size_efficiency(results)

    def add_sss(self, results, ALL_DATA_SIZE):
        self.sss = samples_storage_size_efficiency(results, ALL_DATA_SIZE)

    def add_fwt_asd(self, results, accs):
        self.fwt_asd = forward_transfer(results, accs)

    def add_bwt_asd(self, results):
        self.bwt_asd = backward_transfer(results)

    def add_mean_asd(self, results):
        self.mean_asd = np.mean(results[-1])

    def add_fwt_precision(self, results, accs):
        self.fwt_precision = forward_transfer(results, accs)

    def add_bwt_precision(self, results):
        self.bwt_precision = backward_transfer(results)

    def add_mean_precision(self, results):
        self.mean_precision = np.mean(results[-1])

    def add_fwt_recall(self, results, accs):
        self.fwt_recall = forward_transfer(results, accs)

    def add_bwt_recall(self, results):
        self.bwt_recall = backward_transfer(results)

    def add_mean_recall(self, results):
        self.mean_recall = np.mean(results[-1])

    def add_whs(self, results):
        self.whs = np.mean(results)


    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        args = dict(args)  # Logging must not mutate the live training Namespace.
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []

        args['mean_dice'] = self.mean_dice
        new_cols.append('mean_dice')

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['restricted_model_ability'] = self.rmac
        new_cols.append('restricted_model_ability')

        # add_rma historically computes mean diagonal Dice, not a ratio.
        # Keep the missing normalized RMA empty instead of inventing a value.
        args['new_task_dice'] = self.rma
        new_cols.append('new_task_dice')

        args['model_size_efficiency'] = self.ms
        new_cols.append('model_size_efficiency')

        args['samples_storage_size_efficiency'] = self.sss
        new_cols.append('samples_storage_size_efficiency')

        if self.dataset == 'seq-mmwhs':
            args['wholeheart'] = self.whs
            new_cols.append('wholeheart')

        args['mean_asd'] = self.mean_asd
        new_cols.append('mean_asd')

        args['forward_transfer_asd'] = self.fwt_asd
        new_cols.append('forward_transfer_asd')

        args['backward_transfer_asd'] = self.bwt_asd
        new_cols.append('backward_transfer_asd')

        args['mean_precision'] = self.mean_precision
        new_cols.append('mean_precision')

        args['forward_transfer_precision'] = self.fwt_precision
        new_cols.append('forward_transfer_precision')

        args['backward_transfer_precision'] = self.bwt_precision
        new_cols.append('backward_transfer_precision')

        args['mean_recall'] = self.mean_recall
        new_cols.append('mean_recall')

        args['forward_transfer_recall'] = self.fwt_recall
        new_cols.append('forward_transfer_recall')

        args['backward_transfer_recall'] = self.bwt_recall
        new_cols.append('backward_transfer_recall')

        columns = new_cols + columns


        write_headers = False
        create_if_not_exists(self.savepath)
        path = self.savepath + self.savename
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)



