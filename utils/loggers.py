# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path, get_timestamp
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


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
        
        

class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.bwt = None
        self.savepath = self.set_path()

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


    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
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
        path = self.savepath + "/cl_results.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)



