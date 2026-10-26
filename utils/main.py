# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import importlib
import os
import sys
import torch
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_experiment_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.training import train,test
from utils.conf import set_random_seed,get_name
from utils.loggers import setup_logger
from utils import create_if_not_exists
from utils.visualization import infer, save_im_and_pred
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


def main(argv=None):
    parser = ArgumentParser(description='CL_Benchmark', allow_abbrev=False)
    '''parser.add_argument('--model', type=str, default="si",
                        help='Model name.', choices=get_all_models())'''
    add_experiment_args(parser)
    add_management_args(parser)
    args = parser.parse_known_args(argv)[0]
    mod = importlib.import_module('models.' + args.model)


    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args(argv)

    ##########################
    torch.set_num_threads(4)
    if args.seed is not None:
        set_random_seed(args.seed) #Set Seed

    if args.name is None:
        args.name = get_name() #Get experiment name

    ###############Special Network Setting (for some methods)###########
    if args.model == 'mer': setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)

    if dataset.SETTING == 'task-il' or args.model in ['dan']:
        args.multihead = True
    if args.model in ['pnn']:
        args.multihead = False

    ############Set Basic setting###############
    backbone = dataset.get_backbone(args)
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    #set base dir & loggers
    #################################
    base_dir = "./data/results/" + "/" + dataset.NAME + "/" + model.NAME
    args.base_dir = base_dir
    create_if_not_exists(base_dir)
    if args.test_only:
        setup_logger('test', base_dir, 'test_' + args.name, level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('test')
    else:
        setup_logger('base', base_dir, 'train_' + args.name, level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
    #################################


    logger.info("model is {}".format(args.model))
    if args.multihead:
        logger.info("Enable multihead models.")
    logger.info(args)
    #logger.info('# Net parameters:', sum(param.numel() for param in model.net.parameters()))
    if args.input_folder is not None:
        save_im_and_pred(model, dataset, args, logger)
        sys.exit(0)

    if args.test_only:
        test(model, dataset, args, logger)
    else:
        train(model, dataset, args, logger)


if __name__ == '__main__':
    main()
