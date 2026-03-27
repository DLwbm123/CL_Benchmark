# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, default="seq-mmwhs",
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, default="sgd",
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--baseline_size', type=str, default="mid",
                        help='Baseline Size', choices=['small','mid','large'])
    parser.add_argument('--name', default=None, #None, entropy
                        help='experiment name.')
    parser.add_argument('--multihead', action='store_true',
                        help='Enable multihead for each task')
    parser.add_argument('--pseudo_label', default=None,
                        help='Enable pseudo labels for class-il seg')
    parser.add_argument('--mib', action='store_true',
                        help='Enable MiB strategy for class-il seg')
    parser.add_argument('--pretrain', default=None,
                        help='pretrain model path')
    parser.add_argument('--test_only', action='store_true',
                        help='only test')
    parser.add_argument('--save_image', action='store_true',
                        help='only test')
    
    #############################basic training settings###############################
    parser.add_argument('--lr', type=float, default=0.008,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_epoch', type=int, default=80,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size.')
    #parser.add_argument('--validation', action='store_true', help='Test on the validation set')
    parser.add_argument('--n_epochs', type=int, default=150,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')
    parser.add_argument('--GPU_ids', type=str, default=0,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--save_freq', type=int, default=-1,
                        help='Save the trained model for each task after xx epochs')
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--label_folder', type=str, default=None)


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, default=32,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=8,
                        help='The batch size of the memory buffer.')