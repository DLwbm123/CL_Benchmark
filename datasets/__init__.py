# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.seq_heart import SequentialHeart
from datasets.seq_prostate import SequentialProstate
from datasets.seq_lgeheart import SequentialLGEHeart
from datasets.seq_mmwhs import SequentialMMWHS
from datasets.seq_mmwhs_easy import SequentialMMWHSeasy
from datasets.seq_mnms_domain import SequentialMnMsDomain
from datasets.seq_task_incre import SequentialTaskIncre


NAMES = {
    SequentialHeart.NAME: SequentialHeart,
    SequentialProstate.NAME: SequentialProstate,
    SequentialLGEHeart.NAME: SequentialLGEHeart,
    SequentialMMWHS.NAME: SequentialMMWHS,
    SequentialMMWHSeasy.NAME: SequentialMMWHSeasy,
    SequentialMnMsDomain.NAME: SequentialMnMsDomain,
    SequentialTaskIncre.NAME: SequentialTaskIncre    
}



def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
