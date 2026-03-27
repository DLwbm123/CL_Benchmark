# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np
import os
from datetime import datetime

def get_device(gpu_id: str) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    #print("cuda", torch.cuda.is_available())
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    return torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def get_name() -> str:
    """
    Returns the experiment name where to store the models.
    """
    return 'exp'+'_archived_'+get_timestamp()


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
