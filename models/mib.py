import numpy as np
import torch
import logging
from types import MethodType
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
import time
from utils.args import *
import copy
import json
import math
import torch.nn as nn
from models.utils.loss import UnbiasedKnowledgeDistillationLoss,UnbiasedCrossEntropy
#modified from https://github.com/arthurdouillard/CVPR2021_PLOP

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument(
        "--pod",default="local",type=str,choices=["spatial", "local", "global"]
    )
    parser.add_argument("--loss_kd", default=10, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    return parser




class Mib(ContinualModel):
    NAME = 'mib'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Mib, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=args.alpha)
        self.loss_kd = args.loss_kd
        
        self.classif_adaptive_factor = False
        self.classif_adaptive_min_factor = 0.0
        
        self.nb_current_classes = 1
        self.old_classes = 1
        self.model_old = None
        self.ret_intermediate = False
        
        def new_features(self, x, ret_intermediate = False):
        # Encode
        
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.conv2(x1)
            #x1 = self.input_layer(x)
            x2 = self.residual_conv_1(x1)
            x3 = self.residual_conv_2(x2)
            # Bridge
            x4 = self.bridge(x3)
            # Decode
            x4 = self.upsample_1(x4)
            x5 = torch.cat([x4, x3], dim=1)

            x6 = self.up_residual_conv1(x5)

            x6 = self.upsample_2(x6)
            x7 = torch.cat([x6, x2], dim=1)

            x8 = self.up_residual_conv2(x7)

            x8 = self.upsample_3(x8)
            x9 = torch.cat([x8, x1], dim=1)

            x10 = self.up_residual_conv3(x9)

            #output = self.output_layer(x10)
            if ret_intermediate:
                midfeatures = [x1,x2,x3,x4,x5,
                             x6,x7,x8,x9,x10]
                return x10, midfeatures
            return x10
        self.net.features = MethodType(new_features, self.net)
        
        
    def begin_task(self, dataset):
        ########3确定当前class数#####
        #0:background
        self.old_classes = self.nb_current_classes
        self.nb_new_classes = dataset.N_CLASSES_PER_TASK[self.current_task]
        self.nb_current_classes += self.nb_new_classes
        logger = logging.getLogger('base')
        if self.current_task > 0:
            self.loss = UnbiasedCrossEntropy(
                old_cl=self.old_classes, reduction='none'
            )

    def end_task(self, dataset):
        self.model_old = copy.deepcopy(self.net)
        self.current_task += 1
        
        
    def observe(self, inputs, labels, not_aug_inputs):


        # now compute the grad on the current data
        self.opt.zero_grad()
        classif_adaptive_factor = 1.0
        
        x10 = self.net.features(inputs, ret_intermediate=False)
        outputs = self.net.logits(x10,self.current_task)
        loss = self.loss(outputs, labels)
        loss = classif_adaptive_factor*loss
        loss = loss.mean()
        
        lkd = 0.0
        
        if self.model_old is not None and self.current_task > 0: #kd
            x10_old = self.model_old.features(inputs, ret_intermediate=False)
            outputs_old = self.model_old.logits(x10_old, self.current_task-1)
            lkd = self.loss_kd * self.lkd_loss(outputs[:,:self.nb_current_classes], outputs_old[:,:self.old_classes], mask=None)
            lkd = torch.mean(lkd)
            

        
        
        loss_tot = loss + lkd
        loss_tot.backward()
        self.opt.step()

        return loss_tot.item()
    

