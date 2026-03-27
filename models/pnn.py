# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from types import MethodType
import torch.optim as optim
from torch.optim import SGD
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.conf import get_device
from utils.args import *
from datasets import get_dataset
from backbone.utils.modules import ListModule, AlphaModule
from backbone.ResUnet import ResUnet,resunet32
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def replace_features(backbone):
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
            midfeatures = [x2,x3,x4,x6,x8,x10]
            return x10, midfeatures
        return x10
    
    backbone.features = MethodType(new_features, backbone)
    return backbone
    

def wrap_adaptors(new_backbone, current_task):
    if isinstance(new_backbone, ResUnet):
        filters = [32, 64, 128, 256]
        setattr(new_backbone, 'adaptor' + str(1), nn.Sequential(
            AlphaModule((filters[1] * current_task, 1, 1)),
            nn.Conv2d(filters[1] * current_task, filters[1], 1),
            nn.ReLU()
        ))
        setattr(new_backbone, 'adaptor' + str(2), nn.Sequential(
            AlphaModule((filters[2] * current_task, 1, 1)),
            nn.Conv2d(filters[2] * current_task, filters[2], 1),
            nn.ReLU()
        ))
        setattr(new_backbone, 'adaptorbridge', nn.Sequential(
            AlphaModule((filters[3] * current_task, 1, 1)),
            nn.Conv2d(filters[3] * current_task, filters[3], 1),
            nn.ReLU()
        ))
        setattr(new_backbone, 'adaptor' + str(3), nn.Sequential(
            AlphaModule((filters[2] * current_task, 1, 1)),
            nn.Conv2d(filters[2] * current_task, filters[2], 1),
            nn.ReLU()
        ))
        setattr(new_backbone, 'adaptor' + str(4), nn.Sequential(
            AlphaModule((filters[1] * current_task, 1, 1)),
            nn.Conv2d(filters[1] * current_task, filters[1], 1),
            nn.ReLU()
        ))
        setattr(new_backbone, 'adaptor' + str(5), nn.Sequential(
            AlphaModule((filters[0] * current_task, 1, 1)),
            nn.Conv2d(filters[0] * current_task, filters[0], 1),
            nn.ReLU()
        ))
        
    return new_backbone



class Pnn(ContinualModel):
    NAME = 'pnn'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Pnn, self).__init__(backbone, loss, args, transform)
        backbone = replace_features(backbone)
        
        self.net = ListModule(backbone)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        
    def begin_task(self, dataset):
        if self.current_task > 0:
            # instantiate new column & add adaptors
            new_backbone = wrap_adaptors(replace_features(dataset.get_backbone(self.args)), self.current_task)
            self.net.append(new_backbone)
            self.net.to(self.device)
            
            for i in range(self.current_task):
                for p in self.net[i].parameters():
                    p.requires_grad = False
            
            self.opt = optim.SGD(filter(lambda p:p.requires_grad, self.net.parameters()), lr=self.args.lr)
            
            

    
    def forward(self, x, task_id):
        if task_id > self.current_task:
            task_id = self.current_task
        if task_id >= len(self.net):
            task_id = len(self.net) - 1
        if task_id == 0:
            out = self.net[0](x)
        else:    
            old_x2s = []
            old_x3s = []
            old_x4s = []
            old_x6s = []
            old_x8s = []
            old_x10s = []
            with torch.no_grad():
                for i in range(task_id):
                    _,midfeatures = self.net[i].features(x, ret_intermediate = True)
                    x2,x3,x4,x6,x8,x10 = midfeatures
                    old_x2s.append(x2)
                    old_x3s.append(x3)
                    old_x4s.append(x4)
                    old_x6s.append(x6)
                    old_x8s.append(x8)
                    old_x10s.append(x10)
                
            
            
            # Encode
            x1 = self.net[task_id].conv1(x)
            x1 = self.net[task_id].bn1(x1)
            x1 = F.relu(x1)
            x1 = self.net[task_id].conv2(x1)
            #x1 = self.input_layer(x)
            x2 = self.net[task_id].residual_conv_1(x1)
            y = self.net[task_id].adaptor1(torch.cat(old_x2s, 1))
            x3 = self.net[task_id].residual_conv_2(x2+y)
            # Bridge
            y = self.net[task_id].adaptor2(torch.cat(old_x3s, 1))
            x4 = self.net[task_id].bridge(x3+y)
            # Decode
            x4 = self.net[task_id].upsample_1(x4)
            y = self.net[task_id].adaptorbridge(torch.cat(old_x4s, 1))
            x5 = torch.cat([x4+y, x3], dim=1)

            x6 = self.net[task_id].up_residual_conv1(x5)

            x6 = self.net[task_id].upsample_2(x6)
            y = self.net[task_id].adaptor3(torch.cat(old_x6s, 1))
            x7 = torch.cat([x6+y, x2], dim=1)

            x8 = self.net[task_id].up_residual_conv2(x7)

            x8 = self.net[task_id].upsample_3(x8)
            y = self.net[task_id].adaptor4(torch.cat(old_x8s, 1))
            x9 = torch.cat([x8+y, x1], dim=1)

            x10 = self.net[task_id].up_residual_conv3(x9)
            y = self.net[task_id].adaptor5(torch.cat(old_x10s, 1))
            
            out = self.net[task_id].last(x10+y)

        return out



    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.forward(inputs, self.current_task)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
