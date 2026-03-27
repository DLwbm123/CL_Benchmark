# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# modified from https://github.com/rosenfeldamir/incremental_learning
from types import MethodType
import torch.optim as optim
from torch.optim import SGD
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.conf import get_device
from utils.args import *
from datasets import get_dataset
from backbone.utils.modules import ListModule, LinearDiag, LinearSimple, LinearLowRank
from backbone.ResUnet import ResUnet,resunet32,ResidualConv
from models.utils.continual_model import ContinualModel
import numpy as np
from torch.autograd import Variable



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Incremental Learning Through Deep Adaptation.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--controlType', type=str, default='linear',
                        help='controlType for DAN.')
    parser.add_argument('--rnk_ratio', type=float, default=0.5,
                        help='rnk_ratio for Low-rank controller.')
    return parser

class controlledConv(nn.Module):
    def __init__(self, conv, controlType='linear', bias=None, rnk_ratio=.5):
        super(controlledConv, self).__init__()
        
        self.conv = conv
        # Copy the weights as a constant from the original convolution --
        # just to make sure it doesn't change
        s = self.conv.weight.size()
        self.s = list(s)
        w = Variable(torch.zeros(self.s))
        w.data.copy_(self.conv.weight.data)
        w = w.view(self.s[0], -1)
        self.register_buffer('w', w)
        self.ctrl = ListModule()
        self.hasBias = bias is not None
        self.bias = ListModule() if self.hasBias else None
                    
        
    def addctrlModule(self, controlType='linear', rnk_ratio=.5): 
        
        if controlType == 'linear':  # simple linear combination
            ctrl = LinearSimple(self.s)
        elif controlType == 'diagonal':  # only scaling
            ctrl = LinearDiag(s)
        elif controlType == 'low_rank':  # linear combination with low-rank decomposition
            rnk = int(s[0] * rnk_ratio)
            ctrl = LinearLowRank(s, rnk)
        self.ctrl.append(ctrl)
        if self.hasBias:
            s_bias = self.s[0]
            
            bias = Parameter(torch.zeros(
                self.conv.bias.data.size()))
            bias.data.copy_(self.conv.bias.data[:s_bias])
            self.bias.append(bias)
        else:
            self.bias = None

        for p in self.conv.parameters():
            p.requires_grad = False

    def setConvLearnable(self, T):
        for p in self.conv.parameters():
            p.requires_grad = T
        
            
    def setCtrlLearnable(self, task_id, T):
        for p in self.ctrl[task_id].parameters():
            p.requires_grad = T
        if self.hasBias:
            for p in self.bias[task_id].parameters():
                p.requires_grad = T
                
    def updateWdata(self):
        weight = copy.deepcopy(self.conv.weight.data)
        self.w.data.copy_(weight.view(self.s[0], -1))
        

    def forward(self, x, alpha=0):
        # Modify the weights
        #conv = self.conv
        if alpha > 0 and alpha <= len(self.ctrl):
            s = self.s
            w = self.w
            bias = None
            assert self.w is not None
            newWeights = self.ctrl[alpha-1](w)
            if self.hasBias:
                bias = self.bias[alpha-1]
        
            newWeights = newWeights.contiguous()  # TODO: check if this is necessary
            newWeights = newWeights.view(s)

            out = F.conv2d(x, newWeights, bias, stride=self.conv.stride,
                         padding=self.conv.padding, dilation=self.conv.dilation)
        else:
            out = self.conv(x)

        return out
    

def makeItControlled(origModule, newModule, controlAnyway=True, controllerType='linear', rnk_ratio=.5, verbose=False):
    for orig, new in zip(origModule.named_children(), newModule.named_children()):
        # print '.'
        name1, module1 = orig
        name2, module2 = new
        if 'last' in name1:
            continue #skip the last Conv2d
        # if a convolution - make a controlled copy. otherwise, do nothing! everything is as it should be.
        
        #################################################################
        def new_forward(self, x, alpha=0): #modify ResidualConv
            shortcut = self.conv_skip(x)
            x = self.bn1(x)
            out = F.relu(x)
            out = self.conv1(out,alpha)
            out = self.bn2(out)
            out = self.conv2(F.relu(out),alpha)

            return out + shortcut
        
        if isinstance(module2, ResidualConv):
            module2.forward = MethodType(new_forward, module2)
        #################################################################
        
        if type(module1) is nn.Conv2d:

            # print 'setting',name2,'of new module to a controlled conv.'
            O = module1.out_channels
            I = module1.in_channels
            K = np.prod(module1.kernel_size)
            params_before = O * I * K
            if controllerType == 'diagonal':
                params_after = O
            elif controllerType == 'linear':
                params_after = O ** 2
            else:
                params_after = 2 * (O * rnk_ratio)**2

            if verbose:
                print(str(params_before)+'-->'+str(params_after))
            if params_after < params_before or controlAnyway:
                # print 'creating controlled conv'
                m = controlledConv(module1, controllerType, module2.bias, rnk_ratio)
                setattr(newModule, name1, m)
            # else:
            #    pass
                # print '...meh'
            # leave it as it is.
        # print 'going into', name1
        makeItControlled(module1, module2, controlAnyway=controlAnyway,
                         controllerType=controllerType, verbose=verbose, rnk_ratio=rnk_ratio)
    
def addController(net, controlType='linear', rnk_ratio=.5):
    for name, module in net.named_children():
        if isinstance(module, controlledConv):
            module.addctrlModule(controlType, rnk_ratio)
            module.setCtrlLearnable(-1, True)
            module.updateWdata()
        addController(module, controlType, rnk_ratio)


class DAN(ContinualModel):
    NAME = 'dan'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(DAN, self).__init__(backbone, loss, args, transform)
        new_backbone = copy.deepcopy(backbone)
        makeItControlled(backbone, new_backbone, controlAnyway = True, controllerType = args.controlType, verbose=False, rnk_ratio=args.rnk_ratio)
        self.net = new_backbone
        
        def new_features(self, x, alpha = 0):
            # Encode
            
            x1 = self.conv1(x, alpha)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.conv2(x1, alpha)
            #x1 = self.input_layer(x)
            x2 = self.residual_conv_1(x1, alpha)
            x3 = self.residual_conv_2(x2, alpha)
            # Bridge
            x4 = self.bridge(x3, alpha)
            # Decode
            x4 = self.upsample_1(x4)
            x5 = torch.cat([x4, x3], dim=1)

            x6 = self.up_residual_conv1(x5, alpha)

            x6 = self.upsample_2(x6)
            x7 = torch.cat([x6, x2], dim=1)

            x8 = self.up_residual_conv2(x7, alpha)

            x8 = self.upsample_3(x8)
            x9 = torch.cat([x8, x1], dim=1)

            x10 = self.up_residual_conv3(x9, alpha)

            #output = self.output_layer(x10)

            return x10
        self.net.features = MethodType(new_features, self.net)
        #################################################################
        
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        #print(self.net)
        
    def begin_task(self, dataset):
        if self.current_task > 0:
            for p in self.net.parameters(): #fix old params
                p.requires_grad =  False
                
            # instantiate new ctrl
            addController(self.net, self.args.controlType, self.args.rnk_ratio)
            for p in self.net.last[str(self.current_task)].parameters():
                p.requires_grad = True
            #for name, p in self.net.named_parameters():
            #    if p.requires_grad:
            #        print(name)
            self.net.to(self.device)
            self.opt = optim.SGD(filter(lambda p:p.requires_grad, self.net.parameters()), lr=self.args.lr)
            

    
    def forward(self, x, task_id):
        if task_id > self.current_task:
            task_id = self.current_task
        
        x10 = self.net.features(x, task_id)
        out = self.net.logits(x10, task_id)

        return out



    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.forward(inputs, self.current_task)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
