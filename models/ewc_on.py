# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel

#self.net.get_params() -------> self.net.get_params_without_outputlayer()
#self.net.get_grads() --------> self.net.get_grads_without_outputlayer()

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, default=1,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma parameter for EWC online')

    return parser


class EwcOn(ContinualModel):
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            #penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            if self.args.multihead:
                penalty = (self.fish * ((self.net.get_params_without_outputlayer() - self.checkpoint) ** 2)).sum()
            else:
                penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
                
            return penalty

    def end_task(self, dataset):
        if self.args.multihead:
            fish = torch.zeros_like(self.net.get_params_without_outputlayer())
        else:
            fish = torch.zeros_like(self.net.get_params())
        #print("fish", fish.size())
        #print("train loader", dataset.train_loader.dataset.task)
        for j, data in enumerate(dataset.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.forward(ex.unsqueeze(0))
                loss = self.loss(output, lab.unsqueeze(0))
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                if self.args.multihead:
                    fish += exp_cond_prob * self.net.get_grads_without_outputlayer() ** 2
                else:
                    fish += exp_cond_prob * self.net.get_grads() ** 2

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        if self.args.multihead:
        #self.checkpoint = self.net.get_params().data.clone()
            self.checkpoint = self.net.get_params_without_outputlayer().data.clone()
        else:
            self.checkpoint = self.net.get_params().data.clone()

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.forward(inputs)
        penalty = self.penalty()

        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        #print("penalty", penalty, "loss", self.loss(outputs, labels).item())
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
