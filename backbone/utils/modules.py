# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AlphaModule(nn.Module):
    def __init__(self, shape):
        super(AlphaModule, self).__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.alpha = Parameter(torch.rand(tuple([1] + list(shape))) * 0.1,
                               requires_grad=True)

    def forward(self, x):
        return x * self.alpha

    def parameters(self, recurse: bool = True):
        yield self.alpha


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.idx = 0
        for module in args:
            self.add_module(str(self.idx), module)
            self.idx += 1

    def append(self, module):
        self.add_module(str(self.idx), module)
        self.idx += 1

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.idx
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    
    
class LinearDiag(nn.Module):
    def __init__(self, s):
        super(LinearDiag, self).__init__()
        self.P = Parameter(torch.ones(s[0], 1) + torch.randn(s[0], 1) / s[0])
        self.register_parameter('P', self.P)
        # print 's is size: {},registered parameter of size {}'.format(s,self.P.size())

    def forward(self, w):
        return self.P * w  # scale each row by a scalar


# this is the same as linear but for the code to be more readable we define it here anyway
class LinearSimple(nn.Module):
    def __init__(self, s):
        super(LinearSimple, self).__init__()
        self.P = Parameter(
            torch.eye(s[0]) + torch.eye(s[0]) * torch.randn(s[0], s[0]) / s[0])
        self.register_parameter('P', self.P)

    def forward(self, w):
        return torch.matmul(self.P, w)


# A linear layer which is a product of two low-rank matrices
class LinearLowRank(nn.Module):
    def __init__(self, s, rnk=None):
        super(LinearLowRank, self).__init__()
        if rnk is None:
            rnk = s[0] / 2

        #self.p1 = Parameter((torch.rand(s[0],rnk))/s[0]*rnk)
        #self.p2 = Parameter((torch.rand(rnk,s[0]))/s[0]*rnk)

        #self.p1 = Parameter((torch.ones(s[0],rnk)+torch.rand(s[0],rnk))/s[0]*rnk)
        # self.p1 = Parameter(torch.rand(s[0],rnk)/(s[0]*rnk))#+torch.rand(s[0],rnk))/s[0]*rnk)
        self.p1 = Parameter(torch.zeros(s[0], rnk))
        self.p1.data[:rnk, :rnk] = torch.eye(rnk)
        self.p2 = Parameter(torch.zeros(rnk, s[0]))
        self.p2.data[:rnk, :rnk] = torch.eye(rnk)
        #self.p2 = Parameter((torch.ones(rnk,s[0])+torch.rand(rnk,s[0]))/s[0]*rnk)
        self.register_parameter('p1', self.p1)
        self.register_parameter('p2', self.p2)

    def forward(self, w):
        return torch.matmul(torch.matmul(self.p1, self.p2), w)
