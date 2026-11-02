"""Task-routed 2D ResUNet encoder/decoders for a clNet adaptation.

The paper's progressive pruning, EMA, body-part merging and pretrained 3D
encoder are not implemented here. This module alone is not Continual Segment.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.up1 = copy.deepcopy(backbone.up_residual_conv1)
        self.up2 = copy.deepcopy(backbone.up_residual_conv2)
        self.up3 = copy.deepcopy(backbone.up_residual_conv3)
        self.head = copy.deepcopy(head)

    def forward(self, layers):
        x1, x2, x3, bridge = layers
        h = self.up1(torch.cat((F.interpolate(bridge, scale_factor=2, mode='bilinear', align_corners=False), x3), 1))
        h = self.up2(torch.cat((F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False), x2), 1))
        h = self.up3(torch.cat((F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False), x1), 1))
        return self.head(h)


class TaskRoutedNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if not isinstance(backbone.last, nn.ModuleDict):
            raise ValueError('Each task requires its own binary output head')
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.conv2 = backbone.conv2
        self.down1 = backbone.residual_conv_1
        self.down2 = backbone.residual_conv_2
        self.bridge = backbone.bridge
        self.decoders = nn.ModuleDict({k: Decoder(backbone, head) for k, head in backbone.last.items()})
        self.register_buffer('seen_tasks', torch.tensor(0, dtype=torch.long))
        self.encoder_frozen = False
        self.active_task = None

    def encoder_modules(self):
        return (self.conv1, self.bn1, self.conv2, self.down1, self.down2, self.bridge)

    def configure_task(self, task):
        if str(task) not in self.decoders:
            raise ValueError('Unknown task decoder')
        self.encoder_frozen = task > 0
        self.active_task = task
        for module in self.encoder_modules():
            module.requires_grad_(not self.encoder_frozen)
            module.train(self.training and not self.encoder_frozen)
        for key, decoder in self.decoders.items():
            decoder.requires_grad_(key == str(task))
            decoder.train(self.training and key == str(task))

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            for module in self.encoder_modules():
                module.eval()
        if self.active_task is not None:
            for key, decoder in self.decoders.items():
                decoder.train(mode and key == str(self.active_task))
        return self

    def forward(self, inputs, task_index=None):
        if type(task_index) is not int or str(task_index) not in self.decoders:
            raise ValueError('A valid task index is required')
        x1 = self.conv2(F.relu(self.bn1(self.conv1(inputs))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        bridge = self.bridge(x3)
        return self.decoders[str(task_index)]((x1, x2, x3, bridge))
