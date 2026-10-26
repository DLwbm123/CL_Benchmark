"""2D ResUNet adapter of the MICCAI 2023 class-specific dynamic CLIP heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.utils.medical import resunet_features


class ClipHeadResUnet(nn.Module):
    def __init__(self, backbone, embeddings, feature_width=None):
        super().__init__()
        decoder_channels = backbone.last.in_channels if feature_width is None else feature_width
        encoder_channels = backbone.bridge.conv2.out_channels
        self.backbone = backbone
        # The original output classifier is replaced, not retained as dead params.
        self.backbone.last = nn.Identity()
        self.precls = nn.Sequential(nn.GroupNorm(math.gcd(16, decoder_channels),
                                               decoder_channels), nn.ReLU(),
                                    nn.Conv2d(decoder_channels, 8, 1))
        self.gap = nn.Sequential(nn.GroupNorm(math.gcd(16, encoder_channels),
                                            encoder_channels), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(encoder_channels, 256, 1))
        # Original three-layer dynamic head: 8->8->8->1 (153 parameters).
        self.weight_sizes = (64, 64, 8)
        self.bias_sizes = (8, 8, 1)
        self.controllers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(768, 512, 1), nn.ReLU(), nn.Conv2d(512, 153, 1))
            for _ in range(embeddings.shape[0])])
        self.register_buffer('organ_embeddings', embeddings.float().clone())
        self.register_buffer('seen_classes', torch.tensor(0, dtype=torch.long))

    @property
    def class_num(self):
        return self.organ_embeddings.shape[0]

    def foreground(self, x, count=None, indices=None):
        if count is None:
            count = int(self.seen_classes)
        if indices is None:
            indices = list(range(count))
        else:
            indices = list(indices)
            count = len(indices)
        if not 1 <= count <= self.class_num or any(type(i) is not int or not 0 <= i < self.class_num for i in indices):
            raise ValueError('No learned foreground classes, or invalid class count.')
        features, bridge = resunet_features(self.backbone, x)
        features = self.precls(features)
        context = self.gap(bridge)
        batch, channels, height, width = features.shape
        params = []
        for index in indices:
            text = self.organ_embeddings[index].reshape(1, 512, 1, 1)
            cond = torch.cat([context, text.expand(batch, -1, -1, -1)], 1)
            params.append(self.controllers[index](cond).flatten(1))
        params = torch.stack(params, 1).reshape(batch * count, 153)
        chunks = params.split(self.weight_sizes + self.bias_sizes, 1)
        instances = batch * count
        out = features[:, None].expand(-1, count, -1, -1, -1)
        out = out.reshape(1, instances * channels, height, width)
        for layer in range(3):
            cout = 8 if layer < 2 else 1
            weight = chunks[layer].reshape(instances * cout, 8, 1, 1)
            bias = chunks[layer + 3].reshape(instances * cout)
            out = F.conv2d(out, weight, bias, groups=instances)
            if layer < 2:
                out = F.relu(out)
        return out.reshape(batch, count, height, width)

    def forward(self, x, task_index=None):
        # No oracle head selection: every learned foreground head competes.
        # Zero background logit means a foreground must exceed sigmoid 0.5.
        # A fixed full label axis keeps compatibility with existing evaluators.
        background = x.new_zeros(x.shape[0], 1, *x.shape[-2:])
        count = int(self.seen_classes)
        foreground = (self.foreground(x, count) if count else
                      x.new_empty(x.shape[0], 0, *x.shape[-2:]))
        future = x.new_full((x.shape[0], self.class_num - count, *x.shape[-2:]), -1e4)
        return torch.cat([background, foreground, future], 1)
