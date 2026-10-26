"""CNN-form CIT class-independent attention, adapted to ResUNet decoder features.

Reference: jinchaogjc/cit, mmseg_custom/decode_heads/dv3_head.py.
No query self-attention, no FFN, identity Q/V/output projections; learned K.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CitResUnet(nn.Module):
    def __init__(self, backbone, heads=8, dropout=0.1, stride=8, foreground_classes=None, feature_width=None):
        super().__init__()
        channels = backbone.last.in_channels if feature_width is None else feature_width
        self.class_num = backbone.last.out_channels - 1 if foreground_classes is None else foreground_classes
        if channels % heads:
            raise ValueError('CIT feature width must be divisible by the attention head count.')
        self.backbone = backbone
        self.backbone.last = nn.Identity()
        self.queries = nn.Embedding(self.class_num, channels)
        self.key = nn.Linear(channels, channels, bias=True)
        self.presence = nn.Linear(channels, 2)
        self.norm = nn.LayerNorm(channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.heads = heads
        self.stride = stride
        self.register_buffer('seen_classes', torch.tensor(0, dtype=torch.long))

    def components(self, x, count=None):
        count = int(self.seen_classes) if count is None else count
        if not 1 <= count <= self.class_num:
            raise ValueError('CIT needs at least one learned foreground query.')
        features = self.backbone.features(x)
        if min(features.shape[-2:]) < self.stride:
            raise ValueError('CIT feature stride exceeds the spatial input size.')
        if self.stride > 1:
            features = F.avg_pool2d(features, self.stride, self.stride)
        batch, channels, height, width = features.shape
        memory = features.flatten(2).transpose(1, 2)
        query = self.queries.weight[:count][None].expand(batch, -1, -1)
        head_dim = channels // self.heads
        q = query.reshape(batch, count, self.heads, head_dim).transpose(1, 2)
        k = self.key(memory).reshape(batch, -1, self.heads, head_dim).transpose(1, 2)
        v = memory.reshape(batch, -1, self.heads, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * head_dim ** -0.5
        weights = self.attention_dropout(scores.softmax(-1))
        decoded = (weights @ v).transpose(1, 2).reshape(batch, count, channels)
        decoded = self.norm(query + self.residual_dropout(decoded))
        return {'presence': self.presence(decoded),
                'masks': scores.mean(1).reshape(batch, count, height, width)}

    def forward(self, x, task_index=None):
        count = int(self.seen_classes)
        if not count:
            return torch.cat([x.new_zeros(x.shape[0], 1, *x.shape[-2:]),
                              x.new_full((x.shape[0], self.class_num, *x.shape[-2:]), -1e4)], 1)
        outputs = self.components(x)
        presence = outputs['presence'].softmax(-1)[..., 0]
        scores = presence[:, :, None, None] * outputs['masks'].sigmoid()
        scores = F.interpolate(scores, x.shape[-2:], mode='bilinear', align_corners=False)
        # Author ADE experiments have foreground queries, without a dedicated
        # medical-background token. Supply an explicit 0.5 rejection rule for
        # the benchmark's background label, preserving independent queries.
        background = 1 - scores.amax(1, keepdim=True)
        logits = torch.cat([background, scores], 1).clamp_min(1e-12).log()
        if count < self.class_num:
            logits = torch.cat([logits, x.new_full(
                (x.shape[0], self.class_num - count, *x.shape[-2:]), -1e4)], 1)
        return logits
