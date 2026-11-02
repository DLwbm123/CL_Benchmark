"""Per-slice [-1,1] intensity adaptation for paper-derived SMG experiments.

The paper specifies the range but not an exact normalization implementation.
This min/max choice is an explicit benchmark adaptation; it must also be used
for evaluation, and does not inspect other samples or future domains.
"""
import torch
from torch import nn


def normalize_per_slice(images):
    if images.ndim != 4 or not images.is_floating_point() or images.shape[1] != 1:
        raise ValueError('Expected floating single-channel [B,1,H,W] images')
    if not images.shape[0] or not torch.isfinite(images).all():
        raise ValueError('Images must be nonempty and finite')
    minimum = images.amin(dim=(-2, -1), keepdim=True)
    maximum = images.amax(dim=(-2, -1), keepdim=True)
    span = maximum - minimum
    normal = torch.where(span > 0, 2 * (images - minimum) / span.clamp_min(1e-12) - 1,
                         torch.zeros_like(images))
    return normal


def require_normalized(images):
    if not torch.isfinite(images).all() or (images < -1.00001).any() or (images > 1.00001).any():
        raise ValueError('SMG inputs must be normalized to [-1,1]')


class NormalizedEvaluationNetwork(nn.Module):
    """Evaluation wrapper; train core receives explicitly normalized tensors.

    Keeping the core separate avoids accidentally re-normalizing generated
    replay samples within dual-meta functional calls.
    """
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, raw_images, task_index=None):
        return self.core(normalize_per_slice(raw_images))
