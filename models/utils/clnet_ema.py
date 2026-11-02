"""Checkpointable decoder EMA primitive; the clNet lifecycle is still separate."""
import copy
import torch
from torch import nn


class DecoderEMA(nn.Module):
    def __init__(self, decoder, decay):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must lie in [0,1)')
        self.shadow = copy.deepcopy(decoder).requires_grad_(False).eval()
        self.register_buffer('decay', torch.tensor(float(decay), dtype=torch.float64))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, decoder):
        source, target = decoder.state_dict(), self.shadow.state_dict()
        if source.keys() != target.keys():
            raise ValueError('Decoder EMA structure mismatch')
        for name, dest in target.items():
            src = source[name]
            if dest.shape != src.shape or dest.dtype != src.dtype:
                raise ValueError('Decoder EMA tensor mismatch: ' + name)
            if src.is_floating_point() and not torch.isfinite(src).all():
                raise FloatingPointError('Nonfinite decoder state: ' + name)
        rate = float(self.decay)
        for name, dest in target.items():
            if dest.is_floating_point():
                dest.mul_(rate).add_(source[name], alpha=1-rate)
            else:
                dest.copy_(source[name])
        self.updates.add_(1)

    @torch.no_grad()
    def copy_to(self, decoder):
        decoder.load_state_dict(self.shadow.state_dict(), strict=True)
