"""Checkpointable global magnitude masks for task-specific decoder experiments.

This is a pruning primitive only: it does not implement clNet's validation-
guided lottery-ticket schedule, rewinding, EMA or recovery training.
"""
import torch
from torch import nn


class DecoderMasks(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.names = [name for name, param in decoder.named_parameters()
                      if name.endswith('weight') and param.ndim >= 2]
        if not self.names:
            raise ValueError('Decoder contains no eligible weights')
        params = dict(decoder.named_parameters())
        for index, name in enumerate(self.names):
            self.register_buffer('mask_' + str(index), torch.ones_like(params[name], dtype=torch.bool))

    @torch.no_grad()
    def set_sparsity(self, decoder, target):
        if not 0 <= target < 1:
            raise ValueError('Sparsity must lie in [0,1)')
        params = dict(decoder.named_parameters())
        masks = [getattr(self, 'mask_' + str(i)) for i in range(len(self.names))]
        total = sum(m.numel() for m in masks)
        already = sum(int((~m).sum()) for m in masks)
        remove = int(total * target) - already
        if remove < 0:
            raise ValueError('Cannot reduce sparsity without a rewind')
        if remove == 0:
            return
        scores = torch.cat([params[n].detach().abs()[m] for n, m in zip(self.names, masks)])
        if remove > scores.numel() or not torch.isfinite(scores).all():
            raise ValueError('Invalid or nonfinite active weights')
        chosen = torch.argsort(scores, stable=True)[:remove]
        keep = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
        keep[chosen] = False
        offset = 0
        for mask in masks:
            count = int(mask.sum())
            updated = mask.clone()
            updated[mask] = keep[offset:offset + count]
            mask.copy_(updated)
            offset += count
        self.apply(decoder)

    @torch.no_grad()
    def apply(self, decoder, optimizer=None):
        params = dict(decoder.named_parameters())
        for index, name in enumerate(self.names):
            parameter = params[name]
            mask = getattr(self, 'mask_' + str(index))
            parameter.mul_(mask)
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
            if optimizer is not None:
                for value in optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.mul_(mask)
