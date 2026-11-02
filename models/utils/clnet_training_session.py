"""Train-only task lifecycle for a non-native 2D clNet adaptation.

The caller supplies a fixed pruning schedule and train-only score callbacks.
Author 3D pretrained encoder, validation-based pruning and body-part merging
are not reproduced. Checkpoints, deadlines and scheduling belong to the caller.
"""
import torch
from torch import nn
from models.utils.clnet_pruning import DecoderMasks
from models.utils.clnet_pruning_gate import prune_with_train_gate
from models.utils.clnet_ema import DecoderEMA


class ContinualSegmentSession(nn.Module):
    def __init__(self, network, optimizer, loss_fn, ema_decay):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.masks = nn.ModuleDict({k: DecoderMasks(d) for k, d in network.decoders.items()})
        self.ema = nn.ModuleDict({k: DecoderEMA(d, ema_decay) for k, d in network.decoders.items()})
        self.register_buffer('current_task', torch.tensor(0, dtype=torch.long))
        self.register_buffer('task_open', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('pruning_finished', torch.tensor(False, dtype=torch.bool))

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        task = int(self.current_task)
        if bool(self.task_open):
            self.network.configure_task(task)
        elif task > 0:
            self.network.configure_task(task - 1)
        return result

    def begin_task(self, task):
        if bool(self.task_open) or task != int(self.current_task) or str(task) not in self.network.decoders:
            raise ValueError('Tasks must be started exactly once in stream order')
        self.network.configure_task(task)
        self.network.seen_tasks.fill_(task + 1)
        self.task_open.fill_(True)
        self.pruning_finished.fill_(False)

    def step(self, images, labels):
        if not bool(self.task_open):
            raise RuntimeError('Task not open')
        if labels.dtype != torch.long or labels.ndim != 3:
            raise ValueError('Expected binary labels [B,H,W]')
        if ((labels != 0) & (labels != 1) & (labels != 255)).any():
            raise ValueError('Labels must be binary or ignore255')
        task = int(self.current_task)
        decoder, masks = self.network.decoders[str(task)], self.masks[str(task)]
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.network(images, task), labels)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite segmentation loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in self.network.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite segmentation gradients')
        masks.apply(decoder, self.optimizer)
        self.optimizer.step()
        masks.apply(decoder, self.optimizer)
        if bool(self.pruning_finished):
            self.ema[str(task)].update(decoder)
        return float(loss.detach())

    def try_prune(self, target, train_score, baseline, max_drop=.02, max_attempts=8):
        if not bool(self.task_open) or bool(self.pruning_finished):
            raise RuntimeError('Pruning is only available before recovery training')
        task = str(int(self.current_task))
        return prune_with_train_gate(self.network.decoders[task], self.masks[task],
                                     self.optimizer, target, train_score, baseline,
                                     max_drop=max_drop, max_attempts=max_attempts)

    def start_recovery(self):
        if not bool(self.task_open) or bool(self.pruning_finished):
            raise RuntimeError('Invalid recovery transition')
        task = str(int(self.current_task))
        self.ema[task].shadow.load_state_dict(self.network.decoders[task].state_dict())
        self.ema[task].updates.zero_()
        self.pruning_finished.fill_(True)

    def end_task(self):
        if not bool(self.task_open) or not bool(self.pruning_finished):
            raise RuntimeError('Recovery phase is required before task completion')
        task = str(int(self.current_task))
        if int(self.ema[task].updates) == 0:
            raise RuntimeError('EMA requires at least one recovery update')
        self.ema[task].copy_to(self.network.decoders[task])
        self.masks[task].apply(self.network.decoders[task], self.optimizer)
        self.task_open.fill_(False)
        self.current_task.add_(1)
