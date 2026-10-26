"""ReBac-C branch topology and fusion, adapted to ResUNet decoder features.

Reference: ANDYZAQ/ReBac, dae1b48d461ba62bab99414d57d6474f5aa3d80b.
The DeepLab ASPP is replaced by the benchmark's shared ResUNet features.
"""
import torch
from torch import nn
from torch.nn import functional as F


def fuse_background(base, residuals, foreground_groups, training):
    if not residuals:
        return base
    if training:
        return base.detach() + residuals[-1]
    # Select the originating task of the winning foreground class. The base
    # task contributes zero correction; later tasks only reduce background.
    winner = torch.cat([g.amax(1, keepdim=True) for g in foreground_groups], 1).argmax(1, keepdim=True)
    corrections = torch.cat([torch.zeros_like(base)] + [-F.relu(-r) for r in residuals], 1)
    return base.detach() + corrections.gather(1, winner)


class RebacResUnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.class_num = backbone.last.out_channels
        width = backbone.last.in_channels
        backbone.last = nn.Identity()
        self.backbone = backbone
        self.width = width
        self.branches = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.counts = []
        self.register_buffer('seen_tasks', torch.tensor(0, dtype=torch.long))
        self.register_buffer('seen_classes', torch.tensor(1, dtype=torch.long))
        self.register_buffer('freeze_bn', torch.tensor(False))

    def configure(self, counts):
        counts = list(counts)
        if self.counts:
            if self.counts != counts:
                raise ValueError('ReBac task partition cannot change after initialization.')
            return False
        if not counts or any(type(c) is not int or c <= 0 for c in counts) or sum(counts)+1 != self.class_num:
            raise ValueError('ReBac foreground counts must match the output space.')
        device = self.seen_tasks.device
        self.counts = counts
        for i, count in enumerate(counts):
            self.branches.append(nn.Sequential(nn.Conv2d(self.width, self.width, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(self.width), nn.ReLU()))
            self.heads.append(nn.Conv2d(self.width, count + (i == 0), 1))
            if i:
                self.adapters.append(nn.Conv2d(self.width, 1, 1))
        for module in (self.branches, self.heads, self.adapters):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
        self.to(device)
        return True

    def initialize_branch(self, task):
        if task:
            # Author checkpoint initialization copies conv/BN affine values
            # from the base branch; new BN statistics are kept independent.
            with torch.no_grad():
                self.branches[task][0].weight.copy_(self.branches[0][0].weight)
                self.branches[task][1].weight.copy_(self.branches[0][1].weight)
                self.branches[task][1].bias.copy_(self.branches[0][1].bias)

    def train(self, mode=True):
        super().train(mode)
        if bool(self.freeze_bn):
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        return self

    def components(self, x):
        tasks = int(self.seen_tasks)
        if not tasks:
            raise RuntimeError('Configure and begin a ReBac task before prediction.')
        feature = self.backbone.features(x)
        responses = [self.branches[i](feature) for i in range(tasks)]
        heads = [self.heads[i](responses[i]) for i in range(tasks)]
        residuals = [self.adapters[i-1](responses[i]) - F.relu(heads[i].detach().amax(1, keepdim=True))
                     for i in range(1, tasks)]
        foregrounds = [heads[0][:, 1:]] + heads[1:]
        background = fuse_background(heads[0][:, :1], residuals, foregrounds, self.training)
        logits = torch.cat([background] + foregrounds, 1)
        groups = [heads[0]] + [torch.cat([residuals[i-1], heads[i]], 1) for i in range(1, tasks)]
        return logits, {'responses': responses, 'groups': groups, 'residuals': residuals}

    def forward(self, x, task_index=None):
        if not int(self.seen_tasks):
            return torch.cat([x.new_zeros(x.shape[0], 1, *x.shape[-2:]),
                              x.new_full((x.shape[0], self.class_num-1, *x.shape[-2:]), -1e4)], 1)
        logits, _ = self.components(x)
        return F.pad(logits, (0, 0, 0, 0, 0, self.class_num-logits.shape[1]), value=-1e4)
