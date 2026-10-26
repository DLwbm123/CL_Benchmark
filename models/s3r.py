"""S3R (TMI 2023), binary Domain-CL, benchmark backbone adaptation.

Implements uncertainty-weighted semantics + SDF + ACNN shape-code sensitivity,
online importance accumulation, and quadratic selective regularization.
"""
from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F

from models.utils.medical import (MedicalModel, binary_targets, load_tensors,
                                 medical_parser, require_single_head)
from models.utils.shape import ShapeResUnet, signed_distance


def get_parser():
    parser = medical_parser('Shape and semantics selective regularization')
    parser.set_defaults(lr=5e-4, lr_decay_rate=1.0)
    parser.add_argument('--s3r_optimizer', choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--s3r_weight_decay', type=float, default=1e-4)
    parser.add_argument('--s3r_epoch_gamma', type=float, default=0.98)
    parser.add_argument('--s3r_lambda', type=float, default=10000.)
    parser.add_argument('--s3r_alpha', type=float, default=0.001)
    parser.add_argument('--s3r_alpha1', type=float, default=0.05)
    parser.add_argument('--s3r_beta', type=float, default=1.)
    parser.add_argument('--s3r_beta1', type=float, default=1.)
    parser.add_argument('--s3r_dropout', type=float, default=0.5)
    parser.add_argument('--s3r_mc_samples', type=int, default=8)
    parser.add_argument('--s3r_importance_batches', type=int, default=0,
                        help='0: all current-task training batches; >0: explicit approximation')
    parser.add_argument('--s3r_shape_checkpoint', type=str, default=None,
                        help='Adapter ACNN state dict; optional {task} placeholder (zero-based)')
    parser.add_argument('--s3r_shape_epochs', type=int, default=0,
                        help='Explicit current-task-only ACNN pretraining; 0 requires checkpoint')
    parser.add_argument('--s3r_shape_lr', type=float, default=0.001)
    return parser


def confidence_mask(mc_probabilities):
    average = mc_probabilities.mean(0)
    confidence = torch.exp(-(average * (average + 1e-6).log()).sum(1, keepdim=True))
    low = confidence.amin(dim=(-2, -1), keepdim=True)
    span = confidence.amax(dim=(-2, -1), keepdim=True) - low
    # Upstream divides by zero on spatially constant predictions. A uniform
    # confidence mask retains the semantic term instead of introducing NaNs.
    return torch.where(span > 0, (confidence - low) / span.clamp_min(1e-12),
                       torch.ones_like(confidence))


class S3r(MedicalModel):
    NAME = 's3r'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        if not 0 <= args.s3r_dropout < 1 or args.s3r_mc_samples < 1:
            raise ValueError('Invalid dropout probability or MC sample count.')
        if args.s3r_importance_batches < 0 or args.s3r_shape_epochs < 0:
            raise ValueError('Batch and epoch limits must be nonnegative.')
        if any(getattr(args, name) < 0 for name in (
                's3r_lambda', 's3r_alpha', 's3r_alpha1', 's3r_beta', 's3r_beta1')):
            raise ValueError('S3R loss/importance weights must be nonnegative.')
        super().__init__(ShapeResUnet(backbone, args.s3r_dropout), loss, args, transform)
        if args.s3r_weight_decay < 0 or not 0 < args.s3r_epoch_gamma <= 1:
            raise ValueError('Invalid S3R optimizer configuration.')
        if args.lr_decay_rate != 1.0:
            raise ValueError('S3R applies its own epoch decay; set lr_decay_rate=1 to avoid double decay.')
        optimizer = getattr(torch.optim, args.s3r_optimizer)
        self.opt = optimizer((p for p in self.net.parameters() if p.requires_grad),
                             lr=args.lr, weight_decay=args.s3r_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.lr_decay_epoch,
                                                         gamma=1.0)
        self.anchor = {}
        self.importance = {}

    def begin_epoch(self, epoch, n_epochs):
        # Pinned author source resets the LR for each task and steps its
        # ExponentialLR(gamma=.98) before each epoch, retaining Adam moments.
        for group in self.opt.param_groups:
            group['lr'] = self.args.lr * self.args.s3r_epoch_gamma ** (epoch + 1)

    def knowledge_state(self):
        return {'anchor': self.anchor, 'importance': self.importance}

    def evaluation_parameter_count(self):
        # SDF head and ACNN are training auxiliaries; report their costs
        # separately, rather than calling them deployed segmentation weights.
        return sum(p.numel() for p in self.net.backbone.parameters())

    def restore_knowledge(self, state):
        self.anchor = state['anchor']
        self.importance = state['importance']

    def begin_task(self, dataset):
        self.check_task(dataset)
        shape = self.net.shape_encoder
        if self.args.s3r_shape_checkpoint:
            path = self.args.s3r_shape_checkpoint.format(task=self.current_task)
            shape.load_state_dict(load_tensors(path), strict=True)
        elif self.args.s3r_shape_epochs > 0:
            # Author code requires a pretrained ACNN but does not provide its
            # training procedure. This explicit adaptation uses only current
            # training masks; its extra epochs must be counted in the budget.
            shape.requires_grad_(True).train()
            opt = torch.optim.Adam(shape.parameters(), lr=self.args.s3r_shape_lr)
            try:
                for _ in range(self.args.s3r_shape_epochs):
                    count = 0
                    for batch in dataset.train_loader:
                        labels = batch[1].to(self.device, dtype=torch.long)
                        masks = binary_targets(labels, 2)
                        opt.zero_grad(set_to_none=True)
                        _, reconstruction = shape(masks)
                        loss = F.cross_entropy(reconstruction, labels)
                        loss.backward()
                        opt.step()
                        count += 1
                    if not count:
                        raise ValueError('Cannot pretrain ACNN on an empty loader.')
            finally:
                shape.requires_grad_(False).eval()
                shape.zero_grad(set_to_none=True)
        else:
            raise ValueError('S3R requires s3r_shape_checkpoint or explicit '
                             's3r_shape_epochs > 0. Random frozen shape codes '
                             'are not a valid substitute for a trained ACNN.')
        shape.requires_grad_(False).eval()
        self.net.shape_ready.fill_(True)
        super().begin_task(dataset)

    def regularization_loss(self):
        total = next(self.net.parameters()).new_zeros(())
        for name, param in self.net.named_parameters():
            if name in self.importance:
                total = total + (self.importance[name].to(param) *
                                 (param - self.anchor[name].to(param)).square()).sum()
        return total

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        masks = binary_targets(labels, 2)
        logits, regression = self.net.auxiliary(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('S3R non-finite segmentation logits before BCE')
        probs = logits.softmax(1)
        if not torch.isfinite(probs).all():
            raise FloatingPointError('S3R non-finite segmentation probabilities before BCE')
        # Upstream S3R uses half BCE and half foreground Dice.
        foreground = probs[:, 1]
        target = masks[:, 1]
        dice = 1 - (2 * (foreground * target).sum() + 1e-5) / (
            foreground.sum() + target.sum() + 1e-5)
        supervised = 0.5 * (F.binary_cross_entropy(probs, masks) + dice)
        distance = F.mse_loss(regression, signed_distance(labels))
        with torch.no_grad():
            target_code = self.net.shape_encoder.encode(masks)
        code = self.net.shape_encoder.encode(probs)
        embedding = F.mse_loss(code, target_code)
        penalty = self.regularization_loss()
        total = (supervised + self.args.s3r_beta * distance +
                 self.args.s3r_beta1 * embedding + self.args.s3r_lambda * penalty)
        return self.optimization_step(total, supervised=supervised, sdf=distance,
                                      embedding=embedding, regularization=penalty)

    def end_task(self, dataset):
        self.require_open_task()
        params = {n: p for n, p in self.net.named_parameters() if p.requires_grad}
        additions = {n: torch.zeros_like(p) for n, p in params.items()}
        modes = {module: module.training for module in self.net.modules()}
        self.net.eval()
        count = 0
        loader = dataset.train_loader
        if self.args.s3r_importance_batches:
            loader = islice(loader, self.args.s3r_importance_batches)
        try:
            for batch in loader:
                inputs = batch[0].to(self.device, dtype=torch.float32)
                self.net.zero_grad(set_to_none=True)
                logits, regression = self.net.auxiliary(inputs)
                probs = logits.softmax(1)
                code = self.net.shape_encoder.encode(probs)
                for module in self.net.modules():
                    if isinstance(module, nn.Dropout2d):
                        module.train()
                with torch.no_grad():
                    mc_probs = torch.stack([self.net(inputs).softmax(1)
                                            for _ in range(self.args.s3r_mc_samples)])
                    mask = confidence_mask(mc_probs)
                for module in self.net.modules():
                    if isinstance(module, nn.Dropout2d):
                        module.eval()
                objective = ((mask * probs).square().mean() +
                             self.args.s3r_alpha * regression.square().mean() +
                             self.args.s3r_alpha1 * code.square().mean())
                objective.backward()
                for name, param in params.items():
                    if param.grad is not None:
                        additions[name].add_(param.grad.detach().abs())
                count += 1
            if count == 0:
                raise ValueError('S3R importance estimation needs current training data.')
            self.importance = {n: self.importance.get(n, torch.zeros_like(p)).to(p)
                               + additions[n] / count for n, p in params.items()}
            self.anchor = {n: p.detach().clone() for n, p in params.items()}
        finally:
            self.net.zero_grad(set_to_none=True)
            for module, mode in modes.items():
                module.training = mode
            self.net.shape_encoder.eval()
        self.task_open = False
        self.current_task += 1
