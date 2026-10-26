"""NeST + MiB, ECCV 2024, with a frozen ResUNet-2D feature extractor.

Implements cross-task channel/class similarity, trainable importance/projection
matrices, background transformation, UNCE pre-tuning, and classifier compilation.
Reference: zhengyuan-xie/ECCV24_NeST; attribution and deviations in docs.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.utils.class_incremental import (
    ClassIncrementalModel, ciss_parser, unbiased_cross_entropy,
    unbiased_distillation, validate_partial_labels)


def get_parser():
    parser = ciss_parser('New-classifier pre-tuning + MiB')
    parser.add_argument('--nest_kd_weight', type=float, default=10.)
    parser.add_argument('--nest_teacher_scale', type=float, default=1.)
    parser.add_argument('--nest_pre_epochs', type=int, default=5,
                        help='Additional current-task-only classifier pre-tuning epochs')
    parser.add_argument('--nest_pre_lr_scale', type=float, default=1.)
    parser.add_argument('--nest_pre_weight_decay', type=float, default=1e-4)
    parser.add_argument('--nest_selection_size', type=int, default=128)
    return parser


@torch.no_grad()
def cross_task_similarity(teacher, loader, old_count, new_count, device, size=128):
    """Author select() statistics, avoiding an N_pixels x N_old x D expansion."""
    weights = teacher.classifier.weight[:old_count].flatten(1)
    bucket = weights.new_zeros(new_count, old_count, weights.shape[1])
    counts = weights.new_zeros(new_count)
    teacher.eval()
    for batch in loader:
        inputs = batch[0].to(device, dtype=torch.float32)
        labels = batch[1].to(device, dtype=torch.long)
        validate_partial_labels(labels, old_count, old_count + new_count)
        features = teacher.features(inputs)
        # Resize logits before softmax to match author select(), not probabilities.
        logits = teacher.classify(features)
        logits = F.interpolate(logits, (size, size), mode='bilinear', align_corners=False)
        probabilities = logits.softmax(1).permute(0, 2, 3, 1).reshape(-1, old_count)
        features = F.interpolate(features, (size, size), mode='bilinear', align_corners=False)
        features = features.permute(0, 2, 3, 1).reshape(-1, weights.shape[1])
        labels = F.interpolate(labels[:, None].float(), (size, size), mode='nearest')
        labels = labels.long().flatten()
        for novel in range(new_count):
            mask = labels == old_count + novel
            selected = features[mask]
            if not selected.numel():
                continue
            probs = probabilities[mask]
            for previous in range(old_count):
                positive_channels = (selected * weights[previous] > 0).to(weights.dtype)
                bucket[novel, previous].add_((positive_channels * probs[:, previous:previous+1]).sum(0))
            counts[novel] += selected.shape[0]
    if (counts == 0).any():
        missing = (counts == 0).nonzero().flatten().add(old_count).tolist()
        raise ValueError('NeST needs current training pixels for each novel class; missing IDs: '
                         + str(missing))
    return bucket / counts[:, None, None]


class ClassifierTransform(nn.Module):
    def __init__(self, old_weights, old_bias, similarity):
        super().__init__()
        self.register_buffer('old_weights', old_weights.detach().clone())
        self.register_buffer('old_bias', old_bias.detach().clone())
        self.importance = nn.Parameter(similarity.detach().clone())
        self.projection = nn.Parameter(similarity.sum(-1).softmax(-1))
        self.background_importance = nn.Parameter(torch.ones_like(old_weights[:1]))
        self.background_projection = nn.Parameter(old_weights.new_ones(1, 1))
        self.new_bias = nn.Parameter(old_bias[:1].repeat(similarity.shape[0]))

    def generate(self):
        novel = torch.einsum('no,nod,od->nd', self.projection,
                             self.importance, self.old_weights)
        background = (self.background_importance * self.old_weights[:1]
                      * self.background_projection)
        return novel, background, self.new_bias

    def forward(self, features):
        novel, background, bias = self.generate()
        weights = torch.cat([background, self.old_weights[1:], novel], 0)
        biases = torch.cat([self.old_bias, bias], 0)
        return F.conv2d(features, weights[:, :, None, None], biases)


def norm_matched(value, target_norm, vectorwise=False):
    norm = value.norm(dim=1).mean() if vectorwise else value.norm()
    # Zero-norm transformations/biases are possible on degenerate synthetic
    # inputs. Do not divide by zero or invent a random replacement classifier.
    if norm <= 1e-12:
        return value
    return value * (target_norm / norm)


class Nest(ClassIncrementalModel):
    NAME = 'nest'

    def __init__(self, backbone, loss, args, transform):
        if (args.nest_kd_weight < 0 or args.nest_teacher_scale <= 0 or
                args.nest_pre_epochs < 1 or args.nest_pre_lr_scale <= 0 or
                args.nest_pre_weight_decay < 0 or args.nest_selection_size < 1):
            raise ValueError('Invalid NeST coefficient, pre-tuning budget, or selection size.')
        super().__init__(backbone, loss, args, transform)

    def initialize_new_classes(self, dataset):
        if self.teacher is None:
            return
        # Teacher is exactly the pre-increment feature extractor. Using it
        # in eval/no_grad also prevents accidental BatchNorm-state updates.
        similarity = cross_task_similarity(self.teacher, dataset.train_loader,
                                            self.old_count, self.new_count,
                                            self.device, self.args.nest_selection_size)
        old_weights = self.teacher.classifier.weight[:self.old_count].flatten(1)
        old_bias = self.teacher.classifier.bias[:self.old_count]
        transform = ClassifierTransform(old_weights, old_bias, similarity).to(self.device)
        optimizer = torch.optim.SGD(transform.parameters(),
                                    lr=self.args.lr * self.args.nest_pre_lr_scale,
                                    momentum=0.9, nesterov=True,
                                    weight_decay=self.args.nest_pre_weight_decay)
        total_steps = self.args.nest_pre_epochs * len(dataset.train_loader)
        if not total_steps:
            raise ValueError('NeST cannot pre-tune on an empty current-task loader.')
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0., 1 - step / total_steps) ** 0.9)
        last_loss = None
        steps = 0
        for _ in range(self.args.nest_pre_epochs):
            for batch in dataset.train_loader:
                inputs = batch[0].to(self.device, dtype=torch.float32)
                labels = batch[1].to(self.device, dtype=torch.long)
                validate_partial_labels(labels, self.old_count, self.old_count + self.new_count)
                with torch.no_grad():
                    features = self.teacher.features(inputs)
                optimizer.zero_grad(set_to_none=True)
                logits = transform(features)
                loss = unbiased_cross_entropy(logits, labels, self.old_count)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite NeST classifier pre-tuning loss.')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in transform.parameters()):
                    raise FloatingPointError('Non-finite NeST pre-tuning gradient.')
                optimizer.step()
                scheduler.step()
                steps += 1
                last_loss = float(loss.detach())
        with torch.no_grad():
            new_weights, background, new_bias = transform.generate()
            background = norm_matched(background, old_weights[0].norm())
            new_weights = norm_matched(new_weights, old_weights.norm(dim=1).mean(),
                                       vectorwise=True)
            new_bias = norm_matched(new_bias, old_bias.norm())
            classifier = self.net.classifier
            classifier.weight[0].copy_(background[0, :, None, None])
            end = self.old_count + self.new_count
            classifier.weight[self.old_count:end].copy_(new_weights[:, :, None, None])
            classifier.bias[self.old_count:end].copy_(new_bias)
        self.initialization_info = {
            'type': 'cross_task_similarity_and_classifier_pre_tuning',
            'pre_epochs': self.args.nest_pre_epochs, 'pre_steps': steps,
            'last_pre_loss': last_loss, 'similarity': similarity.detach().cpu(),
            'learned_importance': transform.importance.detach().cpu(),
            'learned_projection': transform.projection.detach().cpu()}
        # Transform/optimizer are local temporaries; no transformation parameters
        # are added to the deployed network or the main SGD optimizer.

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        logits = self.net.logits(inputs)
        supervised = self.supervised_loss(logits, labels)
        kd = logits.new_zeros(())
        if self.teacher is not None:
            kd = unbiased_distillation(logits, self.teacher_logits(inputs),
                                       self.args.nest_teacher_scale)
        total = supervised + self.args.nest_kd_weight * kd
        return self.optimization_step(total, supervised=supervised, unkd=kd)
