"""Shared, explicit MiB-compatible protocol for the natural-image CISS adapters.

The original mib.py is unchanged. These adapters need balanced initialization,
a task-agnostic seen-class output space, and complete continual checkpoints.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.utils.medical import MedicalModel, medical_parser, require_single_head


def ciss_parser(description):
    parser = medical_parser(description)
    parser.set_defaults(dataset='seq-mmwhs')
    return parser


def validate_partial_labels(labels, old_count, seen_count):
    if labels.dtype != torch.long or labels.ndim != 3:
        raise ValueError('Expected long global class labels [B,H,W].')
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= seen_count)).any():
        raise ValueError('Current annotations contain invalid or future class IDs.')
    if old_count > 1 and ((labels > 0) & (labels < old_count)).any():
        raise ValueError('Old classes must be background-masked in this partial-label protocol.')


def unbiased_cross_entropy(logits, labels, old_count):
    """MiB UNCE; background pools background and old foreground probabilities."""
    denominator = logits.logsumexp(1)
    log_probs = logits - denominator[:, None]
    background = logits[:, :old_count].logsumexp(1) - denominator
    log_probs = torch.cat([background[:, None], log_probs[:, 1:]], 1)
    mapped = labels.clone()
    mapped[labels < old_count] = 0
    # Match the upstream trainer's unreduced loss followed by mean: ignored
    # pixels contribute zero, including the all-ignored edge case.
    return F.nll_loss(log_probs, mapped, ignore_index=255, reduction='none').mean()


def unbiased_distillation(student, teacher, scale=1.):
    """MiB UNKD, including the upstream division by old channel count."""
    old_count = teacher.shape[1]
    denominator = student.logsumexp(1)
    student_bg = torch.cat([student[:, :1], student[:, old_count:]], 1)
    bg_log_prob = student_bg.logsumexp(1) - denominator
    old_log_prob = student[:, 1:old_count] - denominator[:, None]
    targets = (teacher.detach() * scale).softmax(1)
    value = targets[:, 0] * bg_log_prob + (targets[:, 1:] * old_log_prob).sum(1)
    return -value.mean() / old_count


class SeenClassResUnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.classifier = backbone.last
        if self.classifier.bias is None:
            # Balanced initialization needs classifier bias. Preserve all of
            # the supplied random weights and do not consume another RNG draw.
            self.classifier.bias = nn.Parameter(torch.zeros(
                self.classifier.out_channels, device=self.classifier.weight.device,
                dtype=self.classifier.weight.dtype))
        backbone.last = nn.Identity()
        self.backbone = backbone
        self.register_buffer('seen_classes', torch.tensor(1, dtype=torch.long))
        self.register_buffer('frozen_rows', torch.tensor(0, dtype=torch.long))

    @property
    def class_num(self):
        return self.classifier.out_channels

    def features(self, x):
        return self.backbone.features(x)

    def classify(self, features, count=None):
        count = int(self.seen_classes) if count is None else count
        weights = self.classifier.weight[:count]
        biases = self.classifier.bias[:count]
        frozen = min(int(self.frozen_rows), count)
        if frozen:
            # Freeze old rows, while still propagating loss into the features.
            weights = torch.cat([weights[:frozen].detach(), weights[frozen:]], 0)
            biases = torch.cat([biases[:frozen].detach(), biases[frozen:]], 0)
        return F.conv2d(features, weights, biases)

    def logits(self, x):
        return self.classify(self.features(x))

    def forward(self, x, task_index=None):
        logits = self.logits(x)
        missing = self.class_num - logits.shape[1]
        if missing:
            logits = torch.cat([logits, logits.new_full(
                (logits.shape[0], missing, *logits.shape[-2:]), -1e4)], 1)
        return logits


class ClassIncrementalModel(MedicalModel):
    COMPATIBILITY = ['class-il']
    NETWORK_TYPE = SeenClassResUnet

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        super().__init__(self.NETWORK_TYPE(backbone), loss, args, transform)
        self.old_count = 1
        self.new_count = 0
        self.task_classes = []
        self.initialization_info = {}

    def knowledge_state(self):
        return {'old_count': self.old_count, 'new_count': self.new_count,
                'task_classes': self.task_classes,
                'initialization_info': self.initialization_info}

    def restore_knowledge(self, state):
        self.old_count = state['old_count']
        self.new_count = state['new_count']
        self.task_classes = state['task_classes']
        self.initialization_info = state['initialization_info']

    def initialize_new_classes(self, dataset):
        """MiB balanced background split, applied only when new labels arrive."""
        if self.teacher is None:
            return
        classifier = self.net.classifier
        with torch.no_grad():
            background_w = self.teacher.classifier.weight[0].clone()
            background_b = self.teacher.classifier.bias[0] - math.log(self.new_count + 1)
            end = self.old_count + self.new_count
            classifier.weight[self.old_count:end].copy_(background_w)
            classifier.bias[self.old_count:end].copy_(background_b)
            classifier.bias[0].copy_(background_b)
        self.initialization_info = {'type': 'balanced_background_split'}

    def begin_task(self, dataset):
        self.check_task(dataset)
        counts = dataset.N_CLASSES_PER_TASK
        if (not isinstance(counts, (list, tuple)) or not counts or
                any(not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in counts)
                or sum(counts) + 1 != self.net.class_num):
            raise ValueError('Foreground task counts must match the global classifier space.')
        if not 0 <= self.current_task < len(counts):
            raise ValueError('Task index is outside the declared stream.')
        expected_old = 1 + sum(counts[:self.current_task])
        if int(self.net.seen_classes) != expected_old:
            raise ValueError('Task position and learned-class state disagree; load a complete checkpoint.')
        if self.current_task and self.teacher is None:
            raise ValueError('Incremental tasks require a frozen previous-task teacher.')
        if self.teacher is not None and int(self.teacher.seen_classes) != expected_old:
            raise ValueError('Teacher label space and task position disagree.')
        self.old_count = expected_old
        self.new_count = counts[self.current_task]
        self.task_classes = list(counts)
        self.initialize_new_classes(dataset)
        self.net.seen_classes.fill_(self.old_count + self.new_count)
        self.net.frozen_rows.zero_()
        super().begin_task(dataset)

    def supervised_loss(self, logits, labels):
        validate_partial_labels(labels, self.old_count if self.current_task else 1,
                                int(self.net.seen_classes))
        if self.teacher is None:
            return F.cross_entropy(logits, labels, ignore_index=255, reduction='none').mean()
        return unbiased_cross_entropy(logits, labels, self.old_count)

    def teacher_logits(self, inputs):
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher.logits(inputs)
