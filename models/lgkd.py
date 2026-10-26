"""LGKD + MiB, ICCV 2023, single-backbone 2D ResUNet adaptation.

LabelGuidedKnowledgeDistillationLoss follows Ze-Yang/LGKD; see notices/docs
for attribution, the pinned source, CC BY-NC terms, and backbone differences.
"""
import torch
from torch.nn import functional as F

from models.utils.class_incremental import ClassIncrementalModel, ciss_parser


def get_parser():
    parser = ciss_parser('Label-guided knowledge distillation + MiB')
    # Paper ablation selects lambda_b=5 and lambda_n=1 (rather than the
    # repository parser's generic 10/2 defaults for other experiment scripts).
    parser.add_argument('--lgkd_previous_weight', type=float, default=5.)
    parser.add_argument('--lgkd_novel_weight', type=float, default=1.)
    parser.add_argument('--lgkd_teacher_scale', type=float, default=1.)
    return parser


def label_guided_targets(teacher_logits, labels, seen_count, scale=1.):
    """Expand teacher space and move bg mass to the GT novel class only."""
    old_count = teacher_logits.shape[1]
    teacher = (teacher_logits.detach() * scale).softmax(1)
    expanded = F.pad(teacher, (0, 0, 0, 0, 0, seen_count - old_count))
    novel = (labels >= old_count) & (labels < seen_count) & (labels != 255)
    safe_labels = labels.clamp(0, seen_count - 1)
    destination = F.one_hot(safe_labels, seen_count).permute(0, 3, 1, 2)
    moved = teacher[:, :1] * novel[:, None]
    expanded = expanded + destination * moved
    expanded[:, 0] = torch.where(novel, torch.zeros_like(expanded[:, 0]), expanded[:, 0])
    return expanded


def label_guided_distillation(student, teacher, labels, previous_weight=5.,
                              novel_weight=1., scale=1.):
    valid = labels != 255
    if not valid.any():
        return student.sum() * 0
    expanded = label_guided_targets(teacher, labels, student.shape[1], scale)
    cross_entropy = -(expanded * F.log_softmax(student, 1)).sum(1)
    novel = (labels >= teacher.shape[1]) & valid
    weights = torch.where(novel, novel_weight, previous_weight)
    # The author loss concatenates previous/novel subsets, then averages
    # over all non-ignored pixels; do not independently normalize each subset.
    return (cross_entropy * weights)[valid].mean()


class Lgkd(ClassIncrementalModel):
    NAME = 'lgkd'

    def __init__(self, backbone, loss, args, transform):
        if (args.lgkd_previous_weight < 0 or args.lgkd_novel_weight < 0 or
                args.lgkd_teacher_scale <= 0):
            raise ValueError('LGKD weights must be nonnegative and teacher scale positive.')
        super().__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        logits = self.net.logits(inputs)
        supervised = self.supervised_loss(logits, labels)
        kd = logits.new_zeros(())
        if self.teacher is not None:
            old = self.teacher_logits(inputs)
            kd = label_guided_distillation(logits, old, labels,
                                          self.args.lgkd_previous_weight,
                                          self.args.lgkd_novel_weight,
                                          self.args.lgkd_teacher_scale)
        return self.optimization_step(supervised + kd, supervised=supervised, lgkd=kd)
