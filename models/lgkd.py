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
    parser.add_argument('--lgkd_grad_clip', type=float, default=0.)
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
                args.lgkd_teacher_scale <= 0 or args.lgkd_grad_clip < 0):
            raise ValueError('LGKD weights must be nonnegative and teacher scale positive.')
        super().__init__(backbone, loss, args, transform)
        self.guard_stats = {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}

    @property
    def gradient_clip_max_norm(self):
        return self.args.lgkd_grad_clip if self.current_task else 0.

    def knowledge_state(self):
        return dict(super().knowledge_state(), guard_stats=dict(self.guard_stats))

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.guard_stats = dict(state.get('guard_stats',
                                {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}))

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
        if not torch.isfinite(supervised + kd):
            raise FloatingPointError('lgkd non-finite components: supervised={}, kd={}, input_finite={}, logits_finite={}'.format(
                float(supervised.detach()), float(kd.detach()),
                bool(torch.isfinite(inputs).all()), bool(torch.isfinite(logits).all())))
        value = self.optimization_step(supervised + kd, supervised=supervised, lgkd=kd)
        if self.gradient_clip_max_norm > 0:
            norm = self.last_losses['gradient_norm_before_clip']
            self.guard_stats['steps'] += 1
            self.guard_stats['clipped'] += int(norm > self.gradient_clip_max_norm)
            self.guard_stats['peak_preclip_norm'] = max(self.guard_stats['peak_preclip_norm'], norm)
        return value
