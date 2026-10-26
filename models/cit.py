"""CIT, Pattern Recognition 2025 (preprint 2024), ResUNet-2D adaptation.

Retains class-independent attention and accumulated original-task teachers.
Does not replace the method with a single last-model sigmoid distillation loss.
Source versions, supervised-loss details and background adaptation are documented.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.utils.cit_head import CitResUnet
from models.utils.class_incremental import ciss_parser, validate_partial_labels
from models.utils.medical import MedicalModel, frozen_copy, require_single_head
from models.utils.cit_optimizer import configure_cit_optimizer, cit_optimization_step


def get_parser():
    parser = ciss_parser('Class-independent transformation with accumulative distillation')
    parser.set_defaults(lr=2e-5)
    parser.add_argument('--cit_optimizer',choices=['AdamW','SGD'],default='AdamW')
    parser.add_argument('--cit_weight_decay',type=float,default=.01)
    parser.add_argument('--cit_head_lr_mult',type=float,default=10.)
    parser.add_argument('--cit_heads', type=int, default=8)
    parser.add_argument('--cit_dropout', type=float, default=0.1)
    parser.add_argument('--cit_feature_stride', type=int, default=8)
    parser.add_argument('--cit_class_weight', type=float, default=1.)
    parser.add_argument('--cit_focal_weight', type=float, default=20.)
    parser.add_argument('--cit_dice_weight', type=float, default=1.)
    parser.add_argument('--cit_presence_kd_weight', type=float, default=5.)
    parser.add_argument('--cit_mask_kd_weight', type=float, default=10.)
    parser.add_argument('--cit_teacher_scale', type=float, default=1.)
    parser.add_argument('--cit_zero_gt', action='store_true', default=True)
    parser.add_argument('--cit_keep_gt', dest='cit_zero_gt', action='store_false')
    return parser


def supervised_cit_losses(outputs, labels, old_count, seen_count):
    """Direct query-label assignment and author-form class/focal/Dice losses.

    old_count/seen_count count foreground only; medical labels are one-based.
    Explicit ignore=255 is excluded; current background remains mask negatives.
    """
    presence = outputs['presence'][:, old_count:seen_count]
    masks = outputs['masks'][:, old_count:seen_count]
    classes = torch.arange(old_count + 1, seen_count + 1, device=labels.device)
    binary = labels[:, None] == classes[None, :, None, None]
    present = binary.flatten(2).any(2)
    valid = labels != 255
    usable_images = valid.flatten(1).any(1)
    zero = presence.sum() * 0 + masks.sum() * 0
    if usable_images.any():
        targets = (~present[usable_images]).long()  # present=0; no-object=1
        classification = F.cross_entropy(presence[usable_images].transpose(1, 2),
                                         targets, weight=presence.new_tensor([1., .1]))
    else:
        classification = zero
    if not present.any():
        return classification, zero, zero
    masks = F.interpolate(masks, labels.shape[-2:], mode='bilinear', align_corners=False)
    predictions = masks[present].flatten(1)
    targets = binary[present].to(predictions.dtype).flatten(1)
    valid_masks = valid[:, None].expand_as(binary)[present].flatten(1).to(predictions.dtype)
    probabilities = predictions.sigmoid()
    ce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    pt = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha = .25 * targets + .75 * (1 - targets)
    focal = ((ce * (1 - pt).square() * alpha * valid_masks).sum(1)
             / valid_masks.sum(1).clamp_min(1)).mean()
    p = probabilities * valid_masks
    t = targets * valid_masks
    dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    return classification, focal, dice


def accumulated_cit_losses(outputs, targets, labels, old_count, scale=1., zero_gt=True):
    if not old_count:
        zero = outputs['presence'].sum() * 0
        return zero, zero
    presence_target = (targets['presence'].detach() * scale).sigmoid()
    cls_kd = F.binary_cross_entropy_with_logits(outputs['presence'][:, :old_count], presence_target)
    pseudo = (targets['masks'].detach() * scale).sigmoid()
    pseudo = F.interpolate(pseudo, labels.shape[-2:], mode='bilinear', align_corners=False)
    if zero_gt:
        # Source non-ignored GT is current foreground. Medical label 0 is
        # partial/unlabelled background, so it must not erase all old targets.
        current_gt = (labels > 0) & (labels != 255)
        pseudo = pseudo.masked_fill(current_gt[:, None], 0.)
    prediction = F.interpolate(outputs['masks'][:, :old_count], labels.shape[-2:],
                               mode='bilinear', align_corners=False)
    valid = (labels != 255)[:, None].expand_as(prediction)
    mask_losses = F.binary_cross_entropy_with_logits(prediction, pseudo, reduction='none')
    mask_kd = mask_losses[valid].mean() if valid.any() else prediction.sum() * 0
    return cls_kd, mask_kd


class Cit(MedicalModel):
    NAME = 'cit'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        if args.cit_heads < 1 or args.cit_feature_stride < 1 or not 0 <= args.cit_dropout < 1:
            raise ValueError('Invalid CIT attention geometry or dropout.')
        weights = (args.cit_class_weight, args.cit_focal_weight, args.cit_dice_weight,
                   args.cit_presence_kd_weight, args.cit_mask_kd_weight)
        if any(w < 0 for w in weights) or args.cit_teacher_scale <= 0:
            raise ValueError('CIT loss weights must be nonnegative and teacher scale positive.')
        net = CitResUnet(backbone, args.cit_heads, args.cit_dropout, args.cit_feature_stride)
        super().__init__(net, loss, args, transform)
        configure_cit_optimizer(self)
        self.teachers = nn.ModuleList()
        self.boundaries = []
        self.task_classes = []
        self.old_count = 0
        self.new_count = 0

    def optimization_step(self,loss,**parts):
        return cit_optimization_step(self,loss,**parts)

    def knowledge_state(self):
        return {'boundaries': self.boundaries, 'task_classes': self.task_classes,
                'old_count': self.old_count, 'new_count': self.new_count}

    def restore_knowledge(self, state):
        self.boundaries = state['boundaries']
        self.task_classes = state['task_classes']
        self.old_count = state['old_count']
        self.new_count = state['new_count']
        if len(self.boundaries) != len(self.teachers):
            raise ValueError('CIT checkpoint teacher bank and task boundaries disagree.')

    def load_state_dict(self, state_dict, strict=True):
        indices = {int(k.split('.')[1]) for k in state_dict if k.startswith('teachers.')}
        if indices != set(range(len(indices))):
            raise ValueError('CIT checkpoint teacher indices are not consecutive.')
        self.teachers = nn.ModuleList([frozen_copy(self.net) for _ in sorted(indices)])
        result = super().load_state_dict(state_dict, strict=strict)
        for teacher in self.teachers:
            teacher.eval()
        return result

    def train(self, mode=True):
        super().train(mode)
        for teacher in self.teachers:
            teacher.eval()
        return self

    def begin_task(self, dataset):
        self.check_task(dataset)
        counts = dataset.N_CLASSES_PER_TASK
        if (not isinstance(counts, (tuple, list)) or not counts or
                any(not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in counts)
                or sum(counts) != self.net.class_num):
            raise ValueError('CIT foreground task counts must match the query space.')
        if not 0 <= self.current_task < len(counts):
            raise ValueError('CIT task position is outside the declared stream.')
        self.old_count = sum(counts[:self.current_task])
        if (int(self.net.seen_classes) != self.old_count or
                len(self.teachers) != self.current_task or
                self.boundaries != [sum(counts[:i+1]) for i in range(self.current_task)]):
            raise ValueError('CIT task position, query space and accumulated teachers disagree.')
        self.new_count = counts[self.current_task]
        self.task_classes = list(counts)
        self.net.seen_classes.fill_(self.old_count + self.new_count)
        for teacher in self.teachers:
            teacher.eval()
        super().begin_task(dataset)
        self.opt.state.clear()

    @torch.no_grad()
    def accumulated_targets(self, inputs):
        collected = {'presence': [], 'masks': []}
        start = 0
        for teacher, end in zip(self.teachers, self.boundaries):
            teacher.eval()
            result = teacher.components(inputs)
            for key in collected:
                # Each category is sourced from the snapshot where it was
                # directly supervised, not repeatedly distilled later stages.
                collected[key].append(result[key][:, start:end])
            start = end
        if not self.teachers:
            return None
        return {key: torch.cat(values, 1) for key, values in collected.items()}

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        seen = int(self.net.seen_classes)
        validate_partial_labels(labels, self.old_count + 1, seen + 1)
        self.opt.zero_grad(set_to_none=True)
        outputs = self.net.components(inputs)
        classification, focal, dice = supervised_cit_losses(outputs, labels, self.old_count, seen)
        cls_kd = mask_kd = classification.new_zeros(())
        if self.teachers:
            targets = self.accumulated_targets(inputs)
            cls_kd, mask_kd = accumulated_cit_losses(outputs, targets, labels,
                                                   self.old_count,
                                                   self.args.cit_teacher_scale,
                                                   self.args.cit_zero_gt)
        total = (self.args.cit_class_weight * classification +
                 self.args.cit_focal_weight * focal + self.args.cit_dice_weight * dice +
                 self.args.cit_presence_kd_weight * cls_kd +
                 self.args.cit_mask_kd_weight * mask_kd)
        return self.optimization_step(total, classification=classification, focal=focal,
                                      dice=dice, presence_kd=cls_kd, mask_kd=mask_kd)

    def end_task(self, dataset):
        self.require_open_task()
        self.teachers.append(frozen_copy(self.net))
        self.boundaries.append(int(self.net.seen_classes))
        self.task_open = False
        self.current_task += 1
