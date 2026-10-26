"""Task-routed image/text-conditioned organ heads with independent pseudo masks."""
import torch
from torch.nn import functional as F
from models.clip_cis import get_parser as class_parser, read_embeddings
from models.utils.clip_heads import ClipHeadResUnet
from models.utils.organ import (OrganModel, organ_parser, validate_organ_backbone,
                                validate_binary_labels, checked_task_id)


def get_parser():
    return organ_parser(class_parser())


def independent_organ_targets(labels, teacher_logits, threshold):
    """New organ GT never overrides a different old task's foreground labels."""
    current = (labels == 1)[:, None].float()
    if teacher_logits is None:
        return current
    return torch.cat([(teacher_logits.detach().sigmoid() >= threshold).to(current), current], 1)


def masked_independent_loss(logits, targets, labels):
    valid = (labels != 255)[:, None].expand_as(logits)
    if not valid.any():
        return logits.sum()*0
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')[valid].mean()
    p = (logits.sigmoid()*valid).flatten(2)
    t = (targets*valid).flatten(2)
    usable = valid.flatten(2).any(2)
    dice = 1-2*(p*t).sum(2)/(p.sum(2)+t.sum(2)+1)
    return bce+dice[usable].mean()


class OrganClipNet(ClipHeadResUnet):
    def forward(self, x, task_index=None):
        index = checked_task_id(task_index, self.class_num)
        fg = self.foreground(x, indices=[index])
        return torch.cat([torch.zeros_like(fg), fg], 1)


class ClipCisOrgan(OrganModel):
    NAME = 'clip_cis_organ'

    def __init__(self, backbone, loss, args, transform):
        width = validate_organ_backbone(backbone, args)
        embeddings, names = read_embeddings(args.clip_embeddings, args.organ_tasks)
        if names != args.organ_task_names:
            raise ValueError('CLIP organ names must exactly match the declared task order.')
        if not 0 < args.clip_pseudo_threshold < 1:
            raise ValueError('Invalid CLIP Organ pseudo-mask threshold.')
        super().__init__(OrganClipNet(backbone, embeddings, feature_width=width), loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        validate_binary_labels(labels)
        self.opt.zero_grad(set_to_none=True)
        previous = None
        if self.teacher is not None:
            with torch.no_grad():
                previous = self.teacher.foreground(inputs, self.current_task)
        targets = independent_organ_targets(labels, previous, self.args.clip_pseudo_threshold)
        logits = self.net.foreground(inputs, self.current_task+1)
        value = masked_independent_loss(logits, targets, labels)
        return self.optimization_step(value, independent_segmentation=value)
