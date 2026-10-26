"""Independent task queries and task-origin accumulated teachers for Organ-CL."""
import torch
from torch import nn
from torch.nn import functional as F
from models.cit import (get_parser as class_parser, supervised_cit_losses, accumulated_cit_losses)
from models.utils.cit_head import CitResUnet
from models.utils.medical import frozen_copy
from models.utils.cit_optimizer import configure_cit_optimizer, cit_optimization_step
from models.utils.organ import (OrganModel, organ_parser, validate_organ_backbone,
                                validate_binary_labels, checked_task_id)


def get_parser():
    p = organ_parser(class_parser())
    p.set_defaults(cit_zero_gt=False)
    return p


class OrganCitNet(CitResUnet):
    def forward(self, x, task_index=None):
        index = checked_task_id(task_index, self.class_num)
        # Heads are preallocated for the legacy forward-transfer evaluation.
        # The selected query defines its own bg, independent of other organs.
        outputs = self.components(x, count=index+1)
        presence = outputs['presence'][:, index].softmax(-1)[:, 0, None, None, None]
        foreground = presence*outputs['masks'][:, index:index+1].sigmoid()
        foreground = F.interpolate(foreground, x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([1-foreground, foreground], 1).clamp_min(1e-12).log()


class CitOrgan(OrganModel):
    NAME = 'cit_organ'
    REQUIRES_TEACHER = False

    def __init__(self, backbone, loss, args, transform):
        width = validate_organ_backbone(backbone, args)
        if (args.cit_heads < 1 or args.cit_feature_stride < 1 or not 0 <= args.cit_dropout < 1 or
                args.cit_teacher_scale <= 0 or args.cit_zero_gt or
                min(args.cit_class_weight, args.cit_focal_weight, args.cit_dice_weight,
                    args.cit_presence_kd_weight, args.cit_mask_kd_weight) < 0):
            raise ValueError('Invalid CIT Organ settings; zero_gt must be false for independent organ labels.')
        net = OrganCitNet(backbone, args.cit_heads, args.cit_dropout, args.cit_feature_stride,
                          foreground_classes=args.organ_tasks, feature_width=width)
        super().__init__(net, loss, args, transform)
        configure_cit_optimizer(self)
        self.teachers = nn.ModuleList()

    def optimization_step(self,loss,**parts):
        return cit_optimization_step(self,loss,**parts)

    def knowledge_state(self):
        return {**super().knowledge_state(), 'teacher_tasks': len(self.teachers)}

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        if state['teacher_tasks'] != len(self.teachers):
            raise ValueError('Organ CIT teacher bank and checkpoint disagree.')

    def load_state_dict(self, state_dict, strict=True):
        indices = {int(k.split('.')[1]) for k in state_dict if k.startswith('teachers.')}
        if indices != set(range(len(indices))):
            raise ValueError('Organ CIT teacher indices must be consecutive.')
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
        if len(self.teachers) != self.current_task:
            raise ValueError('Organ CIT requires one original-task teacher per completed task.')
        super().begin_task(dataset)
        for teacher in self.teachers:
            teacher.eval()

    @torch.no_grad()
    def accumulated_targets(self, inputs):
        targets = {'presence': [], 'masks': []}
        for index, teacher in enumerate(self.teachers):
            teacher.eval()
            output = teacher.components(inputs, count=index+1)
            for key in targets:
                targets[key].append(output[key][:, index:index+1])
        return {key: torch.cat(value, 1) for key, value in targets.items()}

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        validate_binary_labels(labels)
        self.opt.zero_grad(set_to_none=True)
        outputs = self.net.components(inputs, count=self.current_task+1)
        current = {key: value[:, self.current_task:self.current_task+1] for key, value in outputs.items()}
        cls, focal, dice = supervised_cit_losses(current, labels, 0, 1)
        presence = mask = cls*0
        if self.teachers and (labels != 255).any():
            targets = self.accumulated_targets(inputs)
            presence, mask = accumulated_cit_losses(outputs, targets, labels, self.current_task,
                                                   self.args.cit_teacher_scale, zero_gt=False)
        total = (self.args.cit_class_weight*cls+self.args.cit_focal_weight*focal+
                 self.args.cit_dice_weight*dice+self.args.cit_presence_kd_weight*presence+
                 self.args.cit_mask_kd_weight*mask)
        return self.optimization_step(total, classification=cls, focal=focal, dice=dice,
                                       presence_kd=presence, mask_kd=mask)

    def end_task(self, dataset):
        self.require_open_task()
        self.teachers.append(frozen_copy(self.net))
        self.task_open = False
        self.current_task += 1
