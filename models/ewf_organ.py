"""Task-aware Organ-CL adaptation of EWF with task-wise logit distillation."""
import math
import torch
from models.ewf import get_parser as class_parser
from models.utils.organ import (OrganModel, OrganHeads, organ_parser,
                                validate_organ_backbone, validate_binary_labels,
                                binary_cross_entropy, old_head_distillation)


def get_parser():
    return organ_parser(class_parser())


class EwfOrgan(OrganModel):
    NAME = 'ewf_organ'

    def __init__(self, backbone, loss, args, transform):
        validate_organ_backbone(backbone, args)
        if args.ewf_kd_weight < 0 or args.ewf_teacher_scale <= 0:
            raise ValueError('Invalid EWF Organ-CL distillation settings.')
        super().__init__(OrganHeads(backbone), loss, args, transform)
        self.fusion_alpha = None

    def knowledge_state(self):
        return {**super().knowledge_state(), 'fusion_alpha': self.fusion_alpha}

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.fusion_alpha = state['fusion_alpha']

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        validate_binary_labels(labels)
        self.opt.zero_grad(set_to_none=True)
        features = self.net.features(inputs)
        supervised = binary_cross_entropy(self.net.classify(features, self.current_task), labels)
        kd = supervised*0
        if self.teacher is not None:
            with torch.no_grad():
                previous = self.teacher.features(inputs)
                targets = [self.teacher.classify(previous, i) for i in range(self.current_task)]
            students = [self.net.classify(features, i, self.args.ewf_freeze_old) for i in range(self.current_task)]
            kd = old_head_distillation(students, targets, labels, self.args.ewf_teacher_scale)
        return self.optimization_step(supervised+self.args.ewf_kd_weight*kd, supervised=supervised, old_head_kd=kd)

    def fuse_endpoint(self):
        if self.teacher is None:
            return
        # Each task contributes its own bg+fg head: 2 new / (2 old*t + 2).
        alpha = math.sqrt(1/(self.current_task+1))
        old = self.teacher.state_dict()
        with torch.no_grad():
            for name, value in self.net.state_dict().items():
                shared = name.startswith('backbone.')
                previous_head = name.startswith('heads.') and int(name.split('.')[1]) < self.current_task
                if (shared or previous_head) and value.is_floating_point():
                    value.mul_(alpha).add_(old[name], alpha=1-alpha)
        self.fusion_alpha = alpha

    def end_task(self, dataset):
        self.require_open_task()
        self.fuse_endpoint()
        super().end_task(dataset)
