"""EWF + MiB, CVPR 2023; fusion is once per endpoint, not per-batch EMA.

Reference: schuy1er/EWF_official, run.py default (non-threshold) fusion branch.
"""
import math

import torch

from models.utils.class_incremental import (ClassIncrementalModel, ciss_parser,
                                            unbiased_distillation)


def get_parser():
    parser = ciss_parser('Endpoints weight fusion + MiB')
    parser.add_argument('--ewf_kd_weight', type=float, default=10.)
    parser.add_argument('--ewf_teacher_scale', type=float, default=1.)
    parser.add_argument('--ewf_freeze_old', action='store_true', default=True)
    parser.add_argument('--ewf_train_old', dest='ewf_freeze_old', action='store_false')
    return parser


class Ewf(ClassIncrementalModel):
    NAME = 'ewf'

    def __init__(self, backbone, loss, args, transform):
        if args.ewf_kd_weight < 0 or args.ewf_teacher_scale <= 0:
            raise ValueError('EWF KD weight must be nonnegative; teacher scale positive.')
        super().__init__(backbone, loss, args, transform)

    def begin_task(self, dataset):
        super().begin_task(dataset)
        if self.teacher is not None and self.args.ewf_freeze_old:
            self.net.frozen_rows.fill_(self.old_count)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        logits = self.net.logits(inputs)
        supervised = self.supervised_loss(logits, labels)
        kd = logits.new_zeros(())
        if self.teacher is not None:
            kd = unbiased_distillation(logits, self.teacher_logits(inputs),
                                       self.args.ewf_teacher_scale)
        total = supervised + self.args.ewf_kd_weight * kd
        return self.optimization_step(total, supervised=supervised, unkd=kd)

    def fuse_endpoint(self):
        if self.teacher is None:
            return
        alpha = math.sqrt(self.new_count / (self.old_count + self.new_count))
        previous = self.teacher.state_dict()
        with torch.no_grad():
            for name, value in self.net.state_dict().items():
                if name in ('classifier.weight', 'classifier.bias'):
                    value[:self.old_count].mul_(alpha).add_(
                        previous[name][:self.old_count], alpha=1 - alpha)
                elif name.startswith('backbone.') and value.is_floating_point():
                    value.mul_(alpha).add_(previous[name], alpha=1 - alpha)
                # Keep integer BN counters and label-space metadata current.
                # Future classifier rows are unlearned; new rows have no old
                # endpoint counterpart and must not be averaged with random ones.
        self.initialization_info['endpoint_new_weight'] = alpha

    def end_task(self, dataset):
        self.require_open_task()
        self.fuse_endpoint()
        self.net.frozen_rows.zero_()
        # The *fused* endpoint becomes the next teacher and saved network.
        super().end_task(dataset)
