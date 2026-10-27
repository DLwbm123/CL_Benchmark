"""TED, MedIA 2024: SKA + AKT + GUGF, adapted to the shared 2D backbone.

Mechanisms follow PerceptionComputingLab/TED_DCMIS (pinned in docs).
No input replay, no synthetic medical data generator, no future-task access.
"""
import torch
from torch.nn import functional as F

from models.utils.medical import (MedicalModel, dice_bce, medical_parser,
                                 require_single_head)


def get_parser():
    parser = medical_parser('Tri-enhanced distillation for Domain-CL')
    parser.set_defaults(lr=2e-4, lr_decay_rate=1.)
    parser.add_argument('--ted_optimizer', choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--ted_incremental_lr', type=float, default=1e-4)
    parser.add_argument('--ted_epoch_gamma', type=float, default=.99)
    parser.add_argument('--ted_bce_impl', choices=['logits', 'probability'], default='logits')
    parser.add_argument('--ted_lambda', type=float, default=0.001)
    parser.add_argument('--ted_gugf', action='store_true', default=True)
    parser.add_argument('--ted_no_gugf', dest='ted_gugf', action='store_false')
    parser.add_argument('--ted_supervised', choices=['dice_bce', 'benchmark'],
                        default='dice_bce')
    return parser


def adaptive_knowledge_transfer(student_logits, teacher_probs, gugf=True):
    """Literal upstream LossAKT, including its *global batch* broadcasting.

    With B>1, [B,H,W] uncertainty broadcasts against [B,1,H,W] errors.
    This gives [B,B,H,W], not per-pixel per-sample weighting. Retained explicitly
    for reference parity; do not silently 'fix' it and call it original TED.
    """
    teacher_probs = teacher_probs.detach()
    probs = student_logits.softmax(1)
    log_probs = torch.log(probs + 1e-5)
    target = teacher_probs.argmax(1, keepdim=True)
    selected = log_probs.gather(1, target)
    errors = (log_probs.argmax(1, keepdim=True) != target).to(probs.dtype).detach()
    if gugf:
        uncertainty = (1 - (teacher_probs * probs).sum(1)).detach()
    else:
        uncertainty = probs.new_ones(1)
    return -(uncertainty * errors * selected).sum() / (errors.sum() + 1)


class Ted(MedicalModel):
    NAME = 'ted'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        super().__init__(backbone, loss, args, transform)
        if args.ted_lambda < 0:
            raise ValueError('ted_lambda must be nonnegative.')
        if min(args.lr, args.ted_incremental_lr) <= 0 or not 0 < args.ted_epoch_gamma <= 1:
            raise ValueError('Invalid TED source optimizer settings.')
        if args.ted_optimizer == 'Adam':
            if args.lr_decay_rate != 1.:
                raise ValueError('TED source epoch decay requires lr_decay_rate=1.')
            self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, args.lr_decay_epoch, gamma=1.)

    @property
    def task_learning_rate(self):
        return self.args.lr if self.current_task == 0 else self.args.ted_incremental_lr

    def begin_task(self, dataset):
        super().begin_task(dataset)
        if self.args.ted_optimizer == 'Adam':
            self.opt.state.clear()  # author constructs a fresh Adam per domain

    def begin_epoch(self, epoch, n_epochs):
        if self.args.ted_optimizer == 'Adam':
            for group in self.opt.param_groups:
                group['lr'] = self.task_learning_rate * self.args.ted_epoch_gamma ** epoch

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        logits = self.net(inputs)
        supervised = (dice_bce(logits, labels, stable=self.args.ted_bce_impl == 'logits') if self.args.ted_supervised == 'dice_bce'
                      else self.loss(logits, labels).mean())
        image_kd = noise_kd = akt = logits.new_zeros(())
        if self.teacher is not None:
            self.teacher.eval()
            # Stochastic Knowledge Augmentation in the author code is N(0,1)
            # input with the same shape, not noise added to the medical image.
            noise = torch.randn_like(inputs)
            with torch.no_grad():
                old = self.teacher(inputs).softmax(1)
                old_noise = self.teacher(noise).softmax(1)
            image_kd = -(old * F.log_softmax(logits, 1)).sum(1).mean()
            noise_kd = -(old_noise * F.log_softmax(self.net(noise), 1)).sum(1).mean()
            akt = adaptive_knowledge_transfer(logits, old, self.args.ted_gugf)
        distillation = image_kd + noise_kd + akt
        total = supervised + self.args.ted_lambda * distillation
        return self.optimization_step(total, supervised=supervised,
                                      image_kd=image_kd, ska=noise_kd, akt=akt)
