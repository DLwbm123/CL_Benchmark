"""Task-local PML/MAD and shared-feature SKD for Organ-CL, explicit adaptation."""
import torch
from models.ldka import (get_parser as class_parser, progressive_margin_loss,
                         decoder_input, fixed_batch_norm, smooth_knowledge_losses,
                         competition_counts, misclassification_decoupling)
from models.utils.organ import (OrganModel, OrganHeads, organ_parser,
                                validate_organ_backbone, validate_binary_labels)


def get_parser():
    parser = organ_parser(class_parser())
    parser.set_defaults(ldka_grad_clip=0.)  # this adapter has its own incremental-only guard
    parser.add_argument('--ldka_organ_grad_clip', type=float, default=1.0,
                        help='Incremental-only global gradient-norm guard for the Organ adapter; 0 disables')
    parser.add_argument('--ldka_organ_base_grad_clip', type=float, default=0.0,
                        help='Base-task gradient guard for the Organ adapter; 0 disables')
    return parser


class LdkaOrgan(OrganModel):
    NAME = 'ldka_organ'

    def __init__(self, backbone, loss, args, transform):
        validate_organ_backbone(backbone, args)
        if (min(args.lr, args.ldka_incremental_lr) <= 0 or not 0 < args.ldka_progress_epsilon < .5 or
                min(args.ldka_dss_weight, args.ldka_perturb_weight, args.ldka_mad_weight) < 0):
            raise ValueError('Invalid LDKA Organ-CL objective settings.')
        if args.ldka_pseudo_threshold != 0:
            raise ValueError('Organ-CL LDKA uses current binary GT; class-space pseudo-label option is inapplicable.')
        super().__init__(OrganHeads(backbone), loss, args, transform)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=.9, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, args.lr_decay_epoch, args.lr_decay_rate)
        self.register_buffer('epoch_confusion', torch.zeros(args.organ_tasks, 2, 2, dtype=torch.long))
        self.register_buffer('competition', torch.zeros(args.organ_tasks, 2, 2))
        self.epoch, self.total_epochs, self.progress = 0, max(1, args.n_epochs), .5
        self.guard_stats = {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}
        if min(args.ldka_organ_grad_clip, args.ldka_organ_base_grad_clip) < 0:
            raise ValueError('Organ gradient-norm guard must be nonnegative.')

    @property
    def gradient_clip_max_norm(self):
        return self.args.ldka_organ_grad_clip if self.current_task else self.args.ldka_organ_base_grad_clip

    @property
    def task_learning_rate(self):
        return self.args.lr if self.current_task == 0 else self.args.ldka_incremental_lr

    def knowledge_state(self):
        return {**super().knowledge_state(), 'epoch': self.epoch, 'total_epochs': self.total_epochs,
                'progress': self.progress, 'guard_stats': dict(self.guard_stats)}

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.epoch, self.total_epochs, self.progress = state['epoch'], state['total_epochs'], state['progress']
        self.guard_stats = dict(state.get('guard_stats', {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}))

    def optimization_step(self, loss, **parts):
        value = super().optimization_step(loss, **parts)
        if self.gradient_clip_max_norm > 0:
            norm = self.last_losses['gradient_norm_before_clip']
            self.guard_stats['steps'] += 1
            self.guard_stats['clipped'] += int(norm > self.gradient_clip_max_norm)
            self.guard_stats['peak_preclip_norm'] = max(self.guard_stats['peak_preclip_norm'], norm)
        return value

    def begin_task(self, dataset):
        super().begin_task(dataset)
        self.begin_epoch(0, max(1, self.args.n_epochs))
        for group in self.opt.param_groups:
            group['lr'] = self.task_learning_rate

    def begin_epoch(self, epoch, total_epochs):
        if not 0 <= epoch < total_epochs:
            raise ValueError('Invalid Organ-CL epoch.')
        self.epoch, self.total_epochs = epoch, total_epochs
        self.progress = (epoch+1)/(total_epochs+1)

    def end_epoch(self):
        with torch.no_grad():
            counts = self.epoch_confusion[self.current_task]
            self.competition[self.current_task].copy_(counts/counts.sum(1, keepdim=True).clamp_min(1))
            counts.zero_()

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        validate_binary_labels(labels)
        self.opt.zero_grad(set_to_none=True)
        feature = decoder_input(self.net.backbone, inputs)
        response = self.net.backbone.up_residual_conv3(feature)
        logits = self.net.classify(response, self.current_task)
        supervised, _ = progressive_margin_loss(logits, labels, self.progress, self.args.ldka_progress_epsilon)
        dss = perturb = logits.sum()*0
        if self.teacher is not None and (labels != 255).any():
            with torch.no_grad():
                previous = self.teacher.features(inputs)
            mask = (labels != 255)[:, None].to(feature)
            decoder = self.net.backbone.up_residual_conv3
            with fixed_batch_norm(decoder):
                dss, perturb = smooth_knowledge_losses(feature, previous*mask, lambda f: decoder(f)*mask)
        axes = self.net.heads[str(self.current_task)].weight.flatten(1)
        mad = misclassification_decoupling(axes, self.competition[self.current_task])
        if not (labels != 255).any():
            mad = mad*0
        total = supervised+self.args.ldka_dss_weight*dss+self.args.ldka_perturb_weight*perturb+self.args.ldka_mad_weight*mad
        value = self.optimization_step(total, supervised=supervised, dss=dss, perturbation=perturb, mad=mad, progress=self.progress)
        with torch.no_grad():
            self.epoch_confusion[self.current_task].add_(competition_counts(logits, labels))
        return value
