"""LDKA, ICML 2026: paper/Algorithms 1-2 implementation on ResUNet.

Uses the published practical double-backprop surrogate, not a constant MSE
Hessian, and propagates perturbations through a nonlinear decoder block.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.utils.class_incremental import ClassIncrementalModel, ciss_parser, validate_partial_labels


def get_parser():
    p = ciss_parser('Learnability-driven knowledge assimilation')
    p.set_defaults(lr=1e-3)
    p.add_argument('--ldka_incremental_lr', type=float, default=1e-4)
    p.add_argument('--ldka_progress_epsilon', type=float, default=1e-3)
    p.add_argument('--ldka_pseudo_threshold', type=float, default=0.)
    p.add_argument('--ldka_dss_weight', type=float, default=1.)
    p.add_argument('--ldka_perturb_weight', type=float, default=1.)
    p.add_argument('--ldka_mad_weight', type=float, default=1.)
    p.add_argument('--ldka_grad_clip', type=float, default=1.,
                   help='Explicit ResUNet-adapter global gradient-norm guard; 0 is archived unguarded mode')
    return p


def progressive_margin_loss(logits, labels, progress, epsilon=1e-3):
    valid = labels != 255
    if not valid.any():
        return logits.sum()*0, logits.new_empty(0)
    scores = logits.permute(0, 2, 3, 1)[valid]
    target = labels[valid]
    with torch.no_grad():
        competitor = scores.detach().clone()
        competitor.scatter_(1, target[:, None], -torch.inf)
        j = competitor.argmax(1)
        py = scores.detach().softmax(1).gather(1, target[:, None]).squeeze(1)
        pc = scores.detach().softmax(1).gather(1, j[:, None]).squeeze(1)
        kappa = (py+pc-(py-pc).square()).clamp_min(1e-8)
        margin = scores.detach().gather(1, target[:, None]).squeeze(1)-competitor.amax(1)
        index = margin.abs()/kappa
        # Empirical CDF, including equal-score ties: F_R(r)=P(R<=r).
        ordered = index.sort().values
        rank = torch.searchsorted(ordered, index, right=True).to(logits.dtype)/len(index)
        rho = min(max(float(progress), epsilon), 1-epsilon)
        exponent = math.log((1-rho)/rho)
        # log-domain normalization prevents u**exponent overflow near endpoints.
        weight = (exponent*rank.log()).softmax(0)*len(rank)
    return (F.cross_entropy(scores, target, reduction='none')*weight).mean(), weight


@contextmanager
def fixed_batch_norm(module):
    """Extra regularizer forwards must not update BN statistics again."""
    norms = [(m, m.training) for m in (module.modules() if hasattr(module, 'modules') else ())
             if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for m, _ in norms:
            m.eval()
        yield
    finally:
        for m, mode in norms:
            m.train(mode)


def decoder_input(backbone, x):
    x1 = backbone.conv2(F.relu(backbone.bn1(backbone.conv1(x))))
    x2 = backbone.residual_conv_1(x1)
    x3 = backbone.residual_conv_2(x2)
    bridge = backbone.bridge(x3)
    x6 = backbone.up_residual_conv1(torch.cat([backbone.upsample_1(bridge), x3], 1))
    x8 = backbone.up_residual_conv2(torch.cat([backbone.upsample_2(x6), x2], 1))
    return torch.cat([backbone.upsample_3(x8), x1], 1)


def smooth_knowledge_losses(feature, teacher_response, decoder):
    """Supplement Algorithms 1/2, with explicit dense-pixel mean reductions.

    The perturbation direction is stop-gradient (the paper does not specify its
    differentiation); DSS retains its first derivative for double backprop.
    """
    with fixed_batch_norm(decoder):
        response = decoder(feature)
        discrepancy = (response-teacher_response.detach()).square().sum()
        gradient = torch.autograd.grad(discrepancy, response, create_graph=True, retain_graph=True)[0]
        dss = gradient.square().sum(1).mean()
        direction = torch.autograd.grad(discrepancy, feature, retain_graph=True)[0].detach()
        perturbation = F.normalize(direction, dim=1, eps=1e-12)
        perturbed = decoder(feature+perturbation)
        stabilization = (response-perturbed).square().sum(1).mean()
    return dss, stabilization


def competition_counts(logits, labels):
    classes = logits.shape[1]
    valid = labels != 255
    ids = labels[valid]*classes+logits.detach().argmax(1)[valid]
    return torch.bincount(ids, minlength=classes*classes).reshape(classes, classes)


def misclassification_decoupling(class_axes, competition):
    gram = class_axes @ class_axes.t()
    weights = competition.to(gram).clone()
    weights.fill_diagonal_(0)
    return (weights*gram.square()).sum()


class Ldka(ClassIncrementalModel):
    NAME = 'ldka'

    def __init__(self, backbone, loss, args, transform):
        if (args.lr <= 0 or args.ldka_incremental_lr <= 0 or not 0 < args.ldka_progress_epsilon < .5 or not 0 <= args.ldka_pseudo_threshold <= 1 or
                any(getattr(args, name) < 0 for name in ('ldka_dss_weight', 'ldka_perturb_weight', 'ldka_mad_weight'))):
            raise ValueError('Invalid LDKA progress, threshold or loss weights.')
        super().__init__(backbone, loss, args, transform)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=.9, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, args.lr_decay_epoch, args.lr_decay_rate)
        c = self.net.class_num
        self.register_buffer('epoch_confusion', torch.zeros(c, c, dtype=torch.long))
        self.register_buffer('competition', torch.zeros(c, c))
        self.epoch = 0
        self.total_epochs = max(1, args.n_epochs)
        self.progress = 1/(self.total_epochs+1)
        if args.ldka_grad_clip < 0:
            raise ValueError('LDKA gradient-norm guard must be nonnegative.')
        self.guard_stats = {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}

    @property
    def gradient_clip_max_norm(self):
        return self.args.ldka_grad_clip

    def knowledge_state(self):
        return {**super().knowledge_state(), 'epoch': self.epoch,
                'total_epochs': self.total_epochs, 'progress': self.progress,
                'guard_stats': dict(self.guard_stats)}

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.epoch, self.total_epochs, self.progress = state['epoch'], state['total_epochs'], state['progress']
        self.guard_stats = dict(state.get('guard_stats', {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}))

    def begin_task(self, dataset):
        super().begin_task(dataset)
        self.opt.state.clear()
        self.epoch_confusion.zero_()
        self.competition.zero_()
        self.begin_epoch(0, max(1, self.args.n_epochs))
        for group in self.opt.param_groups:
            group['lr'] = self.task_learning_rate

    @property
    def task_learning_rate(self):
        # Published VOC stage rates. The template deliberately does not inherit
        # the benchmark's much larger 0.008 rate for the SKD double backward.
        return self.args.lr if self.current_task == 0 else self.args.ldka_incremental_lr

    def begin_epoch(self, epoch, total_epochs):
        if not 0 <= epoch < total_epochs:
            raise ValueError('LDKA epoch must lie inside the current training schedule.')
        self.epoch, self.total_epochs = epoch, total_epochs
        # Zero-based trainer epoch becomes a progress value in the open interval.
        self.progress = (epoch+1)/(total_epochs+1)

    def optimization_step(self, loss, **parts):
        # The dense classifier is preallocated. Native SGD decay/momentum must
        # not alter future rows that never participated in the objective.
        seen = int(self.net.seen_classes)
        parameters = (self.net.classifier.weight, self.net.classifier.bias)
        future = [p[seen:].detach().clone() for p in parameters]
        value = super().optimization_step(loss, **parts)
        if self.gradient_clip_max_norm > 0:
            norm = self.last_losses['gradient_norm_before_clip']
            self.guard_stats['steps'] += 1
            self.guard_stats['clipped'] += int(norm > self.gradient_clip_max_norm)
            self.guard_stats['peak_preclip_norm'] = max(self.guard_stats['peak_preclip_norm'], norm)
        with torch.no_grad():
            for p, saved in zip(parameters, future):
                p[seen:].copy_(saved)
                buffer = self.opt.state[p].get('momentum_buffer')
                if buffer is not None:
                    buffer[seen:].zero_()
        return value

    def end_epoch(self):
        with torch.no_grad():
            self.competition.copy_(self.epoch_confusion/self.epoch_confusion.sum(1, keepdim=True).clamp_min(1))
            self.epoch_confusion.zero_()

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        seen = int(self.net.seen_classes)
        validate_partial_labels(labels, self.old_count, seen)
        self.opt.zero_grad(set_to_none=True)
        feature = decoder_input(self.net.backbone, inputs)
        decoded = self.net.backbone.up_residual_conv3(feature)
        logits = self.net.classify(decoded)
        pseudo = labels.clone()
        teacher_response = None
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                teacher_response = self.teacher.features(inputs)
                old_logits = self.teacher.classify(teacher_response)
                confidence, prediction = old_logits.softmax(1).max(1)
                pseudo = torch.where((labels == 0) & (confidence >= self.args.ldka_pseudo_threshold), prediction, labels)
        segmentation, weight = progressive_margin_loss(logits, pseudo, self.progress, self.args.ldka_progress_epsilon)
        dss = perturb = logits.sum()*0
        # Ignore255 is absent from PML/MAD. For SKD, use a spatial mask at the
        # response level as well; do not distill explicitly ignored pixels.
        if teacher_response is not None and (labels != 255).any():
            valid = (labels != 255)[:, None].to(feature)
            decoder = self.net.backbone.up_residual_conv3
            def masked_decoder(f):
                return decoder(f)*valid
            # Keep BN control on the module, while masking both responses.
            with fixed_batch_norm(decoder):
                dss, perturb = smooth_knowledge_losses(feature, teacher_response*valid, masked_decoder)
        axes = self.net.classifier.weight[:seen].flatten(1)
        mad = misclassification_decoupling(axes, self.competition[:seen, :seen])
        if not (labels != 255).any():
            mad = mad*0
        total = (segmentation+self.args.ldka_dss_weight*dss+
                 self.args.ldka_perturb_weight*perturb+self.args.ldka_mad_weight*mad)
        value = self.optimization_step(total, supervised=segmentation, dss=dss, perturbation=perturb, mad=mad,
                                       progress=self.progress, participation_mean=weight.mean() if weight.numel() else 0.)
        with torch.no_grad():
            self.epoch_confusion[:seen, :seen].add_(competition_counts(logits, pseudo))
        return value
