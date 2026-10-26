"""PCDD, AAAI 2026: paper-derived 2D adaptation of PGCD and DAPD.

The author repository has no implementation at the checked revision. Eq. 3
and 10 have typographical issues; CRCD has an unspecified new-class teacher
space. Explicit interpretations are recorded in README_natural_ciss.md.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.utils.class_incremental import ClassIncrementalModel, SeenClassResUnet, ciss_parser


class PcddResUnet(SeenClassResUnet):
    """Keep student/teacher BN statistics aligned during prototype distillation.

    The source SwinUNETR does not have ResUNet's train/eval BatchNorm mismatch.
    This explicit adapter freezes running statistics after the base task, while
    preserving trainable affine parameters and every original feature channel.
    """
    def __init__(self, backbone):
        super().__init__(backbone)
        self.register_buffer('freeze_statistics', torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        if bool(self.freeze_statistics):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


def get_parser():
    p = ciss_parser('Prototype-guided calibration and dual-aligned distillation')
    p.set_defaults(lr=.01, lr_decay_rate=1.)
    p.add_argument('--pcdd_entropy_threshold', type=float, default=.7)
    p.add_argument('--pcdd_orcd_weight', type=float, default=1.)
    p.add_argument('--pcdd_crcd_weight', type=float, default=.5)
    p.add_argument('--pcdd_local_weight', type=float, default=.5)
    p.add_argument('--pcdd_global_weight', type=float, default=.1)
    p.add_argument('--pcdd_grad_clip', type=float, default=1.,
                   help='ResUNet incremental-only gradient norm guard; 0 disables')
    return p


def local_prototypes(features, labels, classes):
    """Differentiable per-class means; absent classes carry a false validity bit."""
    flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    target = labels.reshape(-1)
    valid = (target >= 0) & (target < classes) & (target != 255)
    counts = torch.bincount(target[valid], minlength=classes)
    sums = features.new_zeros(classes, features.shape[1]).index_add(0, target[valid], flat[valid])
    return sums/counts.clamp_min(1)[:, None], counts


def merge_student_probabilities(logits, old_count):
    logp = logits.log_softmax(1)
    bg = torch.cat([logp[:, :1], logp[:, old_count:]], 1).logsumexp(1, keepdim=True)
    return torch.cat([bg, logp[:, 1:old_count]], 1)


def prototype_affinity(features, prototypes, counts):
    if not (counts > 0).any():
        return features.new_zeros(features.shape[0], len(counts), *features.shape[-2:])
    scores = torch.einsum('bkhw,ck->bchw', F.normalize(features, dim=1), F.normalize(prototypes, dim=1))
    return scores.masked_fill((counts == 0)[None, :, None, None], -1e4).softmax(1)


def calibrated_distillation(student, teacher, student_features, teacher_features,
                           pseudo, old_prototypes, old_counts, current_prototypes, current_counts, old_count):
    """Forward KL (student || teacher), as the paper writes; no KL reversal.

    CRCD's unspecified future teacher channels are resolved in the common MiB
    old-label space: pool current background+new probability and affinity mass.
    Classwise affinity-weighted KL summands need not be individually positive.
    """
    logp = merge_student_probabilities(student, old_count)
    logq = teacher.detach().log_softmax(1)
    terms = logp.exp()*(logp-logq)
    valid = pseudo != 255
    old_region = (pseudo > 0) & (pseudo < old_count) & valid
    new_region = ((pseudo == 0) | (pseudo >= old_count)) & valid
    a_old = prototype_affinity(teacher_features.detach(), old_prototypes[1:old_count], old_counts[1:old_count])
    orcd_map = (a_old*terms[:, 1:]).sum(1)
    a_new = prototype_affinity(student_features, current_prototypes, current_counts)
    a_bg = torch.cat([a_new[:, :1], a_new[:, old_count:]], 1).sum(1, keepdim=True)
    a_new = torch.cat([a_bg, a_new[:, 1:old_count]], 1)
    crcd_map = (a_new*terms).sum(1)
    zero = student.sum()*0
    return (orcd_map[old_region].mean() if old_region.any() else zero,
            crcd_map[new_region].mean() if new_region.any() else zero)


def dual_prototype_alignment(student_features, teacher_features, pseudo, old_count,
                             seen_count, global_prototypes, global_counts):
    local, counts = local_prototypes(student_features, pseudo, seen_count)
    merged = torch.where((pseudo >= old_count) & (pseudo != 255), 0, pseudo)
    previous, previous_counts = local_prototypes(teacher_features.detach(), merged, old_count)
    # Eq.10's sum index is interpreted as p_n, not repeated p_c. This is a
    # sum of present class means, not a voxel-weighted pooled background mean.
    q = torch.cat([torch.cat([local[:1], local[old_count:]], 0).sum(0, keepdim=True), local[1:old_count]], 0)
    q_counts = torch.cat([(counts[:1].sum()+counts[old_count:].sum())[None], counts[1:old_count]])
    usable_local = (q_counts > 0) & (previous_counts > 0)
    usable_global = (q_counts > 0) & (global_counts[:old_count] > 0)
    local_loss = ((q-previous).square().sum(1)*usable_local).sum()/old_count
    global_loss = ((q-global_prototypes[:old_count].detach()).square().sum(1)*usable_global).sum()/old_count
    return local_loss, global_loss


class Pcdd(ClassIncrementalModel):
    NAME = 'pcdd'
    NETWORK_TYPE = PcddResUnet

    def __init__(self, backbone, loss, args, transform):
        if args.pcdd_entropy_threshold < 0 or any(getattr(args, name) < 0 for name in
                ('pcdd_orcd_weight', 'pcdd_crcd_weight', 'pcdd_local_weight', 'pcdd_global_weight', 'pcdd_grad_clip')):
            raise ValueError('PCDD thresholds and loss weights must be nonnegative.')
        super().__init__(backbone, loss, args, transform)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, args.lr_decay_epoch, args.lr_decay_rate)
        shape = (self.net.class_num, self.net.classifier.in_channels)
        for name in ('prototype_sum', 'previous_prototypes'):
            self.register_buffer(name, torch.zeros(shape))
        self.register_buffer('prototype_count', torch.zeros(shape[0], dtype=torch.long))
        self.register_buffer('previous_count', torch.zeros(shape[0], dtype=torch.long))
        self.register_buffer('teacher_bg_sum', torch.zeros(shape[1]))
        self.register_buffer('teacher_bg_count', torch.tensor(0, dtype=torch.long))
        self.guard_stats = {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}

    @property
    def gradient_clip_max_norm(self):
        return self.args.pcdd_grad_clip if self.current_task else 0.

    def knowledge_state(self):
        return dict(super().knowledge_state(), guard_stats=dict(self.guard_stats))

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.guard_stats = dict(state.get('guard_stats',
            {'steps': 0, 'clipped': 0, 'peak_preclip_norm': 0.}))

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
        self.opt.state.clear()
        self.net.freeze_statistics.fill_(self.current_task > 0)
        self.net.train()
        self.previous_prototypes.copy_(self.prototype_sum/self.prototype_count.clamp_min(1)[:, None])
        self.previous_count.copy_(self.prototype_count)
        # Background is task-specific. Retain old foreground prototypes and
        # recompute teacher/current backgrounds from the current task only.
        self.prototype_sum[0].zero_()
        self.prototype_count[0].zero_()
        self.teacher_bg_sum.zero_()
        self.teacher_bg_count.zero_()

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        features = self.net.features(inputs)
        logits = self.net.classify(features)
        supervised = self.supervised_loss(logits, labels)
        seen = int(self.net.seen_classes)
        pseudo = labels.clone()
        old_features = old_logits = None
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                old_features = self.teacher.features(inputs)
                old_logits = self.teacher.classify(old_features)
                prob = old_logits.softmax(1)
                entropy = -(prob*prob.clamp_min(1e-12).log()).sum(1)
                pred = prob.argmax(1)
                pseudo = torch.where((labels == 0) & (pred > 0) &
                                     (entropy < self.args.pcdd_entropy_threshold), pred, labels)
        # Candidate banks include this batch; commit only after a successful
        # finite optimization step. Old foreground bank entries stay frozen.
        with torch.no_grad():
            local, counts = local_prototypes(features.detach(), pseudo, seen)
            update = (torch.arange(seen, device=labels.device) >= self.old_count)
            update[0] = True
            sums = self.prototype_sum[:seen] + local*counts[:, None]*update[:, None]
            sizes = self.prototype_count[:seen] + counts*update
            current_bank = sums/sizes.clamp_min(1)[:, None]
        orcd = crcd = ll = lg = logits.sum()*0
        next_bg_sum, next_bg_count = self.teacher_bg_sum, self.teacher_bg_count
        if self.teacher is not None:
            # Old-model background includes current classes (MiB pooling).
            old_labels = torch.where((pseudo >= self.old_count) & (pseudo != 255), 0, pseudo)
            old_local, old_sizes = local_prototypes(old_features, old_labels, self.old_count)
            next_bg_sum = self.teacher_bg_sum + old_local[0].detach()*old_sizes[0]
            next_bg_count = self.teacher_bg_count + old_sizes[0]
            old_bank = self.previous_prototypes.clone()
            old_bank[0] = next_bg_sum/next_bg_count.clamp_min(1)
            old_bank_counts = self.previous_count.clone()
            old_bank_counts[0] = next_bg_count
            orcd, crcd = calibrated_distillation(logits, old_logits, features, old_features, pseudo,
                                                old_bank, old_bank_counts, current_bank, sizes, self.old_count)
            ll, lg = dual_prototype_alignment(features, old_features, pseudo, self.old_count,
                                              seen, old_bank, old_bank_counts)
        total = (supervised + self.args.pcdd_orcd_weight*orcd + self.args.pcdd_crcd_weight*crcd +
                 self.args.pcdd_local_weight*ll + self.args.pcdd_global_weight*lg)
        value = self.optimization_step(total, supervised=supervised, orcd=orcd, crcd=crcd,
                                       local_alignment=ll, global_alignment=lg)
        with torch.no_grad():
            self.prototype_sum[:seen].copy_(sums)
            self.prototype_count[:seen].copy_(sizes)
            self.teacher_bg_sum.copy_(next_bg_sum)
            self.teacher_bg_count.copy_(next_bg_count)
        return value

    def end_task(self, dataset):
        self.require_open_task()
        seen = int(self.net.seen_classes)
        if (self.prototype_count[self.old_count:seen] == 0).any():
            raise ValueError('PCDD cannot finalize a task with missing new-class prototypes.')
        super().end_task(dataset)
