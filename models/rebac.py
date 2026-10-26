"""ReBac-C / ResUNet-2D, IJCV 2026. See README_natural_ciss.md."""
import torch
from torch.nn import functional as F
from models.utils.class_incremental import ciss_parser, validate_partial_labels
from models.utils.medical import MedicalModel, require_single_head
from models.utils.rebac_head import RebacResUnet


def get_parser():
    p = ciss_parser('ReBac-C residual background correction')
    p.set_defaults(lr=.01)
    p.add_argument('--rebac_incremental_lr', type=float, default=.001)
    p.add_argument('--rebac_old_lr', type=float, default=.0001)
    p.add_argument('--rebac_pseudo_threshold', type=float, default=.7)
    p.add_argument('--rebac_background_weight', type=float, default=5.)
    p.add_argument('--rebac_feature_weight', type=float, default=4.)
    p.add_argument('--rebac_group_weight', type=float, default=1.)
    p.add_argument('--rebac_adaptation_weight', type=float, default=1.)
    return p


def masked_mean(value, mask):
    return value[mask].mean() if mask.any() else value.sum()*0


def binary_segmentation_loss(logits, labels):
    valid = labels != 255
    target = F.one_hot(labels.clamp(0, logits.shape[1]-1), logits.shape[1]).permute(0, 3, 1, 2).to(logits)
    return masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(1), valid)


def rebac_losses(logits, state, previous, labels, old_count, threshold=.7):
    """Literal author reductions, with empty-region/ignore255 safeguards."""
    old_logits, old_state = previous
    with torch.no_grad():
        confidence, prediction = old_logits.sigmoid().max(1)
        old_region = (labels == 0) & (prediction > 0) & (confidence >= threshold)
        pseudo = torch.where(old_region, prediction, labels)
    # Supervision must not train previous foreground heads or the background
    # at a confidently pseudo-labelled old pixel.
    background = torch.where(old_region[:, None], logits[:, :1].detach(), logits[:, :1])
    supervised_logits = torch.cat([background, logits[:, 1:old_count].detach(), logits[:, old_count:]], 1)
    supervised = binary_segmentation_loss(supervised_logits, pseudo)
    positive = (labels > 0) & (labels != 255)
    negative = labels == 0
    residual = state['groups'][-1][:, 0]
    adaptation = masked_mean(F.softplus(residual), positive)
    prob = residual.sigmoid()
    background_loss = masked_mean(((prob-1).square()-prob.square()).clamp_min(0), negative)
    valid = labels != 255
    # Source MSE is averaged over the full feature grid after background masking.
    features = sum(((a-b.detach()).square()*negative[:, None]).mean()
                   for a, b in zip(state['responses'], old_state['responses']))
    groups = sum(masked_mean(F.binary_cross_entropy_with_logits(a, b.detach().sigmoid(), reduction='none').sum(1), valid)
                 for a, b in zip(state['groups'], old_state['groups']))
    return supervised, background_loss, features, groups, adaptation


class Rebac(MedicalModel):
    NAME = 'rebac'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        if min(args.lr, args.rebac_incremental_lr, args.rebac_old_lr) <= 0 or not 0 <= args.rebac_pseudo_threshold <= 1 or any(getattr(args, n) < 0 for n in
                ('rebac_background_weight', 'rebac_feature_weight', 'rebac_group_weight', 'rebac_adaptation_weight')):
            raise ValueError('Invalid ReBac threshold or loss weight.')
        super().__init__(RebacResUnet(backbone), loss, args, transform)
        self.task_classes = []
        self.old_count = 1

    def rebuild_optimizer(self):
        # All branches, including residual adapters, are registered. Upstream's
        # incremental optimizer accidentally omits classifier.adps parameters.
        groups = [{'params': list(self.net.backbone.parameters()), 'branch': -1}]
        for i in range(len(self.net.counts)):
            parameters = list(self.net.branches[i].parameters()) + list(self.net.heads[i].parameters())
            if i:
                parameters += list(self.net.adapters[i-1].parameters())
            groups.append({'params': parameters, 'branch': i})
        self.opt = torch.optim.SGD(groups, lr=self.args.lr, momentum=.9, nesterov=True, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.lr_decay_epoch, self.args.lr_decay_rate)

    @property
    def task_learning_rate(self):
        return self.args.lr if self.current_task == 0 else self.args.rebac_incremental_lr

    def knowledge_state(self):
        return {'task_classes': self.task_classes, 'old_count': self.old_count}

    def restore_knowledge(self, state):
        self.task_classes = state['task_classes']
        self.old_count = state['old_count']

    def load_state_dict(self, state_dict, strict=True):
        counts = state_dict['_extra_state']['knowledge']['task_classes']
        if counts and self.net.configure(counts):
            self.rebuild_optimizer()
        return super().load_state_dict(state_dict, strict=strict)

    def begin_task(self, dataset):
        self.check_task(dataset)
        counts = dataset.N_CLASSES_PER_TASK
        if not isinstance(counts, (tuple, list)):
            raise ValueError('ReBac requires foreground task counts.')
        if self.net.configure(counts):
            self.rebuild_optimizer()
        if not 0 <= self.current_task < len(counts):
            raise ValueError('ReBac task outside the declared stream.')
        if int(self.net.seen_tasks) != self.current_task:
            raise ValueError('ReBac task position and branch state disagree.')
        self.old_count = 1 + sum(counts[:self.current_task])
        if self.current_task and (self.teacher is None or int(self.teacher.seen_classes) != self.old_count):
            raise ValueError('ReBac needs a complete previous-task teacher.')
        self.task_classes = list(counts)
        self.opt.state.clear()  # author launches each task with a fresh SGD optimizer
        self.net.initialize_branch(self.current_task)
        self.net.seen_tasks.fill_(self.current_task + 1)
        self.net.seen_classes.fill_(self.old_count + counts[self.current_task])
        self.net.freeze_bn.fill_(self.current_task > 0)
        for group in self.opt.param_groups:
            old = self.current_task > 0 and group['branch'] < self.current_task
            group['lr_scale'] = self.args.rebac_old_lr/self.args.rebac_incremental_lr if old else 1.
            group['lr'] = self.task_learning_rate*group['lr_scale']
        super().begin_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        validate_partial_labels(labels, self.old_count, int(self.net.seen_classes))
        self.opt.zero_grad(set_to_none=True)
        logits, state = self.net.components(inputs)
        parts = dict(background=logits.sum()*0, feature=logits.sum()*0, group=logits.sum()*0, adaptation=logits.sum()*0)
        if self.teacher is None:
            supervised = binary_segmentation_loss(logits, labels)
        else:
            self.teacher.eval()
            with torch.no_grad():
                previous = self.teacher.components(inputs)
            supervised, bg, feat, group, adaptation = rebac_losses(
                logits, state, previous, labels, self.old_count, self.args.rebac_pseudo_threshold)
            parts.update(background=bg, feature=feat, group=group, adaptation=adaptation)
        total = supervised + sum(getattr(self.args, 'rebac_'+key+'_weight')*value for key, value in parts.items())
        return self.optimization_step(total, supervised=supervised, **parts)
