"""Explicit task-IL protocol for the benchmark's binary Organ-CL stream."""
import torch
from torch import nn
from torch.nn import functional as F
from models.utils.medical import MedicalModel


ORGAN_NAMES = ['left_atrium', 'prostate', 'liver', 'brain_tumor']


def organ_parser(parser):
    parser.set_defaults(dataset='seq-taskincre', multihead=True)
    parser.add_argument('--organ_tasks', type=int, default=4)
    parser.add_argument('--organ_task_names', nargs='+', default=ORGAN_NAMES.copy())
    return parser


def checked_task_id(index, tasks):
    if type(index) is not int or not 0 <= index < tasks:
        raise ValueError('Organ-CL requires an explicit valid integer task ID.')
    return index


def validate_organ_backbone(backbone, args):
    if (not args.multihead or not isinstance(backbone.last, nn.ModuleDict) or
            type(args.organ_tasks) is not int or args.organ_tasks < 1 or
            list(backbone.last) != [str(i) for i in range(args.organ_tasks)]):
        raise ValueError('Organ-CL needs one declared binary head per task and --multihead.')
    heads = list(backbone.last.values())
    if any(not isinstance(h, nn.Conv2d) or h.out_channels != 2 or h.kernel_size != (1, 1)
           or h.in_channels != heads[0].in_channels for h in heads):
        raise ValueError('Organ-CL heads must be equal-width 1x1 binary classifiers.')
    names = args.organ_task_names
    if (len(names) != args.organ_tasks or len(set(names)) != len(names) or
            any(not isinstance(n, str) or not n.strip() for n in names)):
        raise ValueError('Provide one distinct organ name per task, in stream order.')
    return heads[0].in_channels


def validate_binary_labels(labels):
    if labels.dtype != torch.long or labels.ndim != 3:
        raise ValueError('Organ-CL expects long local binary labels [B,H,W].')
    if ((labels != 0) & (labels != 1) & (labels != 255)).any():
        raise ValueError('Organ-CL labels must be local 0/1 or ignore255, not global class IDs.')


def binary_cross_entropy(logits, labels):
    valid = labels != 255
    return (F.cross_entropy(logits, labels, ignore_index=255, reduction='none')[valid].mean()
            if valid.any() else logits.sum()*0)


def old_head_distillation(student_logits, teacher_logits, labels, scale=1.):
    """Mean task-wise categorical KL, with independent background per head."""
    valid = labels != 255
    values = []
    for student, teacher in zip(student_logits, teacher_logits):
        logp = student.log_softmax(1)
        q = (teacher.detach()*scale).softmax(1)
        divergence = F.kl_div(logp, q, reduction='none').sum(1)
        values.append(divergence[valid].mean() if valid.any() else student.sum()*0)
    if not values:
        return labels.new_zeros((), dtype=torch.float32)
    return torch.stack(values).mean()


class OrganHeads(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.heads = backbone.last
        self.tasks = len(self.heads)
        backbone.last = nn.Identity()
        self.backbone = backbone
        self.register_buffer('seen_tasks', torch.tensor(0, dtype=torch.long))

    def features(self, x):
        return self.backbone.features(x)

    def classify(self, features, index, detach_head=False):
        head = self.heads[str(checked_task_id(index, self.tasks))]
        w = head.weight.detach() if detach_head else head.weight
        b = head.bias.detach() if detach_head and head.bias is not None else head.bias
        return F.conv2d(features, w, b)

    def forward(self, x, task_index=None):
        return self.classify(self.features(x), task_index)


class OrganModel(MedicalModel):
    COMPATIBILITY = ['task-il']
    REQUIRES_TEACHER = True

    def __init__(self, net, loss, args, transform):
        super().__init__(net, loss, args, transform)
        self.task_names = list(args.organ_task_names)
        if not hasattr(self.net, 'seen_tasks'):
            self.net.register_buffer('seen_tasks', torch.tensor(0, dtype=torch.long))

    def forward(self, x, task_index=None):
        # Do not let ContinualModel silently substitute current_task at test time.
        return self.net(x, task_index=checked_task_id(task_index, self.args.organ_tasks))

    def knowledge_state(self):
        return {'organ_tasks': self.args.organ_tasks, 'task_names': self.task_names}

    def restore_knowledge(self, state):
        if state['organ_tasks'] != self.args.organ_tasks or state['task_names'] != self.task_names:
            raise ValueError('Organ checkpoint task count/order/name mapping mismatch.')

    def load_state_dict(self, state_dict, strict=True):
        # An equally named but different embedding file must not be silently
        # replaced by checkpoint contents under the same experiment identity.
        key = 'net.organ_embeddings'
        if key in state_dict and not torch.equal(state_dict[key].cpu(), self.net.organ_embeddings.detach().cpu()):
            raise ValueError('Organ checkpoint and supplied CLIP embeddings differ.')
        return super().load_state_dict(state_dict, strict=strict)

    def begin_task(self, dataset):
        self.check_task(dataset)
        if (dataset.N_CLASSES_PER_TASK != 2 or dataset.N_TASKS != self.args.organ_tasks or
                (hasattr(dataset, 'TASK_NAMES') and list(dataset.TASK_NAMES) != self.task_names)):
            raise ValueError('Organ-CL stream must match the declared binary task count/order.')
        checked_task_id(self.current_task, self.args.organ_tasks)
        if int(self.net.seen_tasks) != self.current_task:
            raise ValueError('Organ task position and learned-head state disagree.')
        if self.current_task and self.REQUIRES_TEACHER:
            if self.teacher is None or int(self.teacher.seen_tasks) != self.current_task:
                raise ValueError('Organ-CL requires a complete previous-task teacher.')
        self.net.seen_tasks.fill_(self.current_task+1)
        if hasattr(self.net, 'seen_classes'):
            self.net.seen_classes.fill_(self.current_task+1)
        self.opt.state.clear()
        super().begin_task(dataset)
