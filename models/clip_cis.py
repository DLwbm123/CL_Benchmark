"""MICCAI 2023 continual multi-organ segmentation, 2D backbone adaptation.

Retains class-specific text/image-conditioned dynamic heads, independent sigmoid
predictions, and old-class pseudo masks. Original 3D SwinUNETR/pretraining are not
claimed here. Use the adapter name `clip_cis` when reporting these results.
"""
import torch
from torch.nn import functional as F

from models.utils.clip_heads import ClipHeadResUnet
from models.utils.medical import (MedicalModel, load_tensors, medical_parser,
                                 require_single_head)


def get_parser():
    parser = medical_parser('Medical class-incremental CLIP dynamic heads (2D adapter)')
    parser.add_argument('--clip_embeddings', type=str, required=True,
                        help='Local dict: embeddings [C,512], class_ids, class_names')
    parser.add_argument('--clip_pseudo_threshold', type=float, default=0.5)
    return parser


def read_embeddings(path, class_num):
    payload = load_tensors(path)
    if not isinstance(payload, dict) or not all(
            key in payload for key in ('embeddings', 'class_ids', 'class_names')):
        raise ValueError('CLIP embeddings need embeddings, class_ids, class_names metadata.')
    tensor = payload['embeddings']
    if (not torch.is_tensor(tensor) or tensor.shape != (class_num, 512)
            or not torch.isfinite(tensor).all() or (tensor.norm(dim=1) == 0).any()):
        raise ValueError('Expected finite nonzero embeddings [number of foreground classes,512].')
    if payload['class_ids'] != list(range(1, class_num + 1)):
        raise ValueError('Embedding rows must follow the benchmark global foreground label IDs.')
    names = payload['class_names']
    if (len(names) != class_num or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != class_num):
        raise ValueError('Provide one distinct class name per foreground embedding.')
    return tensor.float(), names


def merged_targets(labels, teacher_logits, old_count, seen_count, threshold):
    if labels.dtype != torch.long or labels.ndim != 3:
        raise ValueError('Expected global long labels [B,H,W].')
    if labels.min() < 0 or labels.max() > seen_count:
        raise ValueError('Unseen or invalid foreground label in current-task annotations.')
    ids = torch.arange(1, seen_count + 1, device=labels.device)[None, :, None, None]
    targets = (labels[:, None] == ids).float()
    if old_count:
        if teacher_logits is None or teacher_logits.shape[1] != old_count:
            raise ValueError('Missing old-class teacher logits.')
        pseudo = (teacher_logits.sigmoid() >= threshold).to(targets.dtype)
        # Current true foreground always takes priority over pseudo labels.
        targets[:, :old_count] = torch.where((labels == 0)[:, None], pseudo,
                                             targets[:, :old_count])
    return targets


def independent_segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = logits.sigmoid().flatten(2)
    targets = targets.flatten(2)
    # Match original per-sample/per-organ BinaryDiceLoss, smooth=1.
    dice = 1 - 2 * (probabilities * targets).sum(2) / (
        probabilities.sum(2) + targets.sum(2) + 1)
    return bce + dice.mean()


class ClipCis(MedicalModel):
    NAME = 'clip_cis'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        class_num = backbone.last.out_channels - 1
        embeddings, self.class_names = read_embeddings(args.clip_embeddings, class_num)
        if not 0 < args.clip_pseudo_threshold < 1:
            raise ValueError('clip_pseudo_threshold must be strictly between 0 and 1.')
        super().__init__(ClipHeadResUnet(backbone, embeddings), loss, args, transform)
        self.old_count = 0
        self.task_classes = []

    def knowledge_state(self):
        return {'old_count': self.old_count, 'task_classes': self.task_classes,
                'class_names': self.class_names}

    def restore_knowledge(self, state):
        if state['class_names'] != self.class_names:
            raise ValueError('Checkpoint and embedding class-name mappings differ.')
        self.old_count = state['old_count']
        self.task_classes = state['task_classes']

    def begin_task(self, dataset):
        self.check_task(dataset)
        counts = dataset.N_CLASSES_PER_TASK
        if (not isinstance(counts, (tuple, list)) or not counts or
                any(not isinstance(c, int) or c <= 0 for c in counts) or
                sum(counts) != self.net.class_num):
            raise ValueError('Class-CL requires positive per-task foreground counts '
                             'matching the global output/embedding space.')
        if self.current_task >= len(counts):
            raise ValueError('No additional task declared.')
        self.old_count = sum(counts[:self.current_task])
        if int(self.net.seen_classes) != self.old_count:
            raise ValueError('Task index and learned-class state disagree; restore a full checkpoint.')
        if self.old_count and self.teacher is None:
            raise ValueError('An old-task teacher is required for continual class learning.')
        self.task_classes = list(counts)
        self.net.seen_classes.fill_(self.old_count + counts[self.current_task])
        super().begin_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        seen = int(self.net.seen_classes)
        teacher_logits = None
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                teacher_logits = self.teacher.foreground(inputs, self.old_count)
        targets = merged_targets(labels, teacher_logits, self.old_count,
                                 seen, self.args.clip_pseudo_threshold)
        logits = self.net.foreground(inputs, seen)
        loss = independent_segmentation_loss(logits, targets)
        return self.optimization_step(loss, supervised=loss)
