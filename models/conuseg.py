"""CoNuSeg-inspired Class-CL ResUNet adaptation, not author-native code.

Uses the published prototype relation/distillation, hard prototype contrast,
and train-only teacher pseudo-label thresholds. Absent classes are masked.
The backbone and scheduler follow the benchmark, not the original DenseNet.
"""
import torch
from torch.nn import functional as F
from models.utils.class_incremental import ClassIncrementalModel, ciss_parser, unbiased_cross_entropy
from models.utils.conuseg_prototypes import batch_prototypes, relation_distillation, prototype_contrast


def get_parser():
    parser = ciss_parser('CoNuSeg prototype relation and contrast adaptation')
    parser.set_defaults(lr=1e-4)
    parser.add_argument('--conuseg_prd', type=float, default=.1)
    parser.add_argument('--conuseg_pcon', type=float, default=.1)
    parser.add_argument('--conuseg_threshold', type=float, default=.75)
    return parser


class Conuseg(ClassIncrementalModel):
    NAME = 'conuseg'

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        n, d = self.net.class_num, self.net.classifier.in_channels
        self.register_buffer('prototypes', torch.zeros(n, d))
        self.register_buffer('prototype_counts', torch.zeros(n, dtype=torch.long))
        self.register_buffer('thresholds', torch.zeros(n))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-8)

    @torch.no_grad()
    def begin_task(self, dataset):
        super().begin_task(dataset)
        if self.teacher is None:
            return
        histogram = torch.zeros(self.old_count, 20, dtype=torch.long, device=self.thresholds.device)
        for inputs, labels in dataset.train_loader:
            inputs, labels = inputs.to(self.thresholds.device), labels.to(self.thresholds.device)
            probabilities = self.teacher.logits(inputs).softmax(1)
            confidence, predicted = probabilities.max(1)
            background = labels == 0
            cls = predicted[background]
            bins = (confidence[background] * 20).long().clamp(max=19)
            histogram.view(-1).scatter_add_(0, cls * 20 + bins, torch.ones_like(cls, dtype=torch.long))
        totals = histogram.sum(1)
        cumulative = histogram.cumsum(1)
        middle = ((totals + 1) // 2)[:, None]
        median_bin = (cumulative >= middle).long().argmax(1)
        values = (median_bin.float() + .5) / 20
        values = torch.where(totals > 0, values.clamp(max=self.args.conuseg_threshold), torch.zeros_like(values))
        self.thresholds.zero_()
        self.thresholds[:self.old_count].copy_(values)

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        self.opt.zero_grad(set_to_none=True)
        features = self.net.features(inputs)
        logits = self.net.classify(features)
        supervised = self.supervised_loss(logits, labels)
        pseudo = labels
        prd = contrast = supervised * 0
        if self.teacher is not None:
            with torch.no_grad():
                teacher_features = self.teacher.features(inputs)
                teacher_logits = self.teacher.classify(teacher_features)
                confidence, predicted = teacher_logits.softmax(1).max(1)
                pseudo = torch.where(labels == 0, predicted, labels)
            reliable = (labels == 0) & (confidence > self.thresholds[predicted])
            pseudo_only = torch.where(reliable, predicted, torch.full_like(labels, 255))
            supervised = (unbiased_cross_entropy(logits, torch.where(reliable, torch.full_like(labels, 255), labels), self.old_count)
                          + F.cross_entropy(logits, pseudo_only, ignore_index=255, reduction='none').mean())
            local, present = batch_prototypes(features, pseudo, self.net.class_num)
            old_local, _ = batch_prototypes(teacher_features, pseudo, self.net.class_num)
            old_mask = torch.arange(self.net.class_num, device=inputs.device) < self.old_count
            prd = relation_distillation(local, old_local, (present > 0) & old_mask)
            contrast = prototype_contrast(features, logits, pseudo, self.prototypes,
                                          self.prototype_counts, self.old_count)
        with torch.no_grad():
            local_gt, counts = batch_prototypes(features.detach(), labels, self.net.class_num)
            counts[0] = 0
            total = self.prototype_counts + counts
            self.prototypes.mul_(self.prototype_counts[:, None]).add_(local_gt * counts[:, None]).div_(total.clamp_min(1)[:, None])
            self.prototype_counts.copy_(total)
        return self.optimization_step(supervised + self.args.conuseg_prd * prd + self.args.conuseg_pcon * contrast,
                                      supervised=supervised, prd=prd, contrast=contrast)
