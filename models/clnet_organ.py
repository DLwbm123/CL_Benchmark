"""Continual Segment-inspired 2D, task-aware Organ-CL benchmark adaptation.

This version learns the first encoder on the first benchmark task, not from the
author's 3D pretrained encoder. Each task has a full separate decoder. The
author's validation-guided pruning gate is applied to training data only, to
respect the benchmark's held-out restrictions. These choices are non-native.
"""
import torch
from models.utils.medical import medical_parser
from models.utils.organ import OrganModel, organ_parser, validate_organ_backbone, binary_cross_entropy
from models.utils.clnet_resunet import TaskRoutedNetwork
from models.utils.clnet_training_session import ContinualSegmentSession


def get_parser():
    parser = organ_parser(medical_parser('Continual Segment 2D Organ-CL adaptation'))
    parser.add_argument('--clnet_ema_decay', type=float, default=.99)
    parser.add_argument('--clnet_max_drop', type=float, default=.02)
    return parser


class ClnetOrgan(OrganModel):
    NAME = 'clnet_organ'
    REQUIRES_TEACHER = False
    SPARSITY = (.8, .82, .84, .86, .88, .9, .92, .94, .96, .98,
                .99, .992, .994, .996, .998)

    def __init__(self, backbone, loss, args, transform):
        validate_organ_backbone(backbone, args)
        super().__init__(TaskRoutedNetwork(backbone), loss, args, transform)
        self.session = ContinualSegmentSession(self.net, self.opt, binary_cross_entropy,
                                               args.clnet_ema_decay)
        self._train_loader = None
        self.pruning_history = []

    def begin_task(self, dataset):
        super().begin_task(dataset)
        self.session.begin_task(self.current_task)
        self._train_loader = dataset.train_loader

    @torch.no_grad()
    def _training_score(self):
        self.net.eval()
        intersection = predicted = actual = 0
        for images, labels in self._train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.net(images, self.current_task)
            valid = labels != 255
            fg = (logits.argmax(1) == 1) & valid
            target = (labels == 1) & valid
            intersection += int((fg & target).sum())
            predicted += int(fg.sum())
            actual += int(target.sum())
        self.net.train()
        return (2 * intersection + 1) / (predicted + actual + 1)

    def begin_epoch(self, epoch, n_epochs):
        if n_epochs != 150:
            raise ValueError('This protocol requires the frozen 150-epoch schedule')
        # Five warm-up epochs; fifteen fixed attempts through epoch 99; fifty
        # recovery epochs. Acceptance is measured exclusively on train data.
        schedule = {5 + (i * 95 // len(self.SPARSITY)): target
                    for i, target in enumerate(self.SPARSITY)}
        if epoch in schedule:
            baseline = self._training_score()
            result = self.session.try_prune(schedule[epoch], self._training_score,
                                            baseline, max_drop=self.args.clnet_max_drop)
            result.update(epoch=epoch, task=self.current_task)
            self.pruning_history.append(result)
        if epoch == 100:
            self.session.start_recovery()

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        return self.session.step(inputs, labels)

    def knowledge_state(self):
        return {**super().knowledge_state(), 'pruning_history': self.pruning_history}

    def restore_knowledge(self, state):
        super().restore_knowledge(state)
        self.pruning_history = state['pruning_history']

    def end_task(self, dataset):
        self.require_open_task()
        self.session.end_task()
        self.task_open = False
        self.current_task += 1
        self._train_loader = None
