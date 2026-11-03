"""Normalized SMG-Learning-inspired Domain-CL benchmark adaptation.

This implementation is non-native: ResUNet, per-slice min/max normalization,
cross-entropy segmentation loss, first-order dual-meta gradients and a fixed
stream-wide allocation of generator updates. Author code was not published.
"""
import json
import os
import random
import numpy as np
import torch
from torch.nn import functional as F

from models.utils.medical import MedicalModel, medical_parser, require_single_head, load_tensors
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import NormalizedEvaluationNetwork, normalize_per_slice
from models.utils.smg_replay import SMGReplayBuffer
from models.utils.smg_generator_step import generator_step
from models.utils.smg_segmentation_step import segmentation_step


def get_parser():
    parser = medical_parser('Normalized SMG domain continual learning adaptation')
    parser.set_defaults(lr=5e-4, lr_decay_rate=1.)
    parser.add_argument('--smg_generator_updates', type=int, default=500000)
    parser.add_argument('--smg_generator_lr', type=float, default=1.5e-4)
    parser.add_argument('--smg_sampling_steps', type=int, default=1000)
    parser.add_argument('--smg_replay_byte_limit', type=int, default=12 * 1024**3)
    parser.add_argument('--smg_progress_interval', type=int, default=5000)
    return parser


class Smg(MedicalModel):
    NAME = 'smg'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args, transform):
        require_single_head(backbone, args)
        if (args.smg_generator_updates < 6 or args.smg_generator_lr <= 0 or
                args.smg_replay_byte_limit <= 0 or args.smg_progress_interval < 1 or
                args.lr_decay_rate != 1.):
            raise ValueError('Invalid SMG budgets, learning rate or scheduler')
        super().__init__(NormalizedEvaluationNetwork(backbone), loss, args, transform)
        self.generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=args.smg_sampling_steps)
        self.opt = torch.optim.Adam(self.net.core.parameters(), lr=args.lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.smg_generator_lr)
        self.generator_updates_total = 0
        self.generator_task_updates = 0
        self.phase = 'idle'
        self.replay_buffer = None

    def knowledge_state(self):
        return {'generator_updates_total': self.generator_updates_total,
                'generator_task_updates': self.generator_task_updates, 'phase': self.phase,
                'generator_budget': self.args.smg_generator_updates,
                'generator_lr': self.args.smg_generator_lr,
                'sampling_steps': self.args.smg_sampling_steps,
                'replay_byte_limit': self.args.smg_replay_byte_limit}

    def restore_knowledge(self, state):
        if (state['generator_budget'] != self.args.smg_generator_updates or
                state['generator_lr'] != self.args.smg_generator_lr or
                state['sampling_steps'] != self.args.smg_sampling_steps or
                state['replay_byte_limit'] != self.args.smg_replay_byte_limit):
            raise ValueError('SMG checkpoint protocol mismatch')
        self.generator_updates_total = state['generator_updates_total']
        self.generator_task_updates = state['generator_task_updates']
        self.phase = state['phase']

    def _generator_budget_for_task(self, task):
        q, remainder = divmod(self.args.smg_generator_updates, 6)
        return q + int(task < remainder)

    def _save_generator_progress(self):
        os.makedirs(self.args.output_folder, exist_ok=True)
        path = os.path.join(self.args.output_folder, 'generator_progress.pth')
        numpy_state = np.random.get_state()
        state = {'format': 'smg-progress-evidence-v1', 'model': self.state_dict(),
                 'generator_optimizer': self.generator_optimizer.state_dict(),
                 'segmenter_optimizer': self.opt.state_dict(),
                 'replay': self.replay_buffer.state_dict() if self.replay_buffer is not None else None,
                 'python_rng': random.getstate(),
                 'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(next(self.net.parameters()).device)}
        torch.save(state, path + '.partial')
        os.replace(path + '.partial', path)

    def begin_task(self, dataset):
        super().begin_task(dataset)
        if dataset.N_TASKS != 6 or not 0 <= self.current_task < 6:
            raise ValueError('SMG protocol requires six ordered domains')
        self.generator_task_updates = 0
        self.phase = 'build_replay'
        if self.current_task:
            masks = []
            for _, labels in dataset.train_loader:
                masks.append(labels[:, None].float().cpu())
            if not masks:
                raise ValueError('Current-domain training loader is empty')
            self.replay_buffer = SMGReplayBuffer(self.args.smg_replay_byte_limit)
            self.replay_buffer.build(self.generator, torch.cat(masks), list(range(self.current_task)),
                                     batch_size=self.args.batch_size)
        self.phase = 'generator'
        budget = self._generator_budget_for_task(self.current_task)
        iterator = iter(dataset.train_loader)
        while self.generator_task_updates < budget:
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(dataset.train_loader)
                inputs, labels = next(iterator)
            inputs = normalize_per_slice(inputs.to(self.device, dtype=torch.float32))
            labels = labels.to(self.device, dtype=torch.long)
            replay = (self.replay_buffer.sample(inputs.shape[0], self.device)
                      if self.replay_buffer is not None else None)
            result = generator_step(self.generator, self.generator_optimizer, inputs,
                                    labels[:, None].float(), self.current_task,
                                    tuple(range(self.current_task)), replay)
            self.generator_task_updates += 1
            self.generator_updates_total += 1
            if self.generator_task_updates % 100 == 0:
                print('SMG generator task {} update {}/{} loss {:.6f}'.format(
                    self.current_task + 1, self.generator_task_updates, budget, result['loss']), flush=True)
            if self.generator_task_updates % self.args.smg_progress_interval == 0:
                self._save_generator_progress()
        self.phase = 'segmenter'

    def observe(self, inputs, labels, not_aug_inputs=None):
        self.require_open_task()
        if self.phase != 'segmenter':
            raise RuntimeError('Generator preparation is not complete')
        normalized = normalize_per_slice(inputs)
        if self.current_task:
            replay = self.replay_buffer.sample(inputs.shape[0], self.device)
            result = segmentation_step(self.net.core, self.opt, F.cross_entropy, self.generator,
                                       normalized, labels, self.current_task, replay_batch=replay)
            return result['loss']
        self.opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.net.core(normalized), labels)
        return self.optimization_step(loss, supervised=loss)

    def end_task(self, dataset):
        self.require_open_task()
        self.replay_buffer = None
        self.phase = 'idle'
        self.task_open = False
        self.current_task += 1

    def save_network(self, network, savepath, task_label, iter_label):
        if network is not self.net or self.task_open or self.phase != 'idle':
            raise ValueError('SMG formal checkpoints require a closed task')
        os.makedirs(savepath, exist_ok=True)
        path = os.path.join(savepath, 'T{}_{}.pth'.format(task_label, iter_label))
        torch.save({'format': 'smg-cl-v1', 'method': self.NAME, 'config': dict(vars(self.args)),
                    'model': self.state_dict(), 'optimizer': self.opt.state_dict(),
                    'generator_optimizer': self.generator_optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

    def load_network(self, load_path, network, strict=True):
        if network is not self.net:
            raise ValueError('SMG checkpoint must load into its own network')
        state = load_tensors(load_path)
        if state.get('format') != 'smg-cl-v1' or state.get('method') != self.NAME:
            raise ValueError('Expected a complete SMG checkpoint')
        self.to(next(self.net.parameters()).device)
        self.load_state_dict(state['model'], strict=strict)
        self.opt.load_state_dict(state['optimizer'])
        self.generator_optimizer.load_state_dict(state['generator_optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        for optimizer in (self.opt, self.generator_optimizer):
            for param, values in optimizer.state.items():
                for key, value in values.items():
                    if torch.is_tensor(value):
                        values[key] = value.to(param.device)
