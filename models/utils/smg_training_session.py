"""In-memory task lifecycle for paper-derived SMG components.

The caller supplies a frozen train-only loader, loss and per-task budgets, and
must enforce resource deadlines, sampler handling and persistent checkpoints.
This is not yet a benchmark runner or an author-native reproduction.
"""
import torch
from models.utils.smg_generator_step import generator_step
from models.utils.smg_segmentation_step import segmentation_step
from models.utils.smg_checkpoint import save_state, load_state
from models.utils.smg_intensity import normalize_per_slice
from models.utils.smg_replay import SMGReplayBuffer


class SMGTrainingSession:
    def __init__(self, segmenter, generator, segmenter_optimizer, generator_optimizer,
                 loss_fn, generator_updates_per_task, segmenter_updates_per_task,
                 replay_byte_limit=4 * 1024**3):
        if (not generator_updates_per_task or
                len(generator_updates_per_task) != len(segmenter_updates_per_task) or
                any(type(n) is not int or n < 1 for n in
                    list(generator_updates_per_task) + list(segmenter_updates_per_task))):
            raise ValueError('Provide positive, frozen update budgets for each task')
        self.segmenter, self.generator = segmenter, generator
        self.segmenter_optimizer, self.generator_optimizer = segmenter_optimizer, generator_optimizer
        self.loss_fn = loss_fn
        self.generator_budgets = tuple(generator_updates_per_task)
        self.segmenter_budgets = tuple(segmenter_updates_per_task)
        if type(replay_byte_limit) is not int or replay_byte_limit < 1:
            raise ValueError('A positive replay byte limit is required')
        self.replay_byte_limit = replay_byte_limit
        self.task, self.phase = 0, 'generator'
        self.generator_updates = self.segmenter_updates = 0
        self.replay_buffer = None

    def set_replay_buffer(self, buffer):
        if self.task < 1 or self.phase != 'generator' or self.generator_updates:
            raise ValueError('Install replay before incremental generator updates')
        if buffer.images is None or tuple(sorted(set(buffer.sites.tolist()))) != tuple(range(self.task)):
            raise ValueError('Replay must cover exactly the prior sites')
        if buffer.max_bytes > self.replay_byte_limit:
            raise ValueError('Replay reservation exceeds frozen session budget')
        self.replay_buffer = buffer

    def progress_state(self):
        return {'task': self.task, 'phase': self.phase,
                'generator_updates': self.generator_updates,
                'segmenter_updates': self.segmenter_updates,
                'generator_budgets': list(self.generator_budgets),
                'segmenter_budgets': list(self.segmenter_budgets),
                'replay_byte_limit': self.replay_byte_limit}

    def restore_progress(self, state):
        if (state['generator_budgets'] != list(self.generator_budgets) or
                state['segmenter_budgets'] != list(self.segmenter_budgets) or
                state['replay_byte_limit'] != self.replay_byte_limit):
            raise ValueError('Frozen update budgets differ')
        task, phase = state['task'], state['phase']
        gen, seg = state['generator_updates'], state['segmenter_updates']
        if (type(task) is not int or type(gen) is not int or type(seg) is not int or
                task < 0 or task > len(self.generator_budgets) or gen < 0 or seg < 0):
            raise ValueError('Invalid progress counters')
        if task == len(self.generator_budgets):
            if phase != 'complete' or gen != 0 or seg != 0:
                raise ValueError('Invalid complete state')
        elif not ((phase == 'generator' and gen < self.generator_budgets[task] and seg == 0) or
                  (phase == 'segmenter' and gen == self.generator_budgets[task] and
                   seg < self.segmenter_budgets[task])):
            raise ValueError('Invalid phase or update counters')
        self.task, self.phase = task, phase
        self.generator_updates, self.segmenter_updates = gen, seg

    def save_checkpoint(self, path, epoch, protocol, *, epoch_boundary):
        if epoch_boundary is not True:
            raise ValueError('Only explicit epoch-boundary checkpoints are supported')
        meta = {'protocol': protocol, 'progress': self.progress_state(),
                'epoch_boundary': True}
        replay = self.replay_buffer.state_dict() if self.replay_buffer is not None else None
        save_state(path, self.segmenter, self.generator, self.segmenter_optimizer,
                   self.generator_optimizer, self.task, epoch, meta, extra={'replay': replay})

    def load_checkpoint(self, path, protocol):
        raw = torch.load(path, map_location='cpu', weights_only=True)
        metadata = raw.get('metadata', {})
        if metadata.get('protocol') != protocol or metadata.get('epoch_boundary') is not True:
            raise ValueError('Checkpoint protocol or boundary mismatch')
        replay_state = (raw.get('extra') or {}).get('replay')
        buffer = None
        if replay_state is not None:
            buffer = SMGReplayBuffer(self.replay_byte_limit)
            buffer.load_state_dict(replay_state)
            task = metadata['progress']['task']
            if tuple(sorted(set(buffer.sites.tolist()))) != tuple(range(task)):
                raise ValueError('Replay checkpoint has invalid site coverage')
        elif metadata['progress']['task'] and metadata['progress']['phase'] != 'complete':
            progress = metadata['progress']
            if progress['generator_updates'] or progress['segmenter_updates']:
                raise ValueError('Incremental checkpoint is missing replay')
        old_progress = self.progress_state()
        self.restore_progress(metadata['progress'])
        try:
            task, epoch = load_state(path, self.segmenter, self.generator,
                                     self.segmenter_optimizer, self.generator_optimizer,
                                     metadata)
            if task != self.task:
                raise ValueError('Checkpoint task and progress disagree')
        except Exception:
            self.restore_progress(old_progress)
            raise
        self.replay_buffer = buffer
        return epoch

    def step_generator(self, images, labels):
        if self.phase != 'generator' or labels.ndim != 3 or labels.dtype != torch.long:
            raise ValueError('Expected generator phase and integer segmentation labels')
        if ((labels != 0) & (labels != 1)).any():
            raise ValueError('Expected binary labels')
        normalized = normalize_per_slice(images)
        if self.task and self.replay_buffer is None:
            raise RuntimeError('Incremental generation requires a prepared replay buffer')
        replay = self.replay_buffer.sample(images.shape[0], images.device) if self.task else None
        out = generator_step(self.generator, self.generator_optimizer, normalized,
                             labels[:, None].float(), self.task, tuple(range(self.task)), replay)
        self.generator_updates += 1
        if self.generator_updates == self.generator_budgets[self.task]:
            self.phase = 'segmenter'
        return out

    def step_segmenter(self, images, labels):
        if self.phase != 'segmenter':
            raise ValueError('Generator phase must finish first')
        images = normalize_per_slice(images)
        if self.task:
            if self.replay_buffer is None:
                raise RuntimeError('Incremental segmentation requires a prepared replay buffer')
            replay = self.replay_buffer.sample(images.shape[0], images.device)
            out = segmentation_step(self.segmenter, self.segmenter_optimizer, self.loss_fn,
                                    self.generator, images, labels, self.task, replay_batch=replay)
        else:
            self.segmenter.train()
            self.segmenter_optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.segmenter(images), labels)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite base segmentation loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in self.segmenter.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite base segmentation gradient')
            self.segmenter_optimizer.step()
            out = {'loss': float(loss.detach()), 'current_samples': images.shape[0],
                   'synthetic_samples': 0}
        self.segmenter_updates += 1
        if self.segmenter_updates == self.segmenter_budgets[self.task]:
            self.task += 1
            self.generator_updates = self.segmenter_updates = 0
            self.replay_buffer = None
            self.phase = 'complete' if self.task == len(self.generator_budgets) else 'generator'
        return out
