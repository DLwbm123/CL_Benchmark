"""Shared integration for the added medical CL methods; no dataset I/O/downloads."""
import copy
import json
import os

import torch
from torch import nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel


def medical_parser(description):
    # Import here to avoid the existing args/models discovery cycle.
    from utils.args import ArgumentParser, add_experiment_args, add_management_args
    parser = ArgumentParser(description=description, allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--rma_reference', type=str, default=None,
                        help='Matched independent-task reference JSON; absent means no normalized RMA')
    return parser


def load_tensors(path):
    # Do not deserialize arbitrary pickled Python objects from supplied weights.
    return torch.load(path, map_location='cpu', weights_only=True)


def frozen_copy(net):
    teacher = copy.deepcopy(net).eval()
    teacher.requires_grad_(False)
    return teacher


def resunet_features(net, x):
    """Same ResUNet computation, also exposing the encoder bridge for CLIP heads."""
    x1 = net.conv2(F.relu(net.bn1(net.conv1(x))))
    x2 = net.residual_conv_1(x1)
    x3 = net.residual_conv_2(x2)
    bridge = net.bridge(x3)
    x6 = net.up_residual_conv1(torch.cat([net.upsample_1(bridge), x3], dim=1))
    x8 = net.up_residual_conv2(torch.cat([net.upsample_2(x6), x2], dim=1))
    features = net.up_residual_conv3(torch.cat([net.upsample_3(x8), x1], dim=1))
    return features, bridge


def require_single_head(backbone, args):
    if getattr(args, 'multihead', False) or not isinstance(backbone.last, nn.Conv2d):
        raise ValueError('This medical method requires a shared single output space; '
                         'task-specific multihead routing is not implemented.')


def binary_targets(labels, classes):
    if labels.ndim != 3 or labels.dtype != torch.long:
        raise ValueError('Expected global integer labels [B,H,W] with dtype long.')
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Label IDs lie outside the declared output space.')
    return F.one_hot(labels, classes).permute(0, 3, 1, 2).float()


def softmax_binary_cross_entropy(logits, labels):
    """One-hot BCE of softmax probabilities, evaluated without probability underflow.

    For two classes this is exactly categorical CE. For more classes the
    negative-class log probability is logsumexp of the other log probabilities.
    """
    if logits.shape[1] == 2:
        return F.cross_entropy(logits, labels)
    target = binary_targets(labels, logits.shape[1])
    logp = F.log_softmax(logits, 1)
    log_not_p = torch.stack([torch.cat([logp[:, :c], logp[:, c+1:]], 1).logsumexp(1)
                             for c in range(logits.shape[1])], 1)
    return -(target * logp + (1-target) * log_not_p).mean()


def dice_bce(logits, labels, stable=True):
    """TED upstream flattened Dice (smooth=1) plus multiclass-probability BCE."""
    target = binary_targets(labels, logits.shape[1])
    probs = logits.softmax(1)
    dice = 1 - (2 * (probs * target).sum() + 1) / (probs.sum() + target.sum() + 1)
    bce = (softmax_binary_cross_entropy(logits, labels) if stable else
           F.binary_cross_entropy(probs, target))
    return bce + dice


class MedicalModel(ContinualModel):
    """Lifecycle and complete checkpoints for teacher/importance-based methods.

    The legacy trainer saves only model.net. These overrides keep the same
    filenames/call signatures while including the state needed for continuation.
    """
    def __init__(self, backbone, loss, args, transform):
        if getattr(args, 'pseudo_label', False) or getattr(args, 'mib', False):
            raise ValueError('Added medical methods implement their own objectives; '
                             'do not combine legacy --pseudo_label/--mib wrappers.')
        super().__init__(backbone, loss, args, transform)
        self.teacher = None
        self.task_open = False
        self.last_losses = {}

    def check_task(self, dataset):
        if dataset.SETTING not in self.COMPATIBILITY:
            raise ValueError('{} supports {}, received {}'.format(
                self.NAME, self.COMPATIBILITY, dataset.SETTING))
        if self.task_open:
            raise RuntimeError('begin_task called twice without end_task.')

    def begin_task(self, dataset):
        self.check_task(dataset)
        # The legacy trainer moves only net. Method-level prototype/confusion
        # buffers and an existing teacher must follow its actual device too.
        self.to(next(self.net.parameters()).device)
        self.task_open = True
        self.net.train()
        if self.teacher is not None:
            self.teacher.eval()

    def require_open_task(self):
        if not self.task_open:
            raise RuntimeError('Call begin_task(dataset) before observe/end_task.')

    def end_task(self, dataset):
        self.require_open_task()
        self.teacher = frozen_copy(self.net)
        self.task_open = False
        self.current_task += 1

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def knowledge_state(self):
        return {}

    def evaluation_parameter_count(self):
        # Teachers are training state, not a second deployed segmenter.
        return sum(p.numel() for p in self.net.parameters())

    def rma_reference_results(self, tasks):
        if not self.args.rma_reference:
            return None
        with open(self.args.rma_reference) as source:
            reference = json.load(source)
        if (reference.get('dataset') != self.args.dataset or
                reference.get('seed') != self.args.seed or
                reference.get('order') != self.args.order):
            raise ValueError('Independent-task RMA reference dataset/seed/order mismatch.')
        matrix = reference['results']
        if len(matrix) != tasks or any(len(row) != tasks for row in matrix):
            raise ValueError('RMA reference must be a matching square task matrix.')
        diagonal = torch.tensor([matrix[i][i] for i in range(tasks)], dtype=torch.float64)
        if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
            raise ValueError('RMA references require finite positive diagonal scores.')
        return matrix

    def restore_knowledge(self, state):
        pass

    def get_extra_state(self):
        return {'current_task': self.current_task, 'task_open': self.task_open,
                'knowledge': self.knowledge_state()}

    def set_extra_state(self, state):
        self.current_task = state['current_task']
        self.task_open = state['task_open']
        self.restore_knowledge(state['knowledge'])

    def load_state_dict(self, state_dict, strict=True):
        self.to(next(self.net.parameters()).device)
        if any(k.startswith('teacher.') for k in state_dict):
            self.teacher = frozen_copy(self.net)
        else:
            self.teacher = None
        result = super().load_state_dict(state_dict, strict=strict)
        if self.teacher is not None:
            self.teacher.eval()
        return result

    def save_network(self, network, savepath, task_label, iter_label):
        if network is not self.net:
            raise ValueError('Medical checkpoint must save this model.net.')
        os.makedirs(savepath, exist_ok=True)
        path = os.path.join(savepath, 'T{}_{}.pth'.format(task_label, iter_label))
        torch.save({'format': 'medical-cl-v1', 'method': self.NAME,
                    'config': dict(vars(self.args)),
                    'model': self.state_dict(), 'optimizer': self.opt.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

    def load_network(self, load_path, network, strict=True):
        if network is not self.net:
            raise ValueError('Medical checkpoint must load into this model.net.')
        state = load_tensors(load_path)
        if state.get('format') != 'medical-cl-v1' or state.get('method') != self.NAME:
            raise ValueError('Expected a complete checkpoint for ' + self.NAME)
        if self.NAME == 's3r' and state['config'].get('s3r_optimizer', 'SGD') != self.args.s3r_optimizer:
            raise ValueError('Checkpoint objective configuration mismatch: s3r_optimizer')
        for key, legacy in ((self.NAME + '_bce_impl', 'probability'),):
            if self.NAME in ('s3r', 'ted') and state['config'].get(key, legacy) != getattr(self.args, key):
                raise ValueError('Checkpoint objective configuration mismatch: ' + key)
        if self.NAME == 'ted' and state['config'].get('ted_optimizer', 'SGD') != self.args.ted_optimizer:
            raise ValueError('Checkpoint objective configuration mismatch: ted_optimizer')
        if self.NAME in ('cit','cit_organ') and state['config'].get('cit_optimizer','SGD') != self.args.cit_optimizer:
            raise ValueError('Checkpoint objective configuration mismatch: cit_optimizer')
        objective_keys = ('ted_lambda', 'ted_gugf', 'ted_supervised',
                          's3r_lambda', 's3r_alpha', 's3r_alpha1', 's3r_beta',
                          's3r_beta1', 's3r_dropout', 's3r_mc_samples',
                          's3r_importance_batches', 'clip_pseudo_threshold')
        objective_keys += tuple(key for key in state['config']
                                if key.startswith(('ted_', 's3r_', 'ewf_', 'lgkd_', 'nest_', 'cit_', 'rebac_', 'ldka_', 'pcdd_', 'organ_')))
        for key in objective_keys:
            if key in state['config'] and state['config'][key] != getattr(self.args, key):
                raise ValueError('Checkpoint objective configuration mismatch: ' + key)
        self.load_state_dict(state['model'], strict=strict)
        self.opt.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        # load_tensors maps to CPU; move optimizer buffers with their parameters.
        for param, values in self.opt.state.items():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(param.device)

    def optimization_step(self, loss, **parts):
        if not torch.isfinite(loss):
            raise FloatingPointError('{} produced a non-finite loss'.format(self.NAME))
        loss.backward()
        finite_checks = [torch.isfinite(param.grad).all()
                         for param in self.net.parameters() if param.grad is not None]
        if finite_checks and not torch.stack(finite_checks).all():
            raise FloatingPointError('{} produced non-finite gradients'.format(self.NAME))
        clip = getattr(self, 'gradient_clip_max_norm', 0.)
        if clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip,
                                                  error_if_nonfinite=True)
            parts.update(gradient_norm_before_clip=norm, gradient_clip_max_norm=clip)
        self.opt.step()
        self.last_losses = {key: float(value.detach()) if torch.is_tensor(value)
                            else float(value) for key, value in parts.items()}
        self.last_losses['total'] = float(loss.detach())
        return self.last_losses['total']
