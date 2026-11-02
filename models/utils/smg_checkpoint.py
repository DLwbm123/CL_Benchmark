"""Checkpoint components needed by a future SMG training lifecycle.

The caller must separately preserve data-loader/sampler state and the original
wall-clock deadline. This module does not imply exact mid-epoch continuation.
"""
import os
import random
import math
import numpy as np
import torch


def _finite(value):
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def save_state(path, segmenter, generator, segmenter_optimizer,
               generator_optimizer, task, epoch, metadata, extra=None):
    if type(task) is not int or task < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError('task and epoch must be nonnegative integers')
    numpy_state = np.random.get_state()
    device = next(segmenter.parameters()).device
    rng = {'python': random.getstate(), 'numpy_kind': numpy_state[0],
           'numpy_keys': numpy_state[1].tolist(), 'numpy_pos': numpy_state[2],
           'numpy_has_gauss': numpy_state[3], 'numpy_cached_gaussian': numpy_state[4],
           'torch': torch.get_rng_state(),
           'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
    state = {'format': 'smg-components-v1', 'task': task, 'epoch': epoch,
             'metadata': metadata, 'segmenter': segmenter.state_dict(),
             'generator': generator.state_dict(), 'segmenter_optimizer': segmenter_optimizer.state_dict(),
             'generator_optimizer': generator_optimizer.state_dict(), 'rng': rng,
             'extra': extra}
    path = os.fspath(path)
    tmp = path + '.partial'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path, segmenter, generator, segmenter_optimizer, generator_optimizer,
               expected_metadata):
    device = next(segmenter.parameters()).device
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('format') != 'smg-components-v1' or state['metadata'] != expected_metadata:
        raise ValueError('Checkpoint format or frozen protocol mismatch')
    if state['rng']['cuda'] is not None and device.type != 'cuda':
        raise ValueError('CUDA RNG checkpoint requires a CUDA segmenter')
    for key in ('segmenter', 'generator', 'segmenter_optimizer', 'generator_optimizer'):
        if not _finite(state[key]):
            raise ValueError('Nonfinite checkpoint state: ' + key)
    segmenter.load_state_dict(state['segmenter'])
    generator.load_state_dict(state['generator'])
    segmenter_optimizer.load_state_dict(state['segmenter_optimizer'])
    generator_optimizer.load_state_dict(state['generator_optimizer'])
    rng = state['rng']
    random.setstate(rng['python'])
    np.random.set_state((rng['numpy_kind'], np.array(rng['numpy_keys'], dtype=np.uint32),
                         rng['numpy_pos'], rng['numpy_has_gauss'], rng['numpy_cached_gaussian']))
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state(rng['cuda'].cpu(), device)
    return state['task'], state['epoch']
