"""Six-domain validation pilots; author components with explicit HDF5 adaptation.
All patient-derived artifacts stay in the private experiment directory.
"""
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from types import MethodType
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from source_components import focus_components, cauaug_components

NAMES = ['BIDMC', 'HK', 'ISBI', 'UCL', 'ISBI_1.5', 'I2CVB']


def finite(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(label)
    elif isinstance(value, dict):
        for v in value.values():
            finite(v, label)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise FloatingPointError(label)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def atomic_checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dataset(root, site, split, block=None, method=None):
    if split not in ('train', 'val'):
        raise ValueError('test access forbidden in this pilot')
    with h5py.File(Path(root)/(NAMES[site-1]+'.h5'), 'r') as f:
        x = np.asarray(f[split+'_images']).transpose(2, 0, 1).copy()
        y = np.asarray(f[split+'_labels']).transpose(2, 0, 1).copy()
        if split == 'val':
            ends = np.asarray(f['patient_info_val']).astype(int)
            if len(ends) < 2 or ends[-1] != len(x)-1 or not np.all(np.diff(ends)>0):
                raise ValueError('invalid sequence boundaries')
            boundary = int(ends[0])+1
            selected = slice(0,boundary) if block == 'selection' else slice(boundary,None)
            x,y = x[selected],y[selected]
    if x.shape != y.shape or x.shape[1:] != (256,256):
        raise ValueError('unexpected data shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.min()<0 or y.max()>1:
        raise ValueError('invalid images or labels')
    # Existing benchmark loader casts fractional HDF5 labels to integer.
    # Preserve that frozen label policy; do not introduce a new threshold here.
    y=y.astype(np.int64)
    if not np.isin(y,[0,1]).all():raise ValueError('invalid integer labels')
    # TED expects [0,1] followed by clipping. With no verified volume boundaries, explicitly
    # use slice-level minmax (an adaptation, NOT author case-level minmax).
    x=torch.from_numpy(x).float().unsqueeze(1)
    if method=='ted':
        low=x.amin((2,3),keepdim=True);span=x.amax((2,3),keepdim=True)-low
        x=torch.where(span>0,(x-low)/span.clamp_min(1e-12),torch.zeros_like(x))
    if method=='ted':x=x.clamp(.001,.99)
    return TensorDataset(x, torch.from_numpy(y).long())


