"""Opt-in, read-only H5 split cache; no transforms or random-number draws.

Call in the parent before iterating a loader so forked workers share the arrays.
Only splits that are actually trained/evaluated are materialized. The registry
budget does not include arrays still referenced by a live dataset or worker.
"""
from collections import OrderedDict
import os

import h5py
import numpy as np

_CACHE = OrderedDict()
_BYTES = 0


def _materialize(data):
    global _BYTES
    if not isinstance(data, h5py.Dataset) or len(data.shape) != 3:
        return data
    limit = int(os.environ.get('CL_H5_RAM_CACHE_MB', '4096')) * 1024**2
    required = int(np.prod(data.shape)) * data.dtype.itemsize
    if required > limit:
        return data
    key = (os.path.realpath(data.file.filename), data.name)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    # Retain [H,W,N] indexing, but make each [:,:,index] plane C-contiguous.
    # Raw dtypes and values are unchanged; resize/casts remain in __getitem__.
    array = np.moveaxis(np.ascontiguousarray(np.moveaxis(data[...], 2, 0)), 0, 2)
    array.flags.writeable = False
    while _CACHE and _BYTES + array.nbytes > limit:
        _, old = _CACHE.popitem(last=False)
        _BYTES -= old.nbytes
    _CACHE[key] = array
    _BYTES += array.nbytes
    return array


def prepare_h5_dataset(dataset):
    if os.environ.get('CL_H5_RAM_CACHE', '0') != '1':
        return
    for attr in ('images_all', 'labels_all'):
        if hasattr(dataset, attr):
            setattr(dataset, attr, _materialize(getattr(dataset, attr)))
