import copy
import os

import h5py
import numpy as np
import pytest
import torch

from datasets.seq_prostate import ProstateCL_Dataset
from datasets.seq_mmwhs import MMWHSCL_Dataset
from datasets.seq_task_incre import TaskIncre_Dataset
from datasets.utils.ram_cache import prepare_h5_dataset
from conftest import dataset_for


@pytest.mark.parametrize('kind,patch', [('domain',8),('domain',13),('class',8),('organ',8),('organ',13)])
def test_cached_tensors_preserve_resize_dtype_label_shift_and_rng(tmp_path, monkeypatch, kind, patch):
    group, name, cls, task = {
        'domain': ('Domain_Prostate', 'BIDMC', ProstateCL_Dataset, '1'),
        'class': ('MMWHS', 'ra_rv', MMWHSCL_Dataset, '2'),
        'organ': ('Task_incre', 'UtahI', TaskIncre_Dataset, '1'),
    }[kind]
    folder = tmp_path/group
    folder.mkdir()
    rng = np.random.default_rng(7)
    with h5py.File(folder/(name+'.h5'), 'w') as f:
        f['train_images'] = rng.normal(size=(8,8,4)).astype('float64')
        labels = (rng.random((8,8,4))*2).astype('int64') if kind == 'class' else rng.random((8,8,4))
        f['train_labels'] = labels
        f['patient_info_val'] = [3]
    monkeypatch.setenv('CL_DATA_ROOT', str(tmp_path))
    monkeypatch.delenv('CL_H5_RAM_CACHE', raising=False)
    ds = cls(task, 'train', patch)
    before = [ds[i] for i in range(4)]
    state = torch.get_rng_state().clone()
    monkeypatch.setenv('CL_H5_RAM_CACHE', '1')
    prepare_h5_dataset(ds)
    assert torch.equal(torch.get_rng_state(), state)
    assert ds.images_all[:,:,0].flags.c_contiguous
    assert ds.labels_all.dtype == labels.dtype
    for i in range(4):
        assert all(torch.equal(a,b) for a,b in zip(before[i], ds[i]))


def test_batched_finite_checks_have_identical_sgd_update(factory, batch):
    fast = factory('ted')
    slow = factory('ted')
    slow.net.load_state_dict(copy.deepcopy(fast.net.state_dict()))
    x,y = batch
    fast.begin_task(dataset_for(x,y))
    slow.begin_task(dataset_for(x,y))
    # Previous implementation, used only as an independent reference here.
    def legacy_step(loss, **parts):
        assert torch.isfinite(loss)
        loss.backward()
        assert not any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in slow.net.parameters())
        slow.opt.step()
        return float(loss.detach())
    slow.optimization_step = legacy_step
    assert fast.observe(x,y,x) == slow.observe(x,y,x)
    for key,value in fast.net.state_dict().items():
        assert torch.equal(value, slow.net.state_dict()[key])


def test_batched_check_still_rejects_nonfinite_gradient(factory):
    model = factory('ted')
    param = next(model.net.parameters())
    param.register_hook(lambda grad: torch.full_like(grad, float('nan')))
    with pytest.raises(FloatingPointError, match='gradients'):
        model.optimization_step(param.square().sum())
