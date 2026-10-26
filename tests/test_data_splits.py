"""The held-out MMWHS entry must not silently evaluate validation slices."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from datasets.seq_mmwhs import SequentialMMWHS


@pytest.mark.parametrize('test_only,expected', [(False, 2), (True, 3)])
def test_mmwhs_selects_actual_h5_split(tmp_path, monkeypatch, test_only, expected):
    root = tmp_path / 'MMWHS'
    root.mkdir()
    for name in ('myo_lv_la', 'ra_rv', 'ao_pa'):
        with h5py.File(root / (name + '.h5'), 'w') as f:
            for split, count in [('train', 1), ('val', 2), ('test', 3)]:
                for kind in ('images', 'labels'):
                    f[split + '_' + kind] = np.zeros((8, 8, count), dtype=np.float32)
                f['patient_info_' + split] = [count - 1]
    monkeypatch.setenv('CL_DATA_ROOT', str(tmp_path))
    stream = SequentialMMWHS(SimpleNamespace(test_only=test_only, batch_size=1))
    train, held_out = stream.get_data_loaders(2)
    assert len(train.dataset) == 1
    assert len(held_out.dataset) == expected
    assert held_out.dataset.mode == ('test' if test_only else 'val')
