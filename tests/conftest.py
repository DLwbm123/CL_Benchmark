"""All fixtures are synthetic; no patient file, GPU, or network is accessed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.ResUnet import resunet32
from models import get_model

CLASS_METHODS = ('clip_cis', 'ewf', 'lgkd', 'nest', 'cit', 'rebac', 'ldka', 'pcdd')
ORGAN_METHODS = ('ewf_organ', 'ldka_organ', 'cit_organ', 'clip_cis_organ')

@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(17)
    torch.set_num_threads(2)


@pytest.fixture
def batch():
    x = torch.randn(2, 1, 32, 32)
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, 6:22, 8:24] = 1
    return x, y


@pytest.fixture
def factory(tmp_path):
    def build(name, classes=2, **overrides):
        from importlib import import_module
        parser = import_module('models.' + name).get_parser()
        dataset_name = 'seq-taskincre' if name in ORGAN_METHODS else 'seq-mmwhs' if name in CLASS_METHODS else 'seq-prostate'
        cli = ['--model', name, '--dataset', dataset_name,
               '--baseline_size', 'small']
        if name in ('clip_cis', 'clip_cis_organ'):
            organs = name in ORGAN_METHODS
            count = 4 if organs else classes-1
            path = tmp_path / ('synthetic_embeddings_organ.pt' if organs else 'synthetic_embeddings_{}.pt'.format(classes))
            if not path.exists():
                torch.save({'embeddings': torch.randn(count, 512),
                            'class_ids': list(range(1, count+1)),
                            'class_names': (['left_atrium', 'prostate', 'liver', 'brain_tumor'] if organs else
                                            ['synthetic_{}'.format(i) for i in range(1, classes)])}, path)
            cli += ['--clip_embeddings', str(path)]
        if name == 's3r':
            # Exercise ACNN pretraining, not a random frozen shape network.
            cli += ['--s3r_shape_epochs', '1', '--s3r_mc_samples', '2',
                    '--s3r_importance_batches', '1']
        if name == 'nest':
            cli += ['--nest_pre_epochs', '1', '--nest_selection_size', '16']
        args = parser.parse_args(cli)
        for key, value in overrides.items():
            setattr(args, key, value)
        backbone = resunet32(args.baseline_size)
        width = backbone.last.in_channels
        backbone.last = (nn.ModuleDict({str(i): nn.Conv2d(width, 2, 1, bias=False) for i in range(args.organ_tasks)})
                         if name in ORGAN_METHODS else nn.Conv2d(width, classes, 1, bias=False))
        model = get_model(args, backbone, nn.functional.cross_entropy, None)
        model.device = torch.device('cpu')
        model.net.cpu()
        return model
    return build


def dataset_for(x, y, setting='domain-il', counts=2):
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    return SimpleNamespace(SETTING=setting, N_CLASSES_PER_TASK=counts,
                           train_loader=loader)
