"""Exercise the real trainer/evaluator/CSV writer using only in-memory tensors."""
import csv
import logging

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from conftest import CLASS_METHODS


class SyntheticStream:
    def __init__(self, args, class_il=False):
        self.args = args
        self.SETTING = 'class-il' if class_il else 'domain-il'
        self.NAME = 'seq-mmwhs' if class_il else 'seq-prostate'
        self.N_TASKS = 3 if class_il else 2
        self.N_CLASSES_PER_TASK = [3, 2, 2] if class_il else 2
        self.test_loaders = []
        self.test_loader = None
        self.x = torch.linspace(-1, 1, 2 * 32 * 32).reshape(2, 1, 32, 32)

    def get_data_loaders(self, task):
        y = torch.zeros(2, 32, 32, dtype=torch.long)
        if self.SETTING == 'class-il':
            for local in range(self.N_CLASSES_PER_TASK[task]):
                y[:, 5:24, 7 + local * 6:13 + local * 6] = local + 1
        else:
            y[:, 5:24, 7:25] = 1
        shift = sum(self.N_CLASSES_PER_TASK[:task]) if self.SETTING == 'class-il' else 0
        global_y = torch.where(y > 0, y + shift, y)
        self.train_dataset = TensorDataset(self.x, global_y)
        self.test_dataset = TensorDataset(self.x, y)
        self.test_dataset.shift = shift
        self.test_dataset.patient_info = [1]  # legacy inclusive slice endpoints
        self.train_loader = DataLoader(self.train_dataset, batch_size=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)
        self.test_loaders.append(self.test_loader)
        return self.train_loader, self.test_loader

    def get_whole_testloader(self):
        y = torch.zeros(2, 32, 32, dtype=torch.long)
        y[:, 5:24, 7:25] = 1
        ds = TensorDataset(self.x, y)
        ds.shift = 0
        self.test_loader = DataLoader(ds, batch_size=1)
        return self.test_loader


@pytest.mark.parametrize('name', ['ted', 's3r', 'clip_cis', 'ewf', 'lgkd', 'nest', 'cit', 'rebac', 'ldka', 'pcdd'])
def test_real_training_loop_saves_complete_checkpoints_and_csv(
        name, factory, tmp_path, monkeypatch):
    import utils.training as training
    model = factory(name, classes=8 if name in CLASS_METHODS else 2)
    args = model.args
    args.n_epochs = 2 if name == 'ldka' else 1
    args.save_freq = 0
    args.name = 'synthetic_only'
    args.output_folder = str(tmp_path / 'checkpoints')
    monkeypatch.chdir(tmp_path)
    stream = SyntheticStream(args, name in CLASS_METHODS)
    monkeypatch.setattr(training, 'get_dataset', lambda _: SyntheticStream(args, name in CLASS_METHODS))
    training.train(model, stream, args, logging.getLogger('synthetic_test'))
    assert model.current_task == stream.N_TASKS
    for task in range(stream.N_TASKS):
        path = tmp_path / 'checkpoints' / 'T{}_latest.pth'.format(task)
        state = torch.load(path, weights_only=True)
        assert state['format'] == 'medical-cl-v1'
        assert state['method'] == name
        assert state['model']['_extra_state']['current_task'] == task + 1
        if name == 'ldka':
            assert state['model']['_extra_state']['knowledge']['progress'] == pytest.approx(2/3)
            assert state['model']['competition'].sum() > 0
            assert state['optimizer']['param_groups'][0]['lr'] <= (args.lr if task == 0 else args.ldka_incremental_lr)
    rows = list(csv.DictReader((tmp_path / 'checkpoints/cl_results.csv').open()))
    assert len(rows) == 1 and rows[0]['restricted_model_ability'] == ''
    assert rows[0]['new_task_dice'] != ''

    # Restore through the existing test-only path, without borrowing its
    # historical hard-coded SGD scores as unrelated normalization references.
    args.pretrain = str(tmp_path / 'checkpoints')
    args.save_image = False
    fresh = factory(name, classes=8 if name in CLASS_METHODS else 2)
    fresh.args = args
    training.test(fresh, SyntheticStream(args, name in CLASS_METHODS), args,
                  logging.getLogger('synthetic_test'))
    evaluated = list(csv.DictReader((tmp_path / 'checkpoints/test_results_taskincre.csv').open()))
    assert len(evaluated) == 1 and evaluated[0]['restricted_model_ability'] == ''


def test_reject_mismatched_checkpoint_objective(factory, batch, tmp_path):
    from conftest import dataset_for
    x, y = batch
    model = factory('ted')
    model.begin_task(dataset_for(x, y))
    model.save_network(model.net, str(tmp_path), 0, 'latest')
    changed = factory('ted', ted_lambda=0.1)
    with pytest.raises(ValueError, match='configuration mismatch'):
        changed.load_network(str(tmp_path / 'T0_latest.pth'), changed.net)


def test_rma_reference_requires_matching_metadata(factory, tmp_path):
    import json
    model = factory('ted')
    path = tmp_path / 'references.json'
    model.args.rma_reference = str(path)
    data = {'dataset': model.args.dataset, 'seed': model.args.seed,
            'order': model.args.order, 'results': [[0.8, 0.1], [0.2, 0.7]]}
    path.write_text(json.dumps(data))
    assert model.rma_reference_results(2) == data['results']
    data['seed'] = 100
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='mismatch'):
        model.rma_reference_results(2)


def test_inference_parameter_count_does_not_double_when_teacher_is_created(factory, batch):
    from conftest import dataset_for
    x, y = batch
    model = factory('ted')
    before = model.evaluation_parameter_count()
    model.begin_task(dataset_for(x, y))
    model.end_task(dataset_for(x, y))
    assert model.evaluation_parameter_count() == before
    assert sum(p.numel() for p in model.parameters()) == 2 * before
