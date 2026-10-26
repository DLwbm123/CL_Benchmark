"""Task-aware Organ-CL adapters: all images, labels and text vectors synthetic."""
import copy
import csv
import logging
import math
from pathlib import Path
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from conftest import ORGAN_METHODS
from test_framework_integration import SyntheticStream
from models.utils.organ import ORGAN_NAMES
from models.clip_cis_organ import independent_organ_targets


def stream_metadata():
    return SimpleNamespace(SETTING='task-il', N_CLASSES_PER_TASK=2, N_TASKS=4, TASK_NAMES=ORGAN_NAMES.copy())


class SyntheticOrganStream(SyntheticStream):
    def __init__(self, args):
        super().__init__(args)
        self.SETTING, self.NAME = 'task-il', 'seq-taskincre'
        self.N_TASKS, self.N_CLASSES_PER_TASK = 4, 2
        self.TASK_NAMES = ORGAN_NAMES.copy()
        self.original_x = self.x.clone()

    def get_data_loaders(self, task):
        self.x = self.original_x+task*.1
        return super().get_data_loaders(task)


def future_parameters(model, task):
    if model.NAME in ('ewf_organ', 'ldka_organ'):
        return [p for i in range(task+1, 4) for p in model.net.heads[str(i)].parameters()]
    if model.NAME == 'cit_organ':
        return [model.net.queries.weight[task+1:]]
    return [p for c in model.net.controllers[task+1:] for p in c.parameters()]


@pytest.mark.parametrize('name', ORGAN_METHODS)
def test_four_task_local_labels_and_complete_continuation(name, factory, batch, tmp_path):
    model = factory(name)
    x, y = batch
    ds = stream_metadata()
    # Random future heads must be evaluable for the original FWT protocol.
    model.net.eval()
    assert model(x, 3).shape == (2, 2, 32, 32)
    for task in range(4):
        xx = x+.1*task
        model.begin_task(ds)
        model.train()
        teachers = list(model.teachers) if name == 'cit_organ' else ([model.teacher] if model.teacher is not None else [])
        frozen = [copy.deepcopy(t.state_dict()) for t in teachers]
        future = [p.detach().clone() for p in future_parameters(model, task)]
        old_heads = ({i: copy.deepcopy(model.net.heads[str(i)].state_dict()) for i in range(task)}
                     if name in ('ewf_organ', 'ldka_organ') else {})
        assert math.isfinite(model.observe(xx, y, xx))  # every task GT remains 0/1
        for expected, parameter in zip(future, future_parameters(model, task)):
            torch.testing.assert_close(expected, parameter, rtol=0, atol=0)
        for before, teacher in zip(frozen, teachers):
            assert not teacher.training
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            for key, tensor in before.items():
                torch.testing.assert_close(tensor, teacher.state_dict()[key], rtol=0, atol=0)
        for i, before in old_heads.items():
            for key, value in before.items():
                torch.testing.assert_close(value, model.net.heads[str(i)].state_dict()[key], rtol=0, atol=0)
        model.net.eval()
        outputs = [model(xx, i).detach().clone() for i in range(4)]
        assert all(torch.isfinite(o).all() and o.shape == (2, 2, 32, 32) for o in outputs)
        assert not torch.equal(outputs[0], outputs[1])
        model.save_network(model.net, str(tmp_path), task, 'mid')
        restored = factory(name)
        restored.load_network(str(tmp_path/f'T{task}_mid.pth'), restored.net)
        restored.net.eval()
        assert restored.task_open and restored.current_task == task
        for i in range(4):
            torch.testing.assert_close(restored(xx, i), outputs[i], rtol=0, atol=0)
        # CIT dropout must use the same RNG realization for continuation equality.
        rng = torch.get_rng_state()
        model.train()
        expected_loss = model.observe(xx, y, xx)
        torch.set_rng_state(rng)
        restored.train()
        assert restored.observe(xx, y, xx) == pytest.approx(expected_loss, rel=1e-6, abs=1e-6)
        restored.end_task(ds)
        model = restored
    assert model.current_task == 4 and not model.task_open


@pytest.mark.parametrize('name', ORGAN_METHODS)
def test_original_trainer_and_test_only_organ_route(name, factory, tmp_path, monkeypatch):
    import utils.training as training
    model = factory(name)
    args = model.args
    args.n_epochs = 2 if name == 'ldka_organ' else 1
    args.save_freq, args.name = 0, 'synthetic_only'
    args.output_folder = str(tmp_path/'out')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, 'get_dataset', lambda _: SyntheticOrganStream(args))
    training.train(model, SyntheticOrganStream(args), args, logging.getLogger('synthetic'))
    assert model.current_task == 4
    for task in range(4):
        checkpoint = torch.load(tmp_path/'out'/f'T{task}_latest.pth', weights_only=True)
        assert checkpoint['method'] == name
        assert checkpoint['model']['_extra_state']['current_task'] == task+1
        if name == 'ldka_organ':
            assert checkpoint['model']['_extra_state']['knowledge']['progress'] == pytest.approx(2/3)
    assert len(list(csv.DictReader((tmp_path/'out/cl_results.csv').open()))) == 1
    args.pretrain, args.save_image = str(tmp_path/'out'), False
    fresh = factory(name)
    fresh.args = args
    training.test(fresh, SyntheticOrganStream(args), args, logging.getLogger('synthetic'))
    rows = list(csv.DictReader((tmp_path/'out/test_results_taskincre.csv').open()))
    assert len(rows) == 1 and rows[0]['restricted_model_ability'] == ''


@pytest.mark.parametrize('name', ORGAN_METHODS)
@pytest.mark.parametrize('label', [0, 255])
def test_empty_organ_regions_are_finite(name, label, factory, batch):
    model = factory(name)
    x, y = batch
    ds = stream_metadata()
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    model.begin_task(ds)
    value = model.observe(x, torch.full_like(y, label), x)
    assert math.isfinite(value)
    if label == 255:
        assert value == 0


@pytest.mark.parametrize('name', ORGAN_METHODS)
def test_organ_requires_task_id_local_labels_and_matching_metadata(name, factory, batch, tmp_path):
    from run import read_configuration
    x, y = batch
    model = factory(name)
    root = Path(__file__).resolve().parents[1]
    assert '--model' in read_configuration(str(root/'configs'/f'{name}.json'))
    for index in [None, -1, 4, True, 1.5]:
        with pytest.raises(ValueError, match='task ID'):
            model(x, index)
    with pytest.raises(ValueError, match='supports'):
        model.begin_task(SimpleNamespace(SETTING='class-il'))
    with pytest.raises(ValueError, match='count/order'):
        model.begin_task(SimpleNamespace(SETTING='task-il', N_CLASSES_PER_TASK=2, N_TASKS=3))
    model.begin_task(stream_metadata())
    with pytest.raises(ValueError, match='local 0/1'):
        model.observe(x, y*2, x)
    model.save_network(model.net, str(tmp_path), 0, 'mid')
    names = ORGAN_NAMES[::-1]
    if name != 'clip_cis_organ':
        changed = factory(name, organ_task_names=names)
        with pytest.raises(ValueError, match='configuration mismatch'):
            changed.load_network(str(tmp_path/'T0_mid.pth'), changed.net)
    else:
        path = tmp_path/'changed_embeddings.pt'
        original = torch.load(model.args.clip_embeddings, weights_only=True)
        original['embeddings'] = original['embeddings']+1
        torch.save(original, path)
        changed = factory(name, clip_embeddings=str(path))
        with pytest.raises(ValueError, match='embeddings differ'):
            changed.load_network(str(tmp_path/'T0_mid.pth'), changed.net)


def test_ewf_organ_fuses_only_shared_and_previous_heads(factory, batch):
    model = factory('ewf_organ')
    x, y = batch
    ds = stream_metadata()
    model.begin_task(ds)
    with torch.no_grad():
        for p in model.net.parameters():
            p.fill_(2.)
    model.end_task(ds)
    model.begin_task(ds)
    with torch.no_grad():
        for p in model.net.parameters():
            p.fill_(4.)
    model.fuse_endpoint()
    expected = 2+2/math.sqrt(2)
    torch.testing.assert_close(model.net.backbone.conv1.weight, torch.full_like(model.net.backbone.conv1.weight, expected))
    torch.testing.assert_close(model.net.heads['0'].weight, torch.full_like(model.net.heads['0'].weight, expected))
    for i in (1, 2, 3):
        assert (model.net.heads[str(i)].weight == 4).all()
    assert model.fusion_alpha == math.sqrt(.5)


def test_old_clip_organ_masks_survive_current_foreground():
    labels = torch.ones(1, 2, 2, dtype=torch.long)
    old = torch.full((1, 2, 2, 2), 20.)
    targets = independent_organ_targets(labels, old, .5)
    assert torch.equal(targets, torch.ones(1, 3, 2, 2))


@pytest.mark.parametrize('name', ['cit_organ', 'clip_cis_organ'])
def test_independent_organ_background_and_query_controller_routing(name, factory, batch):
    model = factory(name)
    x, y = batch
    model.begin_task(stream_metadata())
    model.net.eval()
    before = model(x, 0).detach().clone()
    with torch.no_grad():
        if name == 'cit_organ':
            model.net.queries.weight[1].mul_(100.)
        else:
            model.net.controllers[1][-1].bias.add_(10.)
    torch.testing.assert_close(before, model(x, 0), rtol=0, atol=0)
    assert torch.isfinite(model(x, 1)).all()
    probabilities = before.softmax(1)
    if name == 'cit_organ':
        out = model.net.components(x, count=1)
        score = out['presence'].softmax(-1)[:, 0, 0, None, None, None]*out['masks'][:, :1].sigmoid()
        score = torch.nn.functional.interpolate(score, x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        score = model.net.foreground(x, indices=[0]).sigmoid()
    torch.testing.assert_close(probabilities[:, 1:], score)
    torch.testing.assert_close(probabilities[:, :1], 1-score)


@pytest.mark.parametrize('name', ORGAN_METHODS)
def test_native_dataset_backbone_and_256_organ_step(name, factory):
    from datasets.seq_task_incre import SequentialTaskIncre
    model = factory(name, baseline_size='mid')
    backbone = SequentialTaskIncre.get_backbone(model.args)
    assert len(backbone.last) == 4
    assert backbone.last['0'].in_channels == 32
    assert backbone.last['0'].out_channels == 2
    model.begin_task(stream_metadata())
    x = torch.randn(1, 1, 256, 256)
    y = torch.zeros(1, 256, 256, dtype=torch.long)
    y[:, 64:192, 64:192] = 1
    assert math.isfinite(model.observe(x, y, x))
    model.net.eval()
    assert model(x, 0).shape == (1, 2, 256, 256)
    assert torch.isfinite(model(x, 0)).all()
