import copy
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from conftest import dataset_for
from models import get_all_models
from models.ted import adaptive_knowledge_transfer
from models.clip_cis import merged_targets, read_embeddings
from models.utils.shape import signed_distance
from models.s3r import confidence_mask


@pytest.mark.parametrize('name', ['s3r', 'ted', 'clip_cis'])
def test_two_task_update_freeze_and_checkpoint(name, factory, batch, tmp_path):
    x, y = batch
    is_class = name == 'clip_cis'
    model = factory(name, classes=3 if is_class else 2)
    first = dataset_for(x, y, 'class-il' if is_class else 'domain-il',
                        [1, 1] if is_class else 2)
    model.begin_task(first)
    before = {n: p.detach().clone() for n, p in model.net.named_parameters()}
    loss = model.observe(x, y, x)
    assert math.isfinite(loss)
    assert any(not torch.equal(before[n], p) for n, p in model.net.named_parameters())
    model.end_task(first)
    assert model.current_task == 1 and not model.task_open
    teacher_state = (copy.deepcopy(model.teacher.state_dict())
                     if model.teacher is not None else None)
    if model.teacher is not None:
        assert not any(p.requires_grad for p in model.teacher.parameters())
        assert all(p.grad is None for p in model.teacher.parameters())
    if name == 's3r':
        assert model.importance and any(p.sum() > 0 for p in model.importance.values())
        assert all(torch.isfinite(p).all() and (p >= 0).all()
                   for p in model.importance.values())
        assert not any(p.requires_grad for p in model.net.shape_encoder.parameters())
        assert all(p.grad is None for p in model.net.shape_encoder.parameters())
        assert all(not m.training for m in model.net.shape_encoder.modules())

    y2 = y * 2 if is_class else 1 - y
    second = dataset_for(x, y2, first.SETTING, first.N_CLASSES_PER_TASK)
    model.begin_task(second)
    assert math.isfinite(model.observe(x, y2, x))
    assert math.isfinite(model.observe(x, y2, x))
    if name == 'ted':
        assert model.last_losses['image_kd'] > 0 and model.last_losses['ska'] > 0
    if name == 's3r':
        assert model.last_losses['regularization'] > 0
    if model.teacher is not None:
        assert not model.teacher.training
        for key, value in teacher_state.items():
            assert torch.equal(value, model.teacher.state_dict()[key]), key
    model.net.eval()
    output = model(x, 0)
    assert output.shape == (2, 3 if is_class else 2, 32, 32)
    assert torch.isfinite(output).all()

    # Mid-task serialization must preserve optimizer, teacher, and importance,
    # not merely produce an inference-only network checkpoint.
    model.save_network(model.net, str(tmp_path), 1, 'latest')
    restored = factory(name, classes=3 if is_class else 2)
    restored.load_network(str(tmp_path / 'T1_latest.pth'), restored.net)
    restored.net.eval()
    assert restored.current_task == model.current_task and restored.task_open
    torch.testing.assert_close(restored(x), output, rtol=0, atol=0)
    restored.net.train()
    assert math.isfinite(restored.observe(x, y2, x))
    restored.end_task(second)
    assert restored.current_task == 2 and not restored.task_open


@pytest.mark.parametrize('name', ['s3r', 'ted', 'clip_cis'])
def test_reject_unsupported_settings_and_lifecycle(name, factory, batch):
    x, y = batch
    model = factory(name, classes=3 if name == 'clip_cis' else 2)
    with pytest.raises(ValueError, match='supports'):
        model.begin_task(dataset_for(x, y, 'task-il'))
    with pytest.raises(RuntimeError, match='begin_task'):
        model.observe(x, y, x)


@pytest.mark.parametrize('batch_size', [1, 2, 3])
@pytest.mark.parametrize('gugf', [False, True])
def test_ted_akt_matches_literal_author_formula(batch_size, gugf):
    logits = torch.randn(batch_size, 2, 8, 8, requires_grad=True)
    old = torch.randn_like(logits).softmax(1)
    probabilities = logits.softmax(1)
    target = old.argmax(1, keepdim=True)
    log_probs = torch.log(probabilities + 1e-5)
    log_target = log_probs.gather(1, target)
    errors = (log_probs.argmax(1, keepdim=True) != target).float().detach()
    uncertainty = ((1 - (old * probabilities).sum(1)).detach()
                   if gugf else torch.ones(1))
    expected = -(uncertainty * errors * log_target).sum() / (errors.sum() + 1)
    result = adaptive_knowledge_transfer(logits, old, gugf)
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_ted_no_disagreement_has_zero_akt():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    result = adaptive_knowledge_transfer(logits, logits.detach().softmax(1))
    assert result.item() == 0
    result.backward()
    assert logits.grad.abs().sum() == 0


def test_s3r_constant_uncertainty_and_sdf_edge_cases():
    mc = torch.full((2, 2, 2, 8, 8), 0.5)
    torch.testing.assert_close(confidence_mask(mc), torch.ones(2, 1, 8, 8))
    masks = torch.zeros(3, 16, 16, dtype=torch.long)
    masks[1] = 1
    masks[2, 3:13, 3:13] = 1
    sdf = signed_distance(masks)
    assert sdf.shape == (3, 1, 16, 16) and torch.isfinite(sdf).all()
    assert sdf.min() >= -1 and sdf.max() <= 1
    assert sdf[0].abs().sum() == 0
    assert sdf[2, 0, 7, 7] < 0 and sdf[2, 0, 0, 0] > 0


def test_s3r_refuses_untrained_shape_prior(factory, batch):
    x, y = batch
    model = factory('s3r', s3r_shape_epochs=0)
    with pytest.raises(ValueError, match='Random frozen shape codes'):
        model.begin_task(dataset_for(x, y))
    assert not model.task_open


def test_s3r_rejects_multiclass_backbone(factory):
    with pytest.raises(ValueError, match='binary Domain-CL'):
        factory('s3r', classes=4)


def test_ted_supports_shared_multiclass_domain_learning(factory, batch):
    x, y = batch
    model = factory('ted', classes=4)
    y = y * 3
    ds = dataset_for(x, y, counts=4)
    model.begin_task(ds)
    assert math.isfinite(model.observe(x, y, x))
    model.end_task(ds)
    model.begin_task(ds)
    assert math.isfinite(model.observe(x, y, x))
    assert model(x).shape == (2, 4, 32, 32)


@pytest.mark.parametrize('name', ['s3r', 'ted', 'clip_cis'])
def test_reject_legacy_loss_wrappers(name, factory):
    with pytest.raises(ValueError, match='legacy'):
        factory(name, classes=3 if name == 'clip_cis' else 2, mib=True)


def test_s3r_shape_checkpoint_branch(factory, batch, tmp_path):
    x, y = batch
    model = factory('s3r')
    ds = dataset_for(x, y)
    model.begin_task(ds)
    path = tmp_path / 'synthetic_acnn.pt'
    torch.save(model.net.shape_encoder.state_dict(), path)
    loaded = factory('s3r', s3r_shape_epochs=0, s3r_shape_checkpoint=str(path))
    loaded.begin_task(ds)
    assert bool(loaded.net.shape_ready)
    assert math.isfinite(loaded.observe(x, y, x))


def test_clip_current_ground_truth_wins_and_labels_are_not_mutated():
    labels = torch.tensor([[[0, 2], [0, 0]]])
    original = labels.clone()
    old_logits = torch.tensor([[[[20., 20.], [-20., 20.]]]])
    targets = merged_targets(labels, old_logits, 1, 2, 0.5)
    assert torch.equal(labels, original)
    assert targets[0, 0, 0, 0] == 1
    assert targets[0, 0, 0, 1] == 0
    assert targets[0, 1, 0, 1] == 1
    assert targets[0, 0, 1, 0] == 0


def test_clip_future_heads_unused_task_agnostic_and_class_mapping(factory, batch):
    x, y = batch
    model = factory('clip_cis', classes=8)
    ds = dataset_for(x, y, 'class-il', [3, 2, 2])
    model.begin_task(ds)
    future = copy.deepcopy(model.net.controllers[3].state_dict())
    assert math.isfinite(model.observe(x, y, x))
    for key, value in future.items():
        assert torch.equal(value, model.net.controllers[3].state_dict()[key])
    assert all(p.grad is None for p in model.net.controllers[3].parameters())
    model.net.eval()
    torch.testing.assert_close(model(x, 0), model(x, 2))
    assert (model(x)[:, 4:] == -1e4).all()
    assert (model(x).argmax(1) <= 3).all()
    with pytest.raises(ValueError, match='Unseen'):
        model.observe(x, y * 7, x)


def test_clip_rejects_ambiguous_embedding_rows(tmp_path):
    path = tmp_path / 'bad.pt'
    torch.save({'embeddings': torch.randn(2, 512), 'class_ids': [2, 1],
                'class_names': ['one', 'two']}, path)
    with pytest.raises(ValueError, match='global foreground label IDs'):
        read_embeddings(path, 2)


def test_dynamic_heads_equal_direct_per_sample_convolution(factory, batch):
    from models.utils.medical import resunet_features
    model = factory('clip_cis', classes=3)
    x, y = batch
    model.begin_task(dataset_for(x, y, 'class-il', [2]))
    net = model.net.eval()
    actual = net.foreground(x)
    features, bridge = resunet_features(net.backbone, x)
    feature = net.precls(features)
    context = net.gap(bridge)
    expected = torch.empty_like(actual)
    for b in range(2):
        for c in range(2):
            cond = torch.cat([context[b:b+1], net.organ_embeddings[c].reshape(1, 512, 1, 1)], 1)
            chunks = net.controllers[c](cond).flatten().split(net.weight_sizes + net.bias_sizes)
            result = feature[b:b+1]
            for layer in range(3):
                cout = 8 if layer < 2 else 1
                result = F.conv2d(result, chunks[layer].reshape(cout, 8, 1, 1), chunks[layer+3])
                if layer < 2:
                    result = F.relu(result)
            expected[b, c] = result[0, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_registry_and_parsers_do_not_import_optional_quadprog():
    assert {'s3r', 'ted', 'clip_cis'} <= set(get_all_models())
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from models.ted import get_parser; get_parser().parse_args([]); assert 'quadprog' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], cwd=root, check=True)


@pytest.mark.parametrize('name', ['s3r', 'ted', 'clip_cis'])
def test_standard_resunet32_at_256_forward_and_backward(name, factory):
    x = torch.randn(1, 1, 256, 256)
    y = torch.zeros(1, 256, 256, dtype=torch.long)
    y[:, 64:192, 64:192] = 1
    model = factory(name, classes=8 if name == 'clip_cis' else 2, baseline_size='mid')
    ds = dataset_for(x, y, 'class-il' if name == 'clip_cis' else 'domain-il',
                     [3, 2, 2] if name == 'clip_cis' else 2)
    model.begin_task(ds)
    assert math.isfinite(model.observe(x, y, x))
    model.net.eval()
    assert model(x).shape == (1, 8 if name == 'clip_cis' else 2, 256, 256)


@pytest.mark.parametrize('name', ['s3r', 'ted', 'clip_cis'])
def test_neutral_launcher_config_templates(name):
    from run import read_configuration
    root = Path(__file__).resolve().parents[1]
    args = read_configuration(str(root / 'configs' / (name + '.json')))
    assert args[args.index('--model') + 1] == name
