"""CIT-2025 tests: independent queries and original-task accumulated teachers."""
import copy
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from conftest import dataset_for
from models.cit import supervised_cit_losses, accumulated_cit_losses


def test_cit_query_independence_and_attention_reference(factory, batch):
    x, y = batch
    model = factory('cit', classes=4, cit_dropout=0.)
    ds = dataset_for(x, y, 'class-il', [1, 2])
    model.begin_task(ds)
    net = model.net.eval()
    reference = net.components(x, count=1)
    expanded = net.components(x, count=3)
    for key in reference:
        torch.testing.assert_close(reference[key], expanded[key][:, :1])
    # Perturb a different query: no query self-attention or cross-class softmax.
    with torch.no_grad():
        net.queries.weight[2].mul_(100.)
    changed = net.components(x, count=3)
    for key in reference:
        torch.testing.assert_close(reference[key], changed[key][:, :1])

    features = F.avg_pool2d(net.backbone.features(x), net.stride, net.stride)
    b, channels, h, w = features.shape
    memory = features.flatten(2).transpose(1, 2)
    query = net.queries.weight[:1][None].expand(b, -1, -1)
    q = query.reshape(b, 1, net.heads, channels // net.heads).transpose(1, 2)
    k = net.key(memory).reshape(b, -1, net.heads, channels // net.heads).transpose(1, 2)
    v = memory.reshape(b, -1, net.heads, channels // net.heads).transpose(1, 2)
    decoded = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    decoded = decoded.transpose(1, 2).reshape(b, 1, channels)
    expected_presence = net.presence(net.norm(query + decoded))
    torch.testing.assert_close(reference['presence'], expected_presence)
    raw_masks = ((q @ k.transpose(-2, -1)) / math.sqrt(channels // net.heads)).mean(1)
    torch.testing.assert_close(reference['masks'], raw_masks.reshape(b, 1, h, w))


def test_cit_accumulates_direct_task_sources_and_roundtrips_teacher_bank(factory, batch, tmp_path):
    x, y = batch
    model = factory('cit', classes=4)
    for task in range(2):
        ds = dataset_for(x, y * (task + 1), 'class-il', [1, 1, 1])
        model.begin_task(ds)
        assert math.isfinite(model.observe(x, y * (task + 1), x))
        model.end_task(ds)
    assert len(model.teachers) == 2 and model.boundaries == [1, 2]
    target = model.accumulated_targets(x)
    # The second checkpoint's *old* query is not a source for class 1.
    with torch.no_grad():
        model.teachers[1].queries.weight[0].mul_(10.)
    unchanged = model.accumulated_targets(x)
    for key in target:
        torch.testing.assert_close(target[key], unchanged[key], rtol=0, atol=0)
    with torch.no_grad():
        model.teachers[0].queries.weight[0].mul_(10.)
    changed = model.accumulated_targets(x)
    assert not torch.allclose(changed['masks'][:, 0], target['masks'][:, 0])
    torch.testing.assert_close(changed['masks'][:, 1], target['masks'][:, 1], rtol=0, atol=0)

    model.begin_task(dataset_for(x, y * 3, 'class-il', [1, 1, 1]))
    frozen = [copy.deepcopy(t.state_dict()) for t in model.teachers]
    model.train()  # Parent train() must not turn historical dropout/BN back on.
    assert all(not t.training for t in model.teachers)
    assert math.isfinite(model.observe(x, y * 3, x))
    for before, teacher in zip(frozen, model.teachers):
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        for key, value in before.items():
            torch.testing.assert_close(value, teacher.state_dict()[key], rtol=0, atol=0)
    model.net.eval()
    output = model(x)
    targets = model.accumulated_targets(x)
    model.save_network(model.net, str(tmp_path), 2, 'mid')
    restored = factory('cit', classes=4)
    restored.load_network(str(tmp_path / 'T2_mid.pth'), restored.net)
    restored.net.eval()
    assert len(restored.teachers) == 2 and restored.boundaries == [1, 2]
    assert restored.current_task == 2 and restored.task_open
    torch.testing.assert_close(output, restored(x), rtol=0, atol=0)
    for key in targets:
        torch.testing.assert_close(targets[key], restored.accumulated_targets(x)[key], rtol=0, atol=0)
    assert math.isfinite(restored.observe(x, y * 3, x))


def test_cit_known_binary_distillation_and_background_mapping():
    outputs = {'presence': torch.full((1, 2, 2), 2., requires_grad=True),
               'masks': torch.ones(1, 2, 2, 2, requires_grad=True)}
    teacher = {'presence': torch.ones(1, 1, 2), 'masks': torch.zeros(1, 1, 2, 2)}
    labels = torch.tensor([[[0, 2], [255, 0]]])
    cls_kd, mask_kd = accumulated_cit_losses(outputs, teacher, labels, 1)
    expected_cls = F.softplus(torch.tensor(2.)) - 2 * torch.sigmoid(torch.tensor(1.))
    # 2 background pixels retain 0.5; the new GT pixel is set to 0; ignore is excluded.
    expected_mask = F.softplus(torch.tensor(1.)) - 1 / 3
    torch.testing.assert_close(cls_kd, expected_cls)
    torch.testing.assert_close(mask_kd, expected_mask)
    assert teacher['masks'].abs().sum() == 0
    (cls_kd + mask_kd).backward()
    assert torch.isfinite(outputs['masks'].grad).all()


@pytest.mark.parametrize('label', [0, 255])
def test_cit_absent_or_ignored_supervision(label):
    outputs = {'presence': torch.randn(1, 2, 2, requires_grad=True),
               'masks': torch.randn(1, 2, 2, 2, requires_grad=True)}
    labels = torch.full((1, 8, 8), label, dtype=torch.long)
    classification, focal, dice = supervised_cit_losses(outputs, labels, 0, 2)
    total = classification + focal + dice
    assert torch.isfinite(total) and focal == 0 and dice == 0
    if label == 255:
        assert total == 0
    total.backward()
    assert torch.isfinite(outputs['presence'].grad).all()


def test_cit_future_queries_isolated_and_256_input(factory):
    model = factory('cit', classes=8, baseline_size='mid')
    x = torch.randn(1, 1, 256, 256)
    y = torch.zeros(1, 256, 256, dtype=torch.long)
    y[:, 64:192, 64:192] = 1
    with pytest.raises(ValueError, match='supports'):
        model.begin_task(dataset_for(x, y, 'domain-il'))
    model.begin_task(dataset_for(x, y, 'class-il', [3, 2, 2]))
    future = model.net.queries.weight[3:].detach().clone()
    deployed = model.evaluation_parameter_count()
    assert math.isfinite(model.observe(x, y, x))
    torch.testing.assert_close(future, model.net.queries.weight[3:], rtol=0, atol=0)
    assert model.net.queries.weight.grad[3:].abs().sum() == 0
    model.net.eval()
    assert model(x).shape == (1, 8, 256, 256)
    torch.testing.assert_close(model(x, 0), model(x, 5), rtol=0, atol=0)
    assert (model(x)[:, 4:] == -1e4).all()
    model.end_task(dataset_for(x, y, 'class-il', [3, 2, 2]))
    assert len(model.teachers) == 1 and model.evaluation_parameter_count() == deployed


def test_cit_config_and_checkpoint_option_validation(factory, batch, tmp_path):
    from run import read_configuration
    root = Path(__file__).resolve().parents[1]
    assert '--model' in read_configuration(str(root / 'configs/cit.json'))
    model = factory('cit', classes=3)
    x, y = batch
    model.begin_task(dataset_for(x, y, 'class-il', [1, 1]))
    model.save_network(model.net, str(tmp_path), 0, 'latest')
    different = factory('cit', classes=3, cit_feature_stride=4)
    with pytest.raises(ValueError, match='configuration mismatch'):
        different.load_network(str(tmp_path / 'T0_latest.pth'), different.net)
    with pytest.raises(ValueError, match='future'):
        model.observe(x, y * 2, x)
