"""Natural-image method adaptations; tensors only, no data files or downloads."""
import copy
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from conftest import dataset_for
from models.lgkd import label_guided_targets, label_guided_distillation
from models.nest import cross_task_similarity, ClassifierTransform
from models.utils.class_incremental import unbiased_cross_entropy, unbiased_distillation

METHODS = ('ewf', 'lgkd', 'nest')


@pytest.mark.parametrize('name', METHODS)
def test_three_task_lifecycle_teacher_freeze_and_checkpoint(name, factory, batch, tmp_path):
    x, foreground = batch
    model = factory(name, classes=4)
    for task in range(3):
        labels = foreground * (task + 1)
        ds = dataset_for(x, labels, 'class-il', [1, 1, 1])
        teacher_before = (copy.deepcopy(model.teacher.state_dict())
                          if model.teacher is not None else None)
        model.begin_task(ds)
        assert int(model.net.seen_classes) == task + 2
        future_before = model.net.classifier.weight[task+2:].detach().clone()
        assert math.isfinite(model.observe(x, labels, x))
        assert math.isfinite(model.observe(x, labels, x))
        torch.testing.assert_close(model.net.classifier.weight[task+2:], future_before,
                                   rtol=0, atol=0)
        if teacher_before is not None:
            assert not model.teacher.training
            assert all(not p.requires_grad and p.grad is None for p in model.teacher.parameters())
            for key, value in teacher_before.items():
                torch.testing.assert_close(value, model.teacher.state_dict()[key], rtol=0, atol=0)
        model.net.eval()
        assert model(x).shape == (2, 4, 32, 32)
        torch.testing.assert_close(model(x, 0), model(x, 10), rtol=0, atol=0)
        model.save_network(model.net, str(tmp_path), task, 'mid')
        restored = factory(name, classes=4)
        restored.load_network(str(tmp_path / 'T{}_mid.pth'.format(task)), restored.net)
        restored.net.eval()
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        assert restored.task_open and restored.current_task == task
        assert math.isfinite(restored.observe(x, labels, x))
        model.end_task(ds)
        assert model.current_task == task + 1
        assert int(model.teacher.seen_classes) == task + 2


def test_lgkd_mass_transport_and_independent_pixel_reference():
    old = torch.randn(2, 3, 4, 5, requires_grad=True)
    new = torch.randn(2, 5, 4, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 4, 255, 0, 4, 3, 0] * 4).reshape(2, 4, 5)
    original = labels.clone()
    expanded = label_guided_targets(old, labels, 5)
    torch.testing.assert_close(expanded.sum(1), torch.ones_like(labels, dtype=torch.float))
    teacher_probs = old.detach().softmax(1)
    reference = []
    for b in range(2):
        for h in range(4):
            for w in range(5):
                label = int(labels[b, h, w])
                if label == 255:
                    continue
                target = torch.zeros(5)
                target[:3] = teacher_probs[b, :, h, w]
                weight = 5.
                if label >= 3:
                    target[label] = target[0]
                    target[0] = 0
                    weight = 1.
                reference.append(-weight * (target * new[b, :, h, w].log_softmax(0)).sum())
    actual = label_guided_distillation(new, old, labels)
    torch.testing.assert_close(actual, torch.stack(reference).mean())
    actual.backward()
    assert old.grad is None and torch.isfinite(new.grad).all()
    assert torch.equal(labels, original)


@pytest.mark.parametrize('label', [0, 3, 255])
def test_lgkd_empty_previous_novel_or_valid_subset_is_finite(label):
    old = torch.randn(1, 3, 4, 4)
    new = torch.randn(1, 5, 4, 4, requires_grad=True)
    labels = torch.full((1, 4, 4), label, dtype=torch.long)
    loss = label_guided_distillation(new, old, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(new.grad).all()
    if label == 255:
        assert loss == 0 and new.grad.abs().sum() == 0


def test_shared_mib_losses_match_existing_reference_implementations():
    from models.utils.loss import UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss
    student = torch.randn(2, 5, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 3, (2, 4, 4))
    labels[labels == 1] = 3
    labels[labels == 2] = 4
    labels[0, 0, 0] = 255
    ce = UnbiasedCrossEntropy(old_cl=3, reduction='none', ignore_index=255)(student, labels).mean()
    kd = UnbiasedKnowledgeDistillationLoss(reduction='none', alpha=1.)(student, teacher).mean()
    torch.testing.assert_close(unbiased_cross_entropy(student, labels, 3), ce)
    torch.testing.assert_close(unbiased_distillation(student, teacher), kd)


@pytest.mark.parametrize('name', ['lgkd', 'ewf'])
def test_balanced_initialization_preserves_teacher_probability_mass(name, factory, batch):
    x, y = batch
    model = factory(name, classes=4)
    ds = dataset_for(x, y, 'class-il', [1, 2])
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    old_probs = model.teacher.logits(x).softmax(1)
    model.begin_task(dataset_for(x, y * 2, 'class-il', [1, 2]))
    model.net.eval()
    new_probs = model.net.logits(x).softmax(1)
    torch.testing.assert_close(new_probs[:, 1], old_probs[:, 1])
    torch.testing.assert_close(new_probs[:, [0, 2, 3]].sum(1), old_probs[:, 0])


def test_ewf_exact_endpoint_fusion_and_new_row_preservation(factory, batch):
    x, y = batch
    model = factory('ewf', classes=4)
    ds = dataset_for(x, y, 'class-il', [1, 1, 1])
    model.begin_task(ds)
    with torch.no_grad():
        for p in model.net.parameters():
            p.fill_(2.)
        model.net.backbone.bn1.running_mean.fill_(2.)
    model.end_task(ds)
    model.begin_task(dataset_for(x, y * 2, 'class-il', [1, 1, 1]))
    with torch.no_grad():
        for p in model.net.parameters():
            p.fill_(6.)
        model.net.backbone.bn1.running_mean.fill_(6.)
        model.net.backbone.bn1.num_batches_tracked.fill_(5)
    model.end_task(ds)
    alpha = math.sqrt(1 / 3)
    expected = 2 * (1 - alpha) + 6 * alpha
    torch.testing.assert_close(model.net.classifier.weight[:2],
                               torch.full_like(model.net.classifier.weight[:2], expected))
    assert (model.net.classifier.weight[2:] == 6).all()
    torch.testing.assert_close(model.net.backbone.bn1.running_mean,
                               torch.full_like(model.net.backbone.bn1.running_mean, expected))
    assert int(model.net.backbone.bn1.num_batches_tracked) == 5
    torch.testing.assert_close(model.teacher.classifier.weight, model.net.classifier.weight)
    with pytest.raises(RuntimeError):
        model.end_task(ds)


def test_ewf_old_classifier_frozen_but_features_receive_gradients(factory, batch):
    x, y = batch
    model = factory('ewf', classes=3)
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    model.end_task(ds)
    model.begin_task(dataset_for(x, y * 2, 'class-il', [1, 1]))
    old = model.net.classifier.weight[:2].detach().clone()
    model.observe(x, y * 2, x)
    torch.testing.assert_close(old, model.net.classifier.weight[:2], rtol=0, atol=0)
    assert model.net.backbone.conv1.weight.grad.abs().sum() > 0


def test_nest_similarity_matches_direct_mask_and_probability_formula():
    class ToyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Conv2d(2, 3, 1)
            self.classifier.weight.data.copy_(torch.tensor([[1., -1.], [-2., 1.], [1., 2.]])[:, :, None, None])
            self.classifier.bias.data.zero_()

        def features(self, x):
            return x

        def classify(self, x):
            return self.classifier(x)
    teacher = ToyTeacher().eval()
    x = torch.randn(1, 2, 4, 4)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    labels[:, :2] = 3
    labels[:, 2:] = 4
    ds = dataset_for(x, labels)
    actual = cross_task_similarity(teacher, ds.train_loader, 3, 2, torch.device('cpu'), size=4)
    probabilities = teacher.classify(x).softmax(1)[0].permute(1, 2, 0)
    features = x[0].permute(1, 2, 0)
    weights = teacher.classifier.weight.flatten(1)
    expected = torch.zeros_like(actual)
    for new_id in range(2):
        mask = labels[0] == new_id + 3
        for old in range(3):
            expected[new_id, old] = (((features * weights[old] > 0).float()
                                      * probabilities[:, :, old, None])[mask]).mean(0)
    torch.testing.assert_close(actual, expected)


def test_nest_pre_tuning_freezes_feature_extractor_and_compiles_no_extra_parameters(factory, batch):
    x, y = batch
    model = factory('nest', classes=3)
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    backbone = copy.deepcopy(model.net.backbone.state_dict())
    count = model.evaluation_parameter_count()
    model.begin_task(dataset_for(x, y * 2, 'class-il', [1, 1]))
    for key, value in backbone.items():
        torch.testing.assert_close(value, model.net.backbone.state_dict()[key], rtol=0, atol=0)
    assert model.initialization_info['pre_steps'] == 1
    assert model.initialization_info['pre_epochs'] == 1
    assert torch.isfinite(model.initialization_info['similarity']).all()
    assert model.evaluation_parameter_count() == count
    assert not any('importance' in n or 'projection' in n for n, _ in model.net.named_parameters())
    old_mean = model.teacher.classifier.weight[:model.old_count].flatten(1).norm(dim=1).mean()
    torch.testing.assert_close(model.net.classifier.weight[2:].flatten(1).norm(dim=1).mean(), old_mean)


def test_nest_missing_class_fails_without_committing_a_classifier(factory, batch):
    x, y = batch
    model = factory('nest', classes=4)
    first = dataset_for(x, y, 'class-il', [1, 2])
    model.begin_task(first)
    model.end_task(first)
    weights = model.net.classifier.weight.detach().clone()
    with pytest.raises(ValueError, match='missing IDs'):
        model.begin_task(dataset_for(x, y * 2, 'class-il', [1, 2]))
    assert not model.task_open and int(model.net.seen_classes) == 2
    torch.testing.assert_close(weights, model.net.classifier.weight, rtol=0, atol=0)


@pytest.mark.parametrize('name', METHODS)
def test_setting_validation_and_standard_256_input(name, factory):
    model = factory(name, classes=8, baseline_size='mid')
    x = torch.randn(1, 1, 256, 256)
    y = torch.zeros(1, 256, 256, dtype=torch.long)
    y[:, 64:192, 64:192] = 1
    with pytest.raises(ValueError, match='supports'):
        model.begin_task(dataset_for(x, y, 'domain-il'))
    model.begin_task(dataset_for(x, y, 'class-il', [3, 2, 2]))
    assert math.isfinite(model.observe(x, y, x))
    assert model(x).shape == (1, 8, 256, 256)
    with pytest.raises(ValueError, match='future'):
        model.observe(x, y * 7, x)


@pytest.mark.parametrize('name', METHODS)
def test_neutral_config_and_objective_mismatch(name, factory, batch, tmp_path):
    from run import read_configuration
    root = Path(__file__).resolve().parents[1]
    assert '--model' in read_configuration(str(root / 'configs' / (name + '.json')))
    model = factory(name, classes=3)
    x, y = batch
    model.begin_task(dataset_for(x, y, 'class-il', [1, 1]))
    model.save_network(model.net, str(tmp_path), 0, 'latest')
    overrides = {'ewf': {'ewf_kd_weight': 20.},
                 'lgkd': {'lgkd_previous_weight': 20.},
                 'nest': {'nest_pre_epochs': 2}}
    restored = factory(name, classes=3, **overrides[name])
    with pytest.raises(ValueError, match='configuration mismatch'):
        restored.load_network(str(tmp_path / 'T0_latest.pth'), restored.net)
