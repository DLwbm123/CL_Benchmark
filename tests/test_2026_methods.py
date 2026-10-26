"""Synthetic, independently checked formulas and continual-state transitions."""
import copy
import math
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from conftest import dataset_for
from models.rebac import binary_segmentation_loss, rebac_losses
from models.utils.rebac_head import fuse_background
from models.ldka import (progressive_margin_loss, smooth_knowledge_losses,
                         competition_counts, misclassification_decoupling)
from models.pcdd import (local_prototypes, merge_student_probabilities,
                         calibrated_distillation, dual_prototype_alignment)


@pytest.mark.parametrize('name', ['rebac', 'ldka', 'pcdd'])
def test_three_task_lifecycle_and_midtask_checkpoint(name, factory, batch, tmp_path):
    x, y = batch
    model = factory(name, classes=4)
    for task in range(3):
        ds = dataset_for(x, y*(task+1), 'class-il', [1, 1, 1])
        model.begin_task(ds)
        teacher = copy.deepcopy(model.teacher.state_dict()) if model.teacher is not None else None
        model.train()
        for _ in range(2):
            assert math.isfinite(model.observe(x, y*(task+1), x))
        if teacher is not None:
            assert not model.teacher.training
            assert all(not p.requires_grad and p.grad is None for p in model.teacher.parameters())
            for key, value in teacher.items():
                torch.testing.assert_close(value, model.teacher.state_dict()[key], rtol=0, atol=0)
        if name == 'ldka':
            assert model.epoch_confusion.sum() > 0
            model.end_epoch()
            assert (model.competition.sum(1) <= 1.000001).all()
        model.net.eval()
        expected = model(x)
        assert torch.isfinite(expected).all()
        deployed = model.evaluation_parameter_count()
        assert expected.shape == (2, 4, 32, 32)
        torch.testing.assert_close(expected, model(x, 99), rtol=0, atol=0)
        if task < 2:
            assert (expected[:, task+2:] == -1e4).all()
        model.save_network(model.net, str(tmp_path), task, 'mid')
        restored = factory(name, classes=4)
        restored.load_network(str(tmp_path / f'T{task}_mid.pth'), restored.net)
        restored.net.eval()
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        assert restored.task_open and restored.current_task == task
        if name == 'pcdd':
            torch.testing.assert_close(restored.prototype_sum, model.prototype_sum, rtol=0, atol=0)
            assert torch.equal(restored.prototype_count, model.prototype_count)
        if name == 'ldka':
            assert restored.progress == model.progress
            torch.testing.assert_close(restored.competition, model.competition, rtol=0, atol=0)
        restored.net.train()
        model.net.train()
        continuation = model.observe(x, y*(task+1), x)
        assert restored.observe(x, y*(task+1), x) == pytest.approx(continuation, rel=1e-6, abs=1e-6)
        restored.end_task(ds)
        assert restored.evaluation_parameter_count() == deployed
        model = restored
    assert model.current_task == 3 and not model.task_open


def test_pcdd_rejects_missing_new_prototypes_at_task_end(factory, batch):
    x, y = batch
    model = factory('pcdd', classes=3)
    ds = dataset_for(x, y, 'class-il', [2])
    model.begin_task(ds)
    model.observe(x, y, x)
    with pytest.raises(ValueError, match='missing new-class prototypes'):
        model.end_task(ds)
    assert model.task_open and model.current_task == 0


def test_pcdd_batchnorm_adapter_persists_after_net_train(factory, batch):
    x, y = batch
    model = factory('pcdd', classes=3)
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    model.begin_task(ds)
    norms = [m for m in model.net.modules() if isinstance(m, nn.BatchNorm2d)]
    stats = [m.running_mean.clone() for m in norms]
    model.net.train()  # the legacy trainer toggles net directly
    assert all(not m.training and m.weight.requires_grad for m in norms)
    model.observe(x, y*2, x)
    for before, m in zip(stats, norms):
        torch.testing.assert_close(before, m.running_mean, rtol=0, atol=0)


def test_rebac_background_train_eval_have_distinct_literal_rules():
    base = torch.tensor([[[[2., 2., 2.]]]], requires_grad=True)
    residuals = [torch.tensor([[[[-3., -4., -5.]]]], requires_grad=True),
                 torch.tensor([[[[-6., 7., -8.]]]], requires_grad=True)]
    groups = [torch.tensor([[[[9., 1., 1.]]]]), torch.tensor([[[[1., 9., 1.]]]]), torch.tensor([[[[1., 1., 9.]]]])]
    train = fuse_background(base, residuals, groups, True)
    evaluation = fuse_background(base, residuals, groups, False)
    torch.testing.assert_close(train, torch.tensor([[[[-4., 9., -6.]]]]))
    torch.testing.assert_close(evaluation, torch.tensor([[[[2., -2., -6.]]]]))
    train.sum().backward()
    assert base.grad is None and residuals[0].grad is None
    assert torch.equal(residuals[1].grad, torch.ones_like(base))


@pytest.mark.parametrize('label', [0, 2, 255])
def test_rebac_loss_known_values_and_empty_regions(label):
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    group0 = torch.zeros(1, 2, 2, 2, requires_grad=True)
    group1 = torch.zeros(1, 2, 2, 2, requires_grad=True)
    state = {'groups': [group0, group1], 'responses': [torch.ones(1, 2, 2, 2, requires_grad=True)]}
    teacher = (torch.zeros(1, 2, 2, 2), {'groups':[torch.zeros(1, 2, 2, 2)], 'responses':[torch.zeros(1, 2, 2, 2)]})
    labels = torch.full((1, 2, 2), label, dtype=torch.long)
    values = rebac_losses(logits, state, teacher, labels, 2)
    expected = ([3*math.log(2), 0., 1., 2*math.log(2), 0.] if label == 0 else
                [3*math.log(2), 0., 0., 2*math.log(2), math.log(2)] if label == 2 else [0.]*5)
    for value, target in zip(values, expected):
        assert value.item() == pytest.approx(target, abs=1e-6)
    sum(values).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 1].abs().sum() == 0  # old foreground supervision detached


def test_rebac_residual_adapters_are_optimized_and_bn_is_frozen(factory, batch):
    x, y = batch
    model = factory('rebac', classes=3)
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    model.begin_task(ds)
    bn = [m for m in model.net.modules() if isinstance(m, nn.BatchNorm2d)]
    before = [m.running_mean.clone() for m in bn]
    adapter = model.net.adapters[0]
    weights = adapter.weight.detach().clone()
    assert all(not m.training and not m.weight.requires_grad for m in bn)
    optimized = {id(p) for group in model.opt.param_groups for p in group['params']}
    assert all(id(p) in optimized for p in adapter.parameters())
    assert math.isfinite(model.observe(x, y*2, x))
    assert not torch.equal(weights, adapter.weight)
    for old, m in zip(before, bn):
        torch.testing.assert_close(old, m.running_mean, rtol=0, atol=0)


@pytest.mark.parametrize('rho', [.1, .5, .9])
def test_ldka_pml_matches_independent_scalar_ecdf(rho):
    logits = torch.tensor([[[[.2, 1., 1., 2.]], [[0., 0., 0., 0.]]]], requires_grad=True)
    labels = torch.tensor([[[0, 0, 0, 255]]])
    result, weight = progressive_margin_loss(logits, labels, rho)
    values = [.2, 1., 1.]
    robust = []
    for m in values:
        p = 1/(1+math.exp(-m))
        robust.append(m/(1-(2*p-1)**2))
    cdf = [sum(v <= r for v in robust)/len(robust) for r in robust]
    raw = [u**math.log((1-rho)/rho) for u in cdf]
    expected_weights = [v/(sum(raw)/3) for v in raw]
    expected = sum(w*math.log1p(math.exp(-m)) for w, m in zip(expected_weights, values))/3
    assert result.item() == pytest.approx(expected, rel=1e-6)
    torch.testing.assert_close(weight, torch.tensor(expected_weights))
    result.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[:, :, :, -1].abs().sum() == 0


def test_ldka_skd_double_backprop_and_nonlinear_perturbation_reference():
    class Square(nn.Module):
        def forward(self, x):
            return x.square()
    feature = torch.tensor([[[[2.]]]], requires_grad=True)
    dss, perturb = smooth_knowledge_losses(feature, torch.ones_like(feature), Square())
    # z=4, teacher=1, grad_z D=6. grad_f D=24 -> unit perturb=+1.
    assert dss.item() == 36 and perturb.item() == 25
    dss.backward(retain_graph=True)
    assert feature.grad.item() == 96  # graph must survive the first derivative
    feature.grad.zero_()
    perturb.backward()
    assert feature.grad.item() == 20
    equal = torch.ones(1, 1, 1, 1, requires_grad=True)
    dss, perturb = smooth_knowledge_losses(equal, torch.ones_like(equal), Square())
    assert dss == 0 and perturb == 0


def test_ldka_competition_and_mad_independent_reference():
    logits = torch.tensor([[[[2., 0., 2.]], [[0., 2., 0.]]]])
    labels = torch.tensor([[[0, 0, 255]]])
    assert torch.equal(competition_counts(logits, labels), torch.tensor([[1, 1], [0, 0]]))
    axes = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    competition = torch.tensor([[.5, .5], [.25, .75]])
    loss = misclassification_decoupling(axes, competition)
    assert loss.item() == .75*11**2
    loss.backward()
    assert axes.grad.abs().sum() > 0


def test_pcdd_local_means_ignore_and_probability_mass():
    feature = torch.tensor([[[[1., 3., 9., 999.]], [[2., 6., 8., 999.]]]], requires_grad=True)
    labels = torch.tensor([[[1, 1, 2, 255]]])
    p, counts = local_prototypes(feature, labels, 4)
    torch.testing.assert_close(p, torch.tensor([[0., 0.], [2., 4.], [9., 8.], [0., 0.]]))
    assert torch.equal(counts, torch.tensor([0, 2, 1, 0]))
    p.sum().backward()
    assert feature.grad[:, :, :, -1].abs().sum() == 0
    logits = torch.tensor([[[[0.]], [[math.log(2.)]], [[math.log(3.)]], [[math.log(4.)]]]])
    probability = merge_student_probabilities(logits, 2).exp()
    torch.testing.assert_close(probability, torch.tensor([[[[.8]], [[.2]]]]))
    torch.testing.assert_close(probability.sum(1), torch.ones(1, 1, 1))


def test_pcdd_calibration_uses_forward_kl_and_common_teacher_space():
    student = torch.tensor([[[[math.log(.2), math.log(.2)]], [[math.log(.3), math.log(.3)]], [[math.log(.5), math.log(.5)]]]], requires_grad=True)
    teacher = torch.tensor([[[[math.log(.6), math.log(.6)]], [[math.log(.4), math.log(.4)]]]])
    features = torch.ones(1, 2, 1, 2)
    prototypes = torch.ones(3, 2)
    sizes = torch.ones(3, dtype=torch.long)
    pseudo = torch.tensor([[[1, 2]]])
    old, current = calibrated_distillation(student, teacher, features, features, pseudo,
                                          prototypes, sizes, prototypes, sizes, 2)
    k0, k1 = .7*math.log(.7/.6), .3*math.log(.3/.4)
    assert old.item() == pytest.approx(k1, abs=1e-6)
    assert current.item() == pytest.approx((2*k0+k1)/3, abs=1e-6)
    (old+current).backward()
    assert torch.isfinite(student.grad).all()


def test_pcdd_dual_background_is_sum_of_class_means_not_pooling():
    feature = torch.tensor([[[[2., 6.]]]], requires_grad=True)
    labels = torch.tensor([[[0, 2]]])
    # Teacher background mean=4; student q_bg=mean(bg)+mean(new)=8.
    local, global_loss = dual_prototype_alignment(feature, feature.detach(), labels, 2, 3,
                                                 torch.tensor([[4.], [0.], [0.]]), torch.tensor([2, 0, 0]))
    assert local.item() == 8 and global_loss.item() == 8
    (local+global_loss).backward()
    torch.testing.assert_close(feature.grad, torch.full_like(feature, 8.))


def test_pcdd_pixel_weighted_running_bank_and_background_reset(factory, monkeypatch):
    model = factory('pcdd', classes=3, lr=0.)
    width = model.net.classifier.in_channels
    monkeypatch.setattr(model.net, 'features', lambda x: x.repeat(1, width, 1, 1))
    ds = dataset_for(torch.ones(1, 1, 1, 3), torch.ones(1, 1, 3, dtype=torch.long), 'class-il', [1, 1])
    model.begin_task(ds)
    x = torch.tensor([[[[1., 3., 9.]]]])
    y = torch.tensor([[[1, 1, 255]]])
    model.observe(x, y, x)
    model.observe(torch.tensor([[[[8., 999., 999.]]]]), torch.tensor([[[1, 255, 255]]]), x)
    assert model.prototype_count[1] == 3
    torch.testing.assert_close(model.prototype_sum[1]/3, torch.full((width,), 4.))
    model.end_task(ds)
    model.begin_task(ds)
    assert model.prototype_count[0] == 0 and model.teacher_bg_count == 0
    torch.testing.assert_close(model.previous_prototypes[1], torch.full((width,), 4.))


@pytest.mark.parametrize('name', ['rebac', 'ldka', 'pcdd'])
@pytest.mark.parametrize('label', [0, 255])
def test_empty_foreground_and_all_ignored_are_finite(name, label, factory, batch):
    x, y = batch
    model = factory(name, classes=3)
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    model.observe(x, y, x)
    model.end_task(ds)
    model.begin_task(ds)
    value = model.observe(x, torch.full_like(y, label), x)
    assert math.isfinite(value)
    if label == 255:
        assert value == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.net.parameters())


@pytest.mark.parametrize('name,key', [('rebac','rebac_group_weight'), ('ldka','ldka_dss_weight'), ('pcdd','pcdd_global_weight')])
def test_2026_config_validation_and_wrong_checkpoint_objective(name, key, factory, batch, tmp_path):
    from run import read_configuration
    root = Path(__file__).resolve().parents[1]
    assert '--model' in read_configuration(str(root/'configs'/f'{name}.json'))
    x, y = batch
    model = factory(name, classes=3)
    with pytest.raises(ValueError, match='supports'):
        model.begin_task(dataset_for(x, y, 'domain-il'))
    with pytest.raises(RuntimeError, match='begin_task'):
        model.observe(x, y, x)
    model.begin_task(dataset_for(x, y, 'class-il', [1, 1]))
    with pytest.raises(ValueError, match='future'):
        model.observe(x, y*2, x)
    model.save_network(model.net, str(tmp_path), 0, 'mid')
    mismatch = factory(name, classes=3, **{key: .123})
    with pytest.raises(ValueError, match='configuration mismatch'):
        mismatch.load_network(str(tmp_path/'T0_mid.pth'), mismatch.net)


@pytest.mark.parametrize('name', ['rebac', 'ldka', 'pcdd'])
def test_2026_standard_resolution_and_future_parameters(name, factory):
    model = factory(name, classes=3, baseline_size='mid')
    x = torch.randn(1, 1, 256, 256)
    y = torch.zeros(1, 256, 256, dtype=torch.long)
    y[:, 64:192, 64:192] = 1
    ds = dataset_for(x, y, 'class-il', [1, 1])
    model.begin_task(ds)
    future = (model.net.heads[1].weight if name == 'rebac' else model.net.classifier.weight[2:]).detach().clone()
    assert math.isfinite(model.observe(x, y, x))
    current = model.net.heads[1].weight if name == 'rebac' else model.net.classifier.weight[2:]
    torch.testing.assert_close(future, current, rtol=0, atol=0)
    model.net.eval()
    assert model(x).shape == (1, 3, 256, 256)
    assert (model(x)[:, 2:] == -1e4).all()
    model.end_task(ds)
    model.begin_task(ds)
    assert math.isfinite(model.observe(x, y*2, x))
