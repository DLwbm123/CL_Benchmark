import pytest
import torch
from torch.nn import functional as F

from conftest import dataset_for
from models.utils.medical import binary_targets, softmax_binary_cross_entropy


@pytest.mark.parametrize('classes', [2, 4])
def test_stable_softmax_bce_matches_probability_bce_and_gradients(classes):
    z = torch.randn(2, classes, 4, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(classes, (2, 4, 5))
    old = F.binary_cross_entropy(z.softmax(1), binary_targets(y, classes).to(z))
    stable = softmax_binary_cross_entropy(z, y)
    assert torch.allclose(old, stable, atol=1e-12)
    assert torch.allclose(torch.autograd.grad(old, z, retain_graph=True)[0],
                          torch.autograd.grad(stable, z)[0], atol=1e-12)


def test_wrong_saturated_background_keeps_corrective_gradient():
    z = torch.tensor([[[[10000.]], [[-10000.]]]], requires_grad=True)
    y = torch.ones(1, 1, 1, dtype=torch.long)
    old = F.binary_cross_entropy(z.softmax(1), binary_targets(y, 2))
    old_gradient = torch.autograd.grad(old, z, retain_graph=True)[0]
    stable = softmax_binary_cross_entropy(z, y)
    gradient = torch.autograd.grad(stable, z)[0]
    assert old_gradient.count_nonzero() == 0
    assert torch.isfinite(stable) and torch.isfinite(gradient).all()
    assert gradient[0, 0, 0, 0] > 0 and gradient[0, 1, 0, 0] < 0


def test_ted_source_optimizer_resets_per_domain_and_decays_after_epoch(factory, batch):
    model = factory('ted')
    x, y = batch
    assert isinstance(model.opt, torch.optim.Adam)
    assert model.opt.param_groups[0]['weight_decay'] == 0
    ds = dataset_for(x, y)
    model.begin_task(ds)
    model.begin_epoch(0, 150)
    assert model.opt.param_groups[0]['lr'] == pytest.approx(2e-4)
    model.observe(x, y, x)
    assert model.opt.state
    model.begin_epoch(1, 150)
    assert model.opt.param_groups[0]['lr'] == pytest.approx(2e-4*.99)
    model.end_task(ds)
    model.begin_task(ds)
    assert not model.opt.state
    model.begin_epoch(0, 150)
    assert model.opt.param_groups[0]['lr'] == pytest.approx(1e-4)


@pytest.mark.parametrize('name', ['ted', 's3r'])
def test_legacy_probability_checkpoint_cannot_be_silently_resumed(factory, tmp_path, name):
    model = factory(name)
    model.save_network(model.net, str(tmp_path), 0, 'latest')
    p = tmp_path/'T0_latest.pth'
    state = torch.load(p, weights_only=True)
    del state['config'][name+'_bce_impl']
    torch.save(state, p)
    with pytest.raises(ValueError, match='bce_impl'):
        model.load_network(p, model.net)
