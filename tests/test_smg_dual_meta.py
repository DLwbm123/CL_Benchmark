import pytest
import torch
from models.utils.smg_dual_meta import first_order_dual_meta


def test_first_order_meta_grad_and_no_live_state_changes():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    batch = (torch.ones(1, 1), torch.zeros(1, 1))
    before = model.weight.detach().clone()
    loss, gradients = first_order_dual_meta(
        model, torch.nn.MSELoss(), batch, batch, batch, batch, gamma=.1, beta=.2)
    # Norm-one inner gradients give adapted weights .9 and .8.
    assert gradients['weight'].item() == pytest.approx(7.4)
    assert loss.item() == pytest.approx(3.45)
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None


def test_rejects_invalid_inner_rate():
    model = torch.nn.Linear(1, 1)
    batch = (torch.ones(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match='positive'):
        first_order_dual_meta(model, torch.nn.MSELoss(), batch, batch, batch, batch, gamma=0)
