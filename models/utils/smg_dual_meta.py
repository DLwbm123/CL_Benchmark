"""Paper-derived first-order dual-meta objective, not a complete SMG implementation.

Replay generation, site prompts, sampling and the training lifecycle are not
implemented here. The caller must supply labeled current, replay and split
batches, and must not use held-out data for any of them.
"""
import torch
from torch.func import functional_call


def first_order_dual_meta(model, loss_fn, current, replay, meta_train, meta_test,
                          gamma=5e-5, beta=5e-4):
    """Return first-order parameter gradients for the two paper objectives.

    Each batch is an (inputs, labels) pair. Inner gradients are detached so the
    derivative of each adapted parameter w.r.t. its original is identity.
    Buffers are copied for each forward to avoid changing the live model.
    """
    if gamma <= 0 or beta <= 0:
        raise ValueError('Inner learning rates must be positive.')
    params = dict(model.named_parameters())
    if not params or not all(p.requires_grad for p in params.values()):
        raise ValueError('Model parameters must all require gradients.')
    buffers = dict(model.named_buffers())

    def evaluate(p, batch):
        x, y = batch
        b = {name: value.detach().clone() for name, value in buffers.items()}
        value = loss_fn(functional_call(model, (p, b), (x,)), y)
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError('Loss must be a finite scalar.')
        return value

    def objective(inner_batch, outer_batch, rate):
        inner = evaluate(params, inner_batch)
        inner_grad = torch.autograd.grad(inner, tuple(params.values()), retain_graph=True)
        adapted = {name: p - rate * g.detach()
                   for (name, p), g in zip(params.items(), inner_grad)}
        return inner + evaluate(adapted, outer_batch)

    total = objective(current, replay, gamma) + objective(meta_train, meta_test, beta)
    grads = torch.autograd.grad(total, tuple(params.values()))
    if not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('Non-finite dual-meta gradients.')
    return total.detach(), dict(zip(params, (g.detach() for g in grads)))
