import torch
from conftest import dataset_for


def test_rebac_guard_only_incremental_and_persists_counts(factory, batch):
    model = factory('rebac', classes=4)
    x, y = batch
    stream = dataset_for(x, y, 'class-il', [1, 1, 1])
    model.begin_task(stream)
    assert model.gradient_clip_max_norm == 0
    model.observe(x, y, x)
    model.end_task(stream)
    model.begin_task(stream)
    assert model.gradient_clip_max_norm == 1
    model.opt.zero_grad(set_to_none=True)
    model.optimization_step(model.net.heads[1].weight.square().sum()*1e6)
    norm = torch.stack([p.grad.square().sum() for p in model.net.parameters()
                        if p.grad is not None]).sum().sqrt()
    assert norm <= 1.00001
    assert model.guard_stats['steps'] == model.guard_stats['clipped'] == 1
    saved = model.knowledge_state()
    model.guard_stats = {}
    model.restore_knowledge(saved)
    assert model.guard_stats['clipped'] == 1
