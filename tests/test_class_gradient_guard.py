import torch
from conftest import dataset_for


def test_class_guard_also_protects_base_task_and_future_rows(factory,batch):
    model=factory('ldka',classes=4)
    x,y=batch
    model.begin_task(dataset_for(x,y,'class-il',[1,1,1]))
    assert model.gradient_clip_max_norm==1
    future=model.net.classifier.weight[2:].detach().clone()
    model.opt.zero_grad(set_to_none=True)
    model.optimization_step(sum(p.square().sum() for p in model.net.parameters())*1e6)
    norm=torch.stack([p.grad.square().sum() for p in model.net.parameters() if p.grad is not None]).sum().sqrt()
    assert norm<=1.00001
    assert torch.equal(future,model.net.classifier.weight[2:])
    assert model.guard_stats['steps']==1 and model.guard_stats['clipped']==1
    assert model.opt.param_groups[0]['lr']==model.args.lr
