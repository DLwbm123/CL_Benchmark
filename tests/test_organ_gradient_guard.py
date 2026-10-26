import json
from types import SimpleNamespace

import pytest
import torch


def test_incremental_guard_caps_update_without_changing_stage_rate(factory, batch):
    from models.utils.organ import ORGAN_NAMES
    model = factory('ldka_organ')
    ds = SimpleNamespace(SETTING='task-il',N_CLASSES_PER_TASK=2,N_TASKS=4,TASK_NAMES=ORGAN_NAMES)
    x,y = batch
    model.begin_task(ds)
    assert model.gradient_clip_max_norm == 0
    model.observe(x,y,x)
    model.end_task(ds)
    model.begin_task(ds)
    assert model.gradient_clip_max_norm == 1
    model.opt.zero_grad(set_to_none=True)
    loss = sum(p.square().sum() for p in model.net.parameters()) * 1e6
    model.optimization_step(loss)
    norm = torch.stack([p.grad.square().sum() for p in model.net.parameters() if p.grad is not None]).sum().sqrt()
    assert norm <= 1.00001
    assert model.last_losses['gradient_norm_before_clip'] > 1
    assert model.guard_stats['steps']==1 and model.guard_stats['clipped']==1
    assert model.opt.param_groups[0]['lr'] == model.args.ldka_incremental_lr


def test_versioned_guard_cannot_override_optimizer_or_identity(tmp_path, monkeypatch):
    from run import read_configuration
    from models.ldka_organ import get_parser
    monkeypatch.setenv('SESSION_ROOT',str(tmp_path))
    config=dict(model='ldka_organ',dataset='seq-taskincre',name='ldka_organ_s1',seed=1,
                lr=.001,n_epochs=150,batch_size=8)
    path=tmp_path/'submitted.json';path.write_text(json.dumps(config))
    folder=tmp_path/'protocols';folder.mkdir()
    protocol=dict(model='ldka_organ',protocol_id='organ-gradient-guard-v2',
                  overrides={'ldka_organ_grad_clip':1.})
    target=folder/'ldka_organ_s1.json';target.write_text(json.dumps(protocol))
    args=get_parser().parse_args(read_configuration(path))
    assert args.ldka_organ_grad_clip==1 and args.lr==.001 and args.seed==1
    protocol['overrides']['lr']=.00001;target.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='identity'):
        read_configuration(path)
