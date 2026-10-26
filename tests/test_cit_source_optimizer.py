import json
import pytest
import torch
from conftest import dataset_for


@pytest.mark.parametrize('name',['cit','cit_organ'])
def test_source_adamw_and_future_query_decay_protection(name,factory,batch):
    from test_organ_methods import stream_metadata
    model=factory(name,classes=8)
    assert isinstance(model.opt,torch.optim.AdamW)
    assert [g['lr_scale'] for g in model.opt.param_groups]==[1.,10.]
    assert [g['lr'] for g in model.opt.param_groups]==[2e-5,2e-4]
    assert all(g['weight_decay']==.01 for g in model.opt.param_groups)
    x,y=batch
    ds=stream_metadata() if name.endswith('organ') else dataset_for(x,y,'class-il',[1]*7)
    model.begin_task(ds)
    future=model.net.queries.weight[1:].detach().clone()
    for _ in range(2):model.observe(x,y,x)
    assert torch.equal(future,model.net.queries.weight[1:])
    state=model.opt.state[model.net.queries.weight]
    assert not state['exp_avg'][1:].count_nonzero()
    assert not state['exp_avg_sq'][1:].count_nonzero()


def test_cit_protocol_preserves_budget_and_phase(tmp_path,monkeypatch):
    from run import read_configuration
    from models.cit_organ import get_parser
    monkeypatch.setenv('SESSION_ROOT',str(tmp_path))
    p=tmp_path/'submitted.json';p.write_text(json.dumps(dict(model='cit_organ',dataset='seq-taskincre',
        name='cit_organ_s1',seed=1,n_epochs=150,batch_size=8,lr=.008,test_only=True)))
    folder=tmp_path/'protocols';folder.mkdir();q=folder/'cit_organ_s1.json'
    q.write_text(json.dumps(dict(model='cit_organ',protocol_id='cit-source-optimizer-v2',
        overrides=dict(lr=2e-5,cit_optimizer='AdamW',cit_weight_decay=.01,cit_head_lr_mult=10.))))
    args=get_parser().parse_args(read_configuration(p))
    assert args.lr==2e-5 and args.cit_optimizer=='AdamW'
    assert args.n_epochs==150 and args.batch_size==8 and args.test_only and args.seed==1


def test_cit_rejects_legacy_sgd_as_adamw_checkpoint(factory,tmp_path):
    model=factory('cit',classes=8);model.save_network(model.net,str(tmp_path),0,'latest')
    p=tmp_path/'T0_latest.pth';c=torch.load(p,weights_only=True);del c['config']['cit_optimizer'];torch.save(c,p)
    with pytest.raises(ValueError,match='cit_optimizer'):model.load_network(p,model.net)
