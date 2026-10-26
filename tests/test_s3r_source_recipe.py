import json

import pytest
import torch

from conftest import dataset_for


def test_source_adam_decay_and_three_task_importance(factory, batch):
    model = factory('s3r')
    assert isinstance(model.opt, torch.optim.Adam)
    assert model.opt.param_groups[0]['weight_decay'] == 1e-4
    x,y = batch
    for task in range(3):
        ds = dataset_for(x,y)
        model.begin_task(ds)
        model.begin_epoch(0, 150)
        assert model.opt.param_groups[0]['lr'] == pytest.approx(5e-4*.98)
        for _ in range(2):
            assert torch.isfinite(torch.tensor(model.observe(x,y,x)))
        model.begin_epoch(1, 150)
        assert model.opt.param_groups[0]['lr'] == pytest.approx(5e-4*.98**2)
        model.end_task(ds)
        assert all(torch.isfinite(v).all() for v in model.importance.values())
        assert sum(v.sum() for v in model.importance.values()) > 0
        assert model.current_task == task+1


def test_nan_logits_fail_before_probability_bce(factory, batch, monkeypatch):
    model = factory('s3r')
    x,y = batch
    model.begin_task(dataset_for(x,y))
    monkeypatch.setattr(model.net, 'auxiliary', lambda inputs: (
        torch.full((len(inputs),2,*inputs.shape[-2:]), float('nan')),
        torch.zeros(len(inputs),1,*inputs.shape[-2:])))
    with pytest.raises(FloatingPointError, match='logits before BCE'):
        model.observe(x,y,x)


@pytest.mark.parametrize('test_only', [False, True])
def test_live_scheduler_protocol_resolves_only_optimizer_fields(tmp_path, monkeypatch, test_only):
    from run import read_configuration
    from models.s3r import get_parser
    monkeypatch.setenv('SESSION_ROOT', str(tmp_path))
    config = dict(model='s3r', dataset='seq-prostate', name='s3r_s1', seed=1,
                  n_epochs=150, batch_size=8, lr=.008, lr_decay_rate=.5,
                  test_only=test_only)
    path = tmp_path/'submitted.json'
    path.write_text(json.dumps(config))
    folder = tmp_path/'protocols'
    folder.mkdir()
    protocol = dict(protocol_id='s3r-source-adam-v2', model='s3r',
                    overrides=dict(lr=.0005, lr_decay_rate=1., s3r_optimizer='Adam',
                                   s3r_weight_decay=.0001,s3r_epoch_gamma=.98))
    (folder/'s3r_s1.json').write_text(json.dumps(protocol))
    args = get_parser().parse_args(read_configuration(path))
    assert args.lr == .0005 and args.lr_decay_rate == 1
    assert args.seed == 1 and args.batch_size == 8 and args.n_epochs == 150
    assert args.test_only == test_only
    protocol['overrides']['seed'] = 99
    (folder/'s3r_s1.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='identity'):
        read_configuration(path)


def test_reject_legacy_sgd_checkpoint_as_adam_continuation(factory, tmp_path):
    model = factory('s3r')
    model.save_network(model.net, str(tmp_path), 0, 'latest')
    path = tmp_path/'T0_latest.pth'
    state = torch.load(path, weights_only=True)
    del state['config']['s3r_optimizer']
    torch.save(state,path)
    with pytest.raises(ValueError, match='s3r_optimizer'):
        model.load_network(path, model.net)
