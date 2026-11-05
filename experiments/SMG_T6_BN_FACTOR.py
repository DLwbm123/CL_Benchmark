"""Train-only two-site branch from a fixed checkpoint, with validation diagnostics."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import normalize_per_slice
from models.utils.smg_replay import SMGReplayBuffer
import torch
from torch.func import functional_call
from models.utils.smg_dual_meta import first_order_dual_meta

def factor_step(model, optimizer, loss_fn, current, replay, order, update, bn_source):
    x,y=current; rx,ry=replay
    cx=torch.cat((x,rx)); cy=torch.cat((y,ry))
    model.train()
    with torch.no_grad():
        model(x if bn_source == 'current' else cx)
    if update == 'dual':
        mid=len(order)//2
        total, grads=first_order_dual_meta(model,loss_fn,current,replay,(cx[order[:mid]],cy[order[:mid]]),(cx[order[mid:]],cy[order[mid:]]))
        optimizer.zero_grad(set_to_none=True)
        for name,param in model.named_parameters(): param.grad=grads[name]
    elif update == 'joint':
        optimizer.zero_grad(set_to_none=True)
        buffers={name:value.detach().clone() for name,value in model.named_buffers()}
        total=loss_fn(functional_call(model,(dict(model.named_parameters()),buffers),(cx,)),cy)
        if not torch.isfinite(total): raise FloatingPointError('nonfinite loss')
        total.backward()
    else: raise ValueError(update)
    optimizer.step()
    return float(total.detach())


seed = int(os.environ['PROBE_SEED'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(2)
device = torch.device('cuda:0')
mode = os.environ['LOSS_MODE']
bn_source = os.environ['BN_SOURCE']
if bn_source not in ('current', 'mixed'): raise ValueError('invalid BN policy')
if mode not in ('dual', 'joint'):
    raise ValueError('Unsupported replay scaling')
root = Path(os.environ['WORK_ROOT'])
checkpoint = torch.load(os.environ['BASE_CHECKPOINT'], map_location='cpu', weights_only=True)
assert checkpoint['model']['_extra_state']['current_task'] == 5
assert checkpoint['model']['_extra_state']['task_open'] is False
model = resunet32('mid').to(device)
model.load_state_dict({k.removeprefix('net.core.'): v for k, v in checkpoint['model'].items()
                       if k.startswith('net.core.')})
generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=200).to(device)
generator.load_state_dict({k.removeprefix('generator.'): v for k, v in checkpoint['model'].items()
                           if k.startswith('generator.')})
optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
optimizer.load_state_dict(checkpoint['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)
train_data = ProstateCL_Dataset('6', 'train', 256)
mask_loader = DataLoader(train_data, batch_size=8, shuffle=False)
masks = torch.cat([y[:, None].float() for _, y in mask_loader])
buffer = SMGReplayBuffer(12 * 1024**3)
buffer.build(generator, masks, list(range(5)), batch_size=8)
replay_saturation = float((buffer.images.abs() >= .9999).float().mean())
train_rng = torch.Generator().manual_seed(seed + 11000)
replay_rng = torch.Generator().manual_seed(seed + 12000)
split_rng = torch.Generator().manual_seed(seed + 13000)
train = DataLoader(train_data, batch_size=8, shuffle=True, generator=train_rng)

def objective(logits, y):
    loss = F.cross_entropy(logits, y)
    fg = logits.softmax(dim=1)[:, 1]
    target = (y == 1).float()
    return loss + 1 - (2 * (fg * target).sum() + 1) / (fg.sum() + target.sum() + 1)

for epoch in range(20):
    for x, y in train:
        x, y = normalize_per_slice(x.to(device)), y.to(device)
        idx = torch.randint(len(buffer.images), (x.shape[0],), generator=replay_rng)
        replay = (buffer.images[idx].to(device), buffer.labels[idx].to(device))
        order = torch.randperm(2*x.shape[0], generator=split_rng).to(device)
        value = factor_step(model, optimizer, objective, (x,y), replay, order, mode, bn_source)
        info = {'loss': value}
    print(mode, 'epoch', epoch + 1, 'loss', info['loss'], flush=True)
model.eval()
scores = {}
for site in (1, 2, 6):
    tp = predicted = actual = pixels = 0
    with torch.no_grad():
        for x, y in DataLoader(ProstateCL_Dataset(str(site), 'val', 256), batch_size=8):
            pred = model(normalize_per_slice(x.to(device))).argmax(dim=1).cpu()
            tp += int(((pred == 1) & (y == 1)).sum())
            predicted += int((pred == 1).sum())
            actual += int((y == 1).sum())
            pixels += y.numel()
    scores[str(site)] = {'foreground_micro_dice': 2 * tp / (predicted + actual) if predicted + actual else None,
                         'predicted_foreground_fraction': predicted / pixels,
                         'reference_foreground_fraction': actual / pixels}
result = {'mode': mode, 'bn_source': bn_source, 'seed': seed, 'epochs': 20, 'scores': scores,
          'replay_saturation_fraction': replay_saturation,
          'checkpoint_role': 'fixed seed11 pilot fifth-task branch, not exact resume',
          'selection_split': 'sites 1, 2 and 6 validation only', 'test_used': False}
root.mkdir(parents=True, exist_ok=True)
def finite(v):
    if torch.is_tensor(v): return bool(torch.isfinite(v).all())
    if isinstance(v, dict): return all(finite(x) for x in v.values())
    if isinstance(v, (list, tuple)): return all(finite(x) for x in v)
    return True
result['finite'] = finite(model.state_dict()) and finite(optimizer.state_dict())
if not result['finite']: raise FloatingPointError('nonfinite model or optimizer')
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
