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
from models.utils.smg_segmentation_step import segmentation_step

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(2)
device = torch.device('cuda:0')
mode = os.environ['LOSS_MODE']
if mode not in ('clamp', 'minmax'):
    raise ValueError('Unsupported replay scaling')
root = Path(os.environ['WORK_ROOT'])
checkpoint = torch.load(os.environ['BASE_CHECKPOINT'], map_location='cpu', weights_only=True)
assert checkpoint['model']['_extra_state']['current_task'] == 1
assert checkpoint['model']['_extra_state']['task_open'] is False
model = resunet32('mid').to(device)
model.load_state_dict({k.removeprefix('net.core.'): v for k, v in checkpoint['model'].items()
                       if k.startswith('net.core.')})
generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=200).to(device)
generator.load_state_dict({k.removeprefix('generator.'): v for k, v in checkpoint['model'].items()
                           if k.startswith('generator.')})
if mode == 'minmax':
    original_sample = generator.sample
    def scaled_sample(*args, **kwargs):
        return normalize_per_slice(original_sample(*args, **kwargs))
    generator.sample = scaled_sample
optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
optimizer.load_state_dict(checkpoint['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)
train_data = ProstateCL_Dataset('2', 'train', 256)
mask_loader = DataLoader(train_data, batch_size=8, shuffle=False)
masks = torch.cat([y[:, None].float() for _, y in mask_loader])
buffer = SMGReplayBuffer(12 * 1024**3)
buffer.build(generator, masks, [0], batch_size=8)
replay_saturation = float((buffer.images.abs() >= .9999).float().mean())
train = DataLoader(train_data, batch_size=8, shuffle=True)

def objective(logits, y):
    loss = F.cross_entropy(logits, y)
    fg = logits.softmax(dim=1)[:, 1]
    target = (y == 1).float()
    return loss + 1 - (2 * (fg * target).sum() + 1) / (fg.sum() + target.sum() + 1)

for epoch in range(20):
    for x, y in train:
        x, y = normalize_per_slice(x.to(device)), y.to(device)
        replay = buffer.sample(x.shape[0], device)
        info = segmentation_step(model, optimizer, objective, generator, x, y, 1, replay_batch=replay)
    print(mode, 'epoch', epoch + 1, 'loss', info['loss'], flush=True)
model.eval()
scores = {}
for site in (1, 2):
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
result = {'mode': mode, 'seed': seed, 'epochs': 20, 'scores': scores,
          'replay_saturation_fraction': replay_saturation,
          'checkpoint_role': 'fixed seed2 first-task v2 branch, not exact resume',
          'selection_split': 'sites 1 and 2 validation only', 'test_used': False}
root.mkdir(parents=True, exist_ok=True)
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
