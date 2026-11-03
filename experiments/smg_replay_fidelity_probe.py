"""Train-mask replay agreement with a fixed teacher; no training or test access."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import normalize_per_slice

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.set_num_threads(2)
device = torch.device('cuda:0')
checkpoint = torch.load(os.environ['BASE_CHECKPOINT'], map_location='cpu', weights_only=True)
assert checkpoint['model']['_extra_state']['current_task'] == 1
assert checkpoint['model']['_extra_state']['task_open'] is False
model = resunet32('mid').to(device)
model.load_state_dict({k.removeprefix('net.core.'): v for k, v in checkpoint['model'].items() if k.startswith('net.core.')})
model.eval()
generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=200).to(device)
generator.load_state_dict({k.removeprefix('generator.'): v for k, v in checkpoint['model'].items() if k.startswith('generator.')})
generator.eval()

def accumulate(stats, x, y):
    pred = model(x).argmax(1)
    stats[0] += int(((pred == 1) & (y == 1)).sum())
    stats[1] += int((pred == 1).sum())
    stats[2] += int((y == 1).sum())
    stats[3] += y.numel()

def summarize(s):
    tp, predicted, actual, pixels = s
    return {'foreground_micro_dice': 2 * tp / (predicted + actual) if predicted + actual else None,
            'predicted_foreground_fraction': predicted / pixels,
            'reference_foreground_fraction': actual / pixels}

real = [0, 0, 0, 0]
replay = {mode: [0, 0, 0, 0] for mode in ('clamp', 'minmax')}
saturated = {'clamp': 0, 'minmax': 0}
pixel_count = 0
with torch.inference_mode():
    for x, y in DataLoader(ProstateCL_Dataset('1', 'val', 256), batch_size=8):
        accumulate(real, normalize_per_slice(x.to(device)), y.to(device))
    for _, y in DataLoader(ProstateCL_Dataset('2', 'train', 256), batch_size=8, shuffle=False):
        y = y.to(device)
        mask = y[:, None].float()
        site = torch.zeros(len(y), dtype=torch.long, device=device)
        raw = generator.sample(mask, site, tuple(mask.shape))
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite replay')
        for mode, x in (('clamp', raw.clamp(-1, 1)), ('minmax', normalize_per_slice(raw).clamp(-1, 1))):
            accumulate(replay[mode], x, y)
            saturated[mode] += int((x.abs() >= .9999).sum())
        pixel_count += raw.numel()
result = {'seed': 7, 'teacher': 'fixed v2 seed2 task 1 checkpoint',
          'real_site1_validation': summarize(real),
          'replay_site2_train_mask_agreement': {m: {**summarize(s), 'saturation_fraction': saturated[m] / pixel_count} for m, s in replay.items()},
          'interpretation': 'Teacher agreement is a proxy, not true mask fidelity. Both transforms share each raw generated batch.',
          'test_used': False}
Path(os.environ['WORK_ROOT']).joinpath('result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
