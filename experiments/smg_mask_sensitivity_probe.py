"""Counterfactual mask sensitivity with identical initial noise; train split only."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.set_num_threads(2)
device = torch.device('cuda:0')
checkpoint = torch.load(os.environ['BASE_CHECKPOINT'], map_location='cpu', weights_only=True)
assert checkpoint['model']['_extra_state']['current_task'] == 1
assert not checkpoint['model']['_extra_state']['task_open']
generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=200).to(device)
generator.load_state_dict({k.removeprefix('generator.'): v for k, v in checkpoint['model'].items() if k.startswith('generator.')})
generator.eval()
masks = []
for _, y in DataLoader(ProstateCL_Dataset('2', 'train', 256), batch_size=8, shuffle=False):
    for item in y:
        if bool((item == 1).any()):
            masks.append(item)
        if len(masks) == 24:
            break
    if len(masks) == 24:
        break
if len(masks) != 24:
    raise RuntimeError('Insufficient nonempty train masks')
counts = {mode: {'raw_sum_abs': 0., 'clamped_sum_abs': 0., 'mask_region_sum_abs': 0., 'background_sum_abs': 0.} for mode in ('empty', 'shifted')}
pixels = mask_pixels = 0
with torch.inference_mode():
    for offset in range(0, 24, 8):
        mask = torch.stack(masks[offset:offset+8]).to(device)[:, None].float()
        site = torch.zeros(len(mask), dtype=torch.long, device=device)
        noise = torch.randn_like(mask)
        ref = generator.sample(mask, site, tuple(mask.shape), noise=noise)
        if not torch.isfinite(ref).all():
            raise FloatingPointError('Nonfinite generated output')
        region = mask.bool()
        pixels += mask.numel()
        mask_pixels += int(region.sum())
        for name, alt in (('empty', torch.zeros_like(mask)), ('shifted', torch.roll(mask, shifts=64, dims=-1))):
            out = generator.sample(alt, site, tuple(mask.shape), noise=noise)
            if not torch.isfinite(out).all():
                raise FloatingPointError('Nonfinite generated output')
            difference = (ref - out).abs()
            stats = counts[name]
            stats['raw_sum_abs'] += float(difference.sum())
            stats['clamped_sum_abs'] += float((ref.clamp(-1, 1) - out.clamp(-1, 1)).abs().sum())
            stats['mask_region_sum_abs'] += float(difference[region].sum())
            stats['background_sum_abs'] += float(difference[~region].sum())
results = {name: {'raw_mean_absolute_difference': s['raw_sum_abs'] / pixels,
                  'clamped_mean_absolute_difference': s['clamped_sum_abs'] / pixels,
                  'original_foreground_region_mean_difference': s['mask_region_sum_abs'] / mask_pixels,
                  'original_background_region_mean_difference': s['background_sum_abs'] / (pixels - mask_pixels)}
           for name, s in counts.items()}
result = {'seed':7, 'samples':24, 'sampling_steps':200, 'split':'first 24 nonempty site2 train masks',
          'controls':'same initial noise, original versus empty and 64 pixel horizontally rolled mask',
          'counterfactuals':results, 'foreground_fraction':mask_pixels / pixels, 'test_used':False,
          'interpretation':'Sensitivity alone is not fidelity or image quality.'}
Path(os.environ['WORK_ROOT']).joinpath('result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
