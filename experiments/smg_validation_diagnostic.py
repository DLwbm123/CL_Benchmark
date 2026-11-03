"""Read-only foreground diagnostics on fixed train/validation splits."""
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_intensity import normalize_per_slice

torch.set_num_threads(2)
root = Path(os.environ['WORK_ROOT'])
base = Path(os.environ['RUNS_ROOT'])
device = torch.device('cuda:0')
results = {}
for seed in (2, 4):
    ckpt = torch.load(base / f'20261102_smg_s{seed}_50k/checkpoints/T5_latest.pth',
                      map_location='cpu', weights_only=True)
    net = resunet32('mid').to(device).eval()
    net.load_state_dict({k.removeprefix('net.core.'): v for k, v in ckpt['model'].items()
                         if k.startswith('net.core.')})
    results[str(seed)] = {}
    for site in range(1, 7):
        loader = DataLoader(ProstateCL_Dataset(str(site), 'val', 256), batch_size=8)
        tp = predicted = actual = pixels = 0
        with torch.no_grad():
            for x, y in loader:
                pred = net(normalize_per_slice(x.to(device))).argmax(dim=1).cpu()
                tp += int(((pred == 1) & (y == 1)).sum())
                predicted += int((pred == 1).sum())
                actual += int((y == 1).sum())
                pixels += y.numel()
        results[str(seed)][str(site)] = {
            'foreground_micro_dice': 2 * tp / (predicted + actual) if predicted + actual else None,
            'predicted_foreground_fraction': predicted / pixels,
            'reference_foreground_fraction': actual / pixels,
        }
    print('seed', seed, 'validation complete', flush=True)
root.mkdir(parents=True, exist_ok=True)
(root / 'validation_diagnostic.json').write_text(json.dumps(results, indent=2) + '\n')
