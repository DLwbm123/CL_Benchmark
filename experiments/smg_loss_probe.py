"""Fixed train/validation probe for SMG's base segmentation loss."""
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
from models.utils.smg_intensity import normalize_per_slice

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(2)
device = torch.device('cuda:0')
mode = os.environ['LOSS_MODE']
if mode not in ('ce', 'ce_dice'):
    raise ValueError('Unsupported loss')
root = Path(os.environ['WORK_ROOT'])
train = DataLoader(ProstateCL_Dataset('1', 'train', 256), batch_size=8, shuffle=True)
val = DataLoader(ProstateCL_Dataset('1', 'val', 256), batch_size=8, shuffle=False)
model = resunet32('mid').to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(20):
    model.train()
    for x, y in train:
        x, y = normalize_per_slice(x.to(device)), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if mode == 'ce_dice':
            fg = logits.softmax(dim=1)[:, 1]
            target = (y == 1).float()
            loss = loss + 1 - (2 * (fg * target).sum() + 1) / (fg.sum() + target.sum() + 1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(mode, 'epoch', epoch + 1, 'loss', float(loss), flush=True)
model.eval()
tp = predicted = actual = pixels = 0
with torch.no_grad():
    for x, y in val:
        pred = model(normalize_per_slice(x.to(device))).argmax(dim=1).cpu()
        tp += int(((pred == 1) & (y == 1)).sum())
        predicted += int((pred == 1).sum())
        actual += int((y == 1).sum())
        pixels += y.numel()
result = {'mode': mode, 'seed': seed, 'epochs': 20,
          'foreground_micro_dice': 2 * tp / (predicted + actual) if predicted + actual else None,
          'predicted_foreground_fraction': predicted / pixels,
          'reference_foreground_fraction': actual / pixels,
          'selection_split': 'site 1 validation only', 'test_used': False}
root.mkdir(parents=True, exist_ok=True)
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
