"""Aggregate foreground prevalence from training splits; no test access."""
import json, os
from pathlib import Path
from datasets.seq_prostate import ProstateCL_Dataset
from torch.utils.data import DataLoader
result = {}
for site in range(1, 7):
    total = foreground = empty = count = 0
    for _, y in DataLoader(ProstateCL_Dataset(str(site), "train", 256), batch_size=8, shuffle=False):
        z = y == 1
        sums = z.sum(dim=(1, 2))
        foreground += int(sums.sum())
        total += z.numel()
        empty += int((sums == 0).sum())
        count += len(y)
    result[str(site)] = {"foreground_fraction": foreground / total, "empty_slice_fraction": empty / count, "slice_count": count}
Path(os.environ["OUTPUT_DIR"]).joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
