import sys
from pathlib import Path

gt_path = Path("humob2026_destination_diffusion/data/raw/humob2026-dataset.tsv")
blind_dates = []
with open(gt_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = line.split('\t')[0]
        if '20240201' <= d <= '20240430':
            blind_dates.append(d)
print(f"Blind dates in GT ({len(blind_dates)}):", blind_dates)
