import torch, numpy as np, pickle
from pathlib import Path

PACKAGE_ROOT = Path("humob2026_destination_diffusion")
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'
TSV_FILE     = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions.tsv'

with open(META_PKL, 'rb') as f: meta_1476 = pickle.load(f)
c_idx = next(r['c_idx'] for r in meta_1476['active_routes'] if r['pair_key'] == '57_44-57_44')
ox = next(r['ox'] for r in meta_1476['active_routes'] if r['pair_key'] == '57_44-57_44')
oy = next(r['oy'] for r in meta_1476['active_routes'] if r['pair_key'] == '57_44-57_44')

# Let's check the raw output Z from the checkpoint, oh wait we need to sample it.
# Actually I have the predictions in dest1476_predictions.tsv. Let's just look at them.
import pandas as pd
data = []
with open(TSV_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2:
            try:
                od = eval(pts[1].replace(': NA', ': None').replace(':NA', ':None'), {'__builtins__': {}}, {'None': None})
                if od and '57_44' in od and '57_44' in od['57_44']:
                    data.append((pts[0], od['57_44']['57_44']))
            except: pass
for i in range(14):
    print(data[i])

