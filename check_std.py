import pickle, numpy as np, pandas as pd
from pathlib import Path

TSV = Path("humob2026_destination_diffusion/data/outputs/dest1476_predictions.tsv")
OD = Path("humob2026_destination_diffusion/data/processed/od_time_series.pkl")

with open(OD, 'rb') as f:
    od_ts = pickle.load(f)
raw = od_ts['39_46-39_46']
valid = [x for x in raw if not np.isnan(x)]
print(f"Red Line (Actual) std: {np.std(valid):.2f}, mean: {np.mean(valid):.2f}")

preds = []
with open(TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2:
            try:
                d = eval(pts[1].replace(': NA', ': None').replace(':NA', ':None'), {'__builtins__': {}}, {'None': None})
                if '39_46' in d and '39_46' in d['39_46']:
                    preds.append(d['39_46']['39_46'])
            except: pass
print(f"Blue Line (Pred) std: {np.std(preds):.2f}, mean: {np.mean(preds):.2f}")
