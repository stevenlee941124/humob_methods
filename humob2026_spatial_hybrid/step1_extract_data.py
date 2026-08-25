"""
===============================================================================
HuMob 2026 Hybrid: Step 1 - Extract OD Time Series
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_TSV = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
OUT_OD  = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
OUT_DT  = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'

print("=" * 80)
print("[Step 1] Extracting OD Time Series from raw TSV...")
print("=" * 80)

OUT_OD.parent.mkdir(parents=True, exist_ok=True)

dates = []
daily_data = {}

with open(RAW_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) < 2:
            continue
        d_str = pts[0]
        dates.append(d_str)
        try:
            raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            od_dict = eval(raw, {'__builtins__': {}}, {'None': None})
            daily_data[d_str] = od_dict if od_dict is not None else {}
        except Exception:
            daily_data[d_str] = {}

all_od_keys = set()
for d_str, od_dict in daily_data.items():
    for orig, dest_dict in od_dict.items():
        for dest in dest_dict.keys():
            all_od_keys.add(f"{orig}-{dest}")

sorted_keys = sorted(list(all_od_keys))
n_days = len(dates)
od_ts = {k: np.full(n_days, np.nan, dtype=np.float32) for k in sorted_keys}

for day_idx, d_str in enumerate(dates):
    od_dict = daily_data.get(d_str, {})
    for orig, dest_dict in od_dict.items():
        for dest, val in dest_dict.items():
            k = f"{orig}-{dest}"
            if k in od_ts and val is not None:
                od_ts[k][day_idx] = float(val)

with open(OUT_OD, 'wb') as f: pickle.dump(od_ts, f)
with open(OUT_DT, 'wb') as f: pickle.dump(dates, f)

print(f"✅ Extracted {len(od_ts):,} OD pairs across {len(dates)} observation days.")
print(f"✅ Saved to: {OUT_OD} and {OUT_DT}")
