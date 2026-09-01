"""
===============================================================================
HuMob 2026: Step 1 - Extract OD Time Series & Identify Evaluation Destinations
===============================================================================
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent

RAW_TSV      = PACKAGE_ROOT / 'data' / 'raw'       / 'humob2026-dataset.tsv'
OUT_OD_TS    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
OUT_DATES    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_DESTS    = PACKAGE_ROOT / 'data' / 'processed' / 'eval_destinations.pkl'
OUT_DEST_MAP = PACKAGE_ROOT / 'data' / 'processed' / 'dest_to_origins.pkl'

print("=" * 75)
print("[Step 1 Dest] Extracting OD Time Series & Indexing Destination Grids")
print("=" * 75)

dates_list = []
daily_data = {}

# 1. 讀取原始 TSV
with open(RAW_TSV, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        d_str = parts[0]
        raw_dict_str = parts[1].replace(': NA', ': None').replace(':NA', ':None')
        try:
            od_dict = eval(raw_dict_str, {'__builtins__': {}}, {'None': None})
            if od_dict is not None:
                dates_list.append(d_str)
                daily_data[d_str] = od_dict
        except Exception as e:
            pass

dates_list.sort()
N_DAYS = len(dates_list)
print(f"✅ 成功讀取 {N_DAYS} 天觀測資料 (從 {dates_list[0]} 到 {dates_list[-1]})")

# 2. 彙整全域所有 OD 路線與目的地倒排索引
od_time_series = {}
dest_to_origins = {}

for d_idx, d_str in enumerate(dates_list):
    od_dict = daily_data[d_str]
    for orig, dests in od_dict.items():
        if not isinstance(dests, dict):
            continue
        for dest, flow in dests.items():
            pair_key = f"{orig}-{dest}"
            if pair_key not in od_time_series:
                od_time_series[pair_key] = np.full(N_DAYS, np.nan, dtype=np.float32)
            if flow is not None:
                od_time_series[pair_key][d_idx] = float(flow)
                
            dest_to_origins.setdefault(dest, set()).add(orig)

# 3. 識別官方評測範圍 (X ∈ [30, 70], Y ∈ [35, 70]) 內的活躍目的地
def in_eval_bbox(grid_str):
    try:
        x, y = map(int, grid_str.split('_'))
        return 30 <= x <= 70 and 35 <= y <= 70
    except:
        return False

eval_destinations = [d for d in dest_to_origins.keys() if in_eval_bbox(d)]
eval_destinations.sort(key=lambda g: (int(g.split('_')[0]), int(g.split('_')[1])))

print(f"✅ 全域總 OD 路線數: {len(od_time_series):,} 條")
print(f"✅ 全域總活躍目的地數: {len(dest_to_origins):,} 個")
print(f"🎯 官方評測範圍內活躍目的地數: {len(eval_destinations):,} 個 (X:30~70, Y:35~70)")

# 4. 儲存提取結果
OUT_OD_TS.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_OD_TS, 'wb')    as f: pickle.dump(od_time_series, f)
with open(OUT_DATES, 'wb')    as f: pickle.dump(dates_list, f)
with open(OUT_DESTS, 'wb')    as f: pickle.dump(eval_destinations, f)
with open(OUT_DEST_MAP, 'wb') as f: pickle.dump({d: list(origs) for d, origs in dest_to_origins.items()}, f)

print(f"✅ 檔案已儲存至: {OUT_OD_TS.parent}")
print("=" * 75)
