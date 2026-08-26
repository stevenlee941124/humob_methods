"""
===============================================================================
HuMob 2026: Step 2 - Build (1476, 70, 100) Dataset with 9-Class Physics Baseline & 4D Temporal Conditions
===============================================================================
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from nine_class_baseline import compute_9class_baseline
from japan_calendar import JAPAN_HOLIDAYS

OD_TS_PKL   = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL   = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'

OUT_META    = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'
OUT_BASE    = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'

print("=" * 75, flush=True)
print("[Step 2] Building (1476, 70, 100) Dataset with 9-Class Physics Baseline & 4D Conditions", flush=True)
print("=" * 75, flush=True)

with open(OD_TS_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)

# 1,476 評測目的地網格清單
eval_dest_grids = [f"{x}_{y}" for x in range(30, 71) for y in range(35, 71)]
dest_to_channel = {d: i for i, d in enumerate(eval_dest_grids)}

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

print(f"1/4 正在計算 15,129 條路線之 9 大類別物理動態 Baseline...", flush=True)
baselines = {}
route_class_info = {}
class_stats = {}

for pair_idx, (pair_key, raw_arr) in enumerate(od_ts.items()):
    y_366 = np.zeros(366, dtype=np.float64)
    for d_str, oi in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            v = raw_arr[oi]
            y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0

    b_366, cls_name, cls_id = compute_9class_baseline(y_366, cal_dates, cal_date_to_idx)
    baselines[pair_key] = b_366
    route_class_info[pair_key] = {'class_name': cls_name, 'class_id': cls_id}
    class_stats[cls_name] = class_stats.get(cls_name, 0) + 1

print("✅ 9 大類別分佈統計:")
for k, v in sorted(class_stats.items(), key=lambda x: -x[1]):
    print(f"  • {k:<55}: {v:>5} 條 ({v/len(od_ts)*100:.1f}%)", flush=True)

OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_BASE, 'wb') as f:
    pickle.dump(baselines, f)

print(f"\n2/4 正在構建 4 維時間條件向量 [星期/6, 國定假日, 月份/11, 天數/365]...", flush=True)
train_obs_dates = [d for d in dates_str if (d < '20240201' or d >= '20240501')]
train_cal_idxs  = [cal_date_to_idx[d] for d in train_obs_dates if d in cal_date_to_idx]
N_TRAIN_DAYS    = len(train_obs_dates)

train_cond = np.zeros((N_TRAIN_DAYS, 4), dtype=np.float32)
for j, d_str in enumerate(train_obs_dates):
    dt = datetime.strptime(d_str, '%Y%m%d')
    train_cond[j, 0] = dt.weekday() / 6.0
    train_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    train_cond[j, 2] = (dt.month - 1) / 11.0
    train_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

print(f"3/4 正在映射 1,476 個目的地通道與活躍路線殘差張量...", flush=True)
active_routes_info = []
for pair_key, raw_arr in od_ts.items():
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]

    if o_str == '-1_-1' or d_str not in dest_to_channel:
        continue

    try:
        ox, oy = map(int, o_str.split('_'))
        if not (1 <= ox <= 70 and 1 <= oy <= 100): continue
        ox_0, oy_0 = ox - 1, oy - 1
    except:
        continue

    c_idx = dest_to_channel[d_str]
    b_366 = baselines.get(pair_key)
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
        continue

    y_train = []
    b_train = []
    for d_str_k in train_obs_dates:
        oi = obs_date_to_idx.get(d_str_k)
        ci = cal_date_to_idx.get(d_str_k)
        if oi is not None and ci is not None:
            v = raw_arr[oi]
            y_train.append(float(v) if not np.isnan(v) else 0.0)
            b_train.append(float(b_366[ci]))

    y_train = np.array(y_train, dtype=np.float32)
    b_train = np.array(b_train, dtype=np.float32)
    res_train = y_train - b_train

    sigma_i = float(np.std(res_train))
    if sigma_i < 0.05: sigma_i = 0.05
    z_train = res_train / sigma_i

    mean_v = float(np.mean(y_train))
    max_v = float(np.max(y_train))
    if mean_v < 0.05 and max_v < 0.5: continue

    active_routes_info.append({
        'pair_key': pair_key,
        'o_str': o_str,
        'd_str': d_str,
        'c_idx': c_idx,
        'ox': ox_0,
        'oy': oy_0,
        'sigma_i': sigma_i,
        'z_train': z_train,
        'mean_v': mean_v,
        'max_v': max_v,
        'class_name': route_class_info.get(pair_key, {}).get('class_name', 'Class 5'),
        'class_id': route_class_info.get(pair_key, {}).get('class_id', 5)
    })

print(f"✅ 篩選出評測通道內活躍路線數: {len(active_routes_info):,} 條！", flush=True)

meta_1476 = {
    'dest_grid_list': eval_dest_grids,
    'dest_to_channel': dest_to_channel,
    'active_routes': active_routes_info,
    'train_obs_dates': train_obs_dates,
    'cal_dates': cal_dates,
    'train_cond': train_cond,
    'n_train': N_TRAIN_DAYS,
    'n_channels': len(eval_dest_grids),
    'grid_w': 70,
    'grid_h': 100,
    'route_class_info': route_class_info
}

with open(OUT_META, 'wb') as f:
    pickle.dump(meta_1476, f)

print(f"✅ 9 大類別物理 Baseline、4D 條件特徵與元數據儲存完成: {OUT_META}", flush=True)
print("=" * 75, flush=True)
