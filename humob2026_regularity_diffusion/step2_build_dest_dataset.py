"""
===============================================================================
HuMob 2026: Step 2 - Build Destination-Centric 2D Spatial Inflow Tensor Dataset
===============================================================================
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from japan_calendar import JAPAN_HOLIDAYS
from baseline_module import compute_full_baseline

OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
DESTS_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'eval_destinations.pkl'
DEST_MAP_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dest_to_origins.pkl'

OUT_BASELINE = PACKAGE_ROOT / 'data' / 'outputs'   / 'full_year_baseline.pkl'
OUT_NPZ      = PACKAGE_ROOT / 'data' / 'outputs'   / 'dest_diffusion_dataset.npz'
OUT_META     = PACKAGE_ROOT / 'data' / 'outputs'   / 'dest_meta.pkl'

GRID_W, GRID_H = 70, 100
N_CHANNELS = 2 # Ch 0: Retention (d->d), Ch 1: Inflows (o->d)

print("=" * 75)
print("[Step 2 Dest] Building Destination-Centric Inflow Tensor Dataset")
print("=" * 75)

with open(OD_PKL, 'rb')       as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)
with open(DESTS_PKL, 'rb')    as f: eval_destinations = pickle.load(f)
with open(DEST_MAP_PKL, 'rb') as f: dest_to_origins = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}
N_OBS = len(dates_str)

# 1. 計算全域所有 OD 路線之 4 月錨定動態 Baseline
print("正在計算全域 15,129 條 OD 路線之 4 月真實錨定動態 Baseline...")
baselines = {}
for k, raw_arr in od_ts.items():
    y_366 = np.zeros(366, dtype=np.float64)
    for d_str, oi in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            v = raw_arr[oi]
            y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0
    b_366, _, _ = compute_full_baseline(y_366, cal_dates, cal_date_to_idx)
    baselines[k] = b_366

OUT_BASELINE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_BASELINE, 'wb') as f:
    pickle.dump(baselines, f)
print(f"✅ Baseline 已儲存至: {OUT_BASELINE}")

# 2. 構建日曆特徵向量 (N_OBS, 4)
cal_features = np.zeros((N_OBS, 4), dtype=np.float32)
for i, d_str in enumerate(dates_str):
    dt = datetime.strptime(d_str, '%Y%m%d')
    cal_features[i, 0] = dt.weekday() / 6.0
    cal_features[i, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    cal_features[i, 2] = (dt.month - 1) / 11.0
    cal_features[i, 3] = cal_date_to_idx[d_str] / 365.0

# 3. 空間座標解析工具
def parse_coord(grid_str):
    if grid_str == '-1_-1': return None
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
            return (x - 1, y - 1)
    except:
        pass
    return None

# 4. 篩選訓練用之目的地 (流量有實質活動者)
dest_stats = {}
train_dest_list = []
dest_sigmas = {} # {dest_str: sigma_tensor (2, 70, 100)}

train_obs_indices = [i for i, d in enumerate(dates_str) if (d < '20240101' or d >= '20240401')]

for d_str in eval_destinations:
    d_coord = parse_coord(d_str)
    if d_coord is None: continue
    
    origs = dest_to_origins.get(d_str, [])
    # 檢查該目的地總流量
    total_flow = 0.0
    for o_str in origs:
        raw_arr = od_ts.get(f"{o_str}-{d_str}")
        if raw_arr is not None:
            valid_vals = [v for v in raw_arr if not np.isnan(v)]
            total_flow += sum(valid_vals)
            
    if total_flow >= 5.0:
        train_dest_list.append(d_str)

print(f"✅ 篩選出具備實質流動之訓練目的地: {len(train_dest_list):,} / {len(eval_destinations):,} 個")

# 5. 構建訓練張量資料庫
# 每個樣本對應一個 (t, D) 的 2D 空間場: (2, 70, 100)
# 為了高效訓練，我們構建 (N_samples, 2, 70, 100), cond (N_samples, 6) = [cal_4, dest_x, dest_y]
sample_z_list = []
sample_cond_list = []

print("正在構建每日各目的地 2D 空間吸引力場張量...")
for d_idx, d_str in enumerate(train_dest_list):
    d_coord = parse_coord(d_str)
    origs = dest_to_origins.get(d_str, [])
    
    # 構建該目的地在觀測日的 (N_OBS, 2, 70, 100) 張量
    y_d = np.zeros((N_OBS, 2, GRID_W, GRID_H), dtype=np.float32)
    b_d = np.zeros((N_OBS, 2, GRID_W, GRID_H), dtype=np.float32)
    
    for o_str in origs:
        pair_key = f"{o_str}-{d_str}"
        raw_arr = od_ts.get(pair_key)
        b_366 = baselines.get(pair_key)
        o_coord = parse_coord(o_str)
        is_retention = (o_str == d_str)
        
        for t_idx, d_date in enumerate(dates_str):
            c_idx = cal_date_to_idx.get(d_date)
            y_val = raw_arr[t_idx] if raw_arr is not None else np.nan
            b_val = b_366[c_idx] if (b_366 is not None and c_idx is not None) else 0.0
            
            vy = float(y_val) if not np.isnan(y_val) else 0.0
            vb = float(b_val) if not np.isnan(b_val) else 0.0
            
            if is_retention and d_coord is not None:
                y_d[t_idx, 0, d_coord[0], d_coord[1]] = vy
                b_d[t_idx, 0, d_coord[0], d_coord[1]] = vb
            elif not is_retention and o_coord is not None:
                y_d[t_idx, 1, o_coord[0], o_coord[1]] = vy
                b_d[t_idx, 1, o_coord[0], o_coord[1]] = vb

    # 計算該目的地空間殘差標準差
    resids = y_d[train_obs_indices] - b_d[train_obs_indices]
    sigma_d = np.std(resids, axis=0) # (2, 70, 100)
    sigma_d = np.maximum(sigma_d, 0.1)
    dest_sigmas[d_str] = sigma_d
    
    # 計算標準化空間殘差張量
    z_d = (y_d - b_d) / sigma_d
    
    # 採樣加入訓練集 (取觀測訓練日)
    dest_cond_pos = np.array([d_coord[0] / float(GRID_W), d_coord[1] / float(GRID_H)], dtype=np.float32)
    
    for ti in train_obs_indices:
        cond_6d = np.concatenate([cal_features[ti], dest_cond_pos])
        sample_z_list.append(z_d[ti])
        sample_cond_list.append(cond_6d)

sample_z_arr = np.array(sample_z_list, dtype=np.float32)
sample_cond_arr = np.array(sample_cond_list, dtype=np.float32)

print(f"✅ 訓練樣本集構建完成: 總張量數 {len(sample_z_arr):,}, 形狀 {sample_z_arr.shape}")
print(f"✅ 條件特徵矩陣形狀: {sample_cond_arr.shape} (4維日曆 + 2維目的地坐標)")

# 6. 儲存空間資料集
np.savez_compressed(
    str(OUT_NPZ),
    sample_z=sample_z_arr,
    sample_cond=sample_cond_arr
)

dest_meta = {
    'grid_w': GRID_W,
    'grid_h': GRID_H,
    'n_channels': N_CHANNELS,
    'eval_destinations': eval_destinations,
    'train_dest_list': train_dest_list,
    'dest_sigmas': dest_sigmas
}

with open(OUT_META, 'wb') as f:
    pickle.dump(dest_meta, f)

print(f"✅  dest_diffusion_dataset.npz -> {OUT_NPZ}")
print(f"✅  dest_meta.pkl              -> {OUT_META}")
print("=" * 75)
