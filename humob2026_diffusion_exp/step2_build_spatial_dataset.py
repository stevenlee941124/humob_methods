"""
===============================================================================
HuMob 2026: Step 2 Spatial - Build 2D Spatial-Temporal Grid Tensor Dataset
===============================================================================
詳細數學模型與推導，請參閱：DIFFUSION_MATHEMATICAL_MODEL.md

資料結構範例 (Data Structure Examples)：
  - 輸入 (od_time_series.pkl): 
    {
      '31_38-31_38': array([12.0, 15.0, np.nan, ...]),
      ...
    }
  - 輸出 (spatial_diffusion_dataset.npz):
    包含 'spatial_z' 陣列 (N, 4, 70, 100)，代表每天 4 個通道的 2D 空間特徵，例如:
    Z[0, 0, 30, 37] = 0.5 (代表第一天、通道0、座標x=30, y=37 的標準化殘差)

輸入:
  - od_time_series.pkl (14,563 條 OD 路線時序, 292天)
  - dates.pkl (292天日期清單)
  - full_year_baseline.pkl (全年中軸 Baseline)
處理:
  1. 空間網格座標界定: X ∈ [1, 70], Y ∈ [1, 100] -> (70, 100) 空間網格
  2. 構建每日 4 通道 2D 空間張量 (Channels: Retention, Outflow, Inflow, External Exchange)
  3. 計算空間中軸 Baseline 張量 B_spatial(t) 與空間殘差標準差 σ_spatial
  4. 輸出標準化空間殘差張量 Z_spatial(t) ∈ R^(4, 70, 100)
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
from japan_calendar import JAPAN_HOLIDAYS
from exponential_baseline import compute_full_baseline

OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs'   / 'full_year_baseline.pkl'

OUT_NPZ      = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_diffusion_dataset.npz'
OUT_META     = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_meta.pkl'

GRID_W, GRID_H = 70, 100
N_CHANNELS = 4

print("=" * 75)
print("[Step 2 Spatial] Building 2D Spatial-Temporal Grid Tensor Dataset")
print("=" * 75)

with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}
N_OBS = len(dates_str)

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

BASELINE_PKL.parent.mkdir(parents=True, exist_ok=True)
with open(BASELINE_PKL, 'wb') as f:
    pickle.dump(baselines, f)
print(f"✅ 已成功計算並儲存 4 月錨定動態 Baseline 至: {BASELINE_PKL}")

# ── 1. 解析所有 OD 路線之 (ox, oy) -> (dx, dy) 空間座標 ──────
def parse_grid_coord(grid_str):
    if grid_str == '-1_-1':
        return None
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
            return (x - 1, y - 1) # 0-indexed
    except:
        pass
    return None

od_spatial_map = []
for pair_key, raw_arr in od_ts.items():
    parts = pair_key.replace('-1_-1-', '-1_-1_').split('-') if pair_key.startswith('-1_-1-') else pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
    
    o_coord = parse_grid_coord(o_str)
    d_coord = parse_grid_coord(d_str)
    
    b_366 = baselines.get(pair_key)
    
    od_spatial_map.append({
        'pair_key': pair_key,
        'o_str': o_str,
        'd_str': d_str,
        'o_coord': o_coord,
        'd_coord': d_coord,
        'is_diag': (o_str == d_str and o_coord is not None),
        'is_ext': (o_str == '-1_-1' or d_str == '-1_-1'),
        'raw_arr': raw_arr,
        'b_366': b_366
    })

print(f"成功解析 {len(od_spatial_map):,} 條 OD 路線的空間拓撲對映關係！")

# ── 2. 構建每日 (292, 4, 70, 100) 的空間張量 ─────────────────
spatial_y = np.zeros((N_OBS, N_CHANNELS, GRID_W, GRID_H), dtype=np.float32)
spatial_b = np.zeros((N_OBS, N_CHANNELS, GRID_W, GRID_H), dtype=np.float32)
spatial_obs_mask = np.zeros((N_OBS, N_CHANNELS, GRID_W, GRID_H), dtype=bool)

for item in od_spatial_map:
    raw_arr = item['raw_arr']
    b_366 = item['b_366']
    o_c = item['o_coord']
    d_c = item['d_coord']
    is_diag = item['is_diag']
    is_ext = item['is_ext']
    
    for t_idx, d_str in enumerate(dates_str):
        c_idx = cal_date_to_idx.get(d_str)
        y_val = raw_arr[t_idx]
        b_val = b_366[c_idx] if (b_366 is not None and c_idx is not None) else 0.0
        
        has_y = not np.isnan(y_val)
        val_y = float(y_val) if has_y else 0.0
        val_b = float(b_val) if not np.isnan(b_val) else 0.0
        
        # Channel 0: 留存流動 (Retention)
        if is_diag and o_c is not None:
            spatial_y[t_idx, 0, o_c[0], o_c[1]] += val_y
            spatial_b[t_idx, 0, o_c[0], o_c[1]] += val_b
            if has_y: spatial_obs_mask[t_idx, 0, o_c[0], o_c[1]] = True
            
        # 內部跨區流動 (Cross-grid Flow)
        elif not is_ext and not is_diag:
            # Channel 1: 內部跨區流出 (Outflow) at Origin
            if o_c is not None:
                spatial_y[t_idx, 1, o_c[0], o_c[1]] += val_y
                spatial_b[t_idx, 1, o_c[0], o_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 1, o_c[0], o_c[1]] = True
            # Channel 2: 內部跨區流入 (Inflow) at Destination
            if d_c is not None:
                spatial_y[t_idx, 2, d_c[0], d_c[1]] += val_y
                spatial_b[t_idx, 2, d_c[0], d_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 2, d_c[0], d_c[1]] = True
            
        # Channel 3: 外域人流交換 (External Exchange)
        elif is_ext:
            target_c = d_c if o_c is None else o_c
            if target_c is not None:
                spatial_y[t_idx, 3, target_c[0], target_c[1]] += val_y
                spatial_b[t_idx, 3, target_c[0], target_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 3, target_c[0], target_c[1]] = True

print(f"✅ 每日 4 通道空間流量場構建完成: 形狀 {spatial_y.shape}")

# ── 3. 計算空間殘差標準差 σ_spatial (4, 70, 100) ──────────────
train_obs_indices = [i for i, d in enumerate(dates_str) if (d < '20240101' or d >= '20240501')]

spatial_residuals = spatial_y[train_obs_indices] - spatial_b[train_obs_indices]
spatial_sigma = np.std(spatial_residuals, axis=0) # (4, 70, 100)
# 數值穩定性截斷
spatial_sigma = np.maximum(spatial_sigma, 0.1)

# 計算標準化空間殘差張量 Z(t) = (Y - B) / σ
spatial_z = (spatial_y - spatial_b) / spatial_sigma

print(f"✅ 空間標準差圖層 σ_spatial 計算完成: 均值 {np.mean(spatial_sigma):.3f}, 最大 {np.max(spatial_sigma):.3f}")
print(f"✅ 空間標準化殘差 Z_spatial 分佈: 均值 {np.mean(spatial_z):.3f}, 標準差 {np.std(spatial_z):.3f}")

# ── 4. 構建日曆特徵向量 (292, 4) ──────────────────────────────
cal_features = np.zeros((N_OBS, 4), dtype=np.float32)
for i, d_str in enumerate(dates_str):
    dt = datetime.strptime(d_str, '%Y%m%d')
    cal_features[i, 0] = dt.weekday() / 6.0
    cal_features[i, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    cal_features[i, 2] = (dt.month - 1) / 11.0
    cal_features[i, 3] = cal_date_to_idx[d_str] / 365.0

# ── 5. 儲存空間資料集 ──────────────────────────────────────────
np.savez_compressed(
    str(OUT_NPZ),
    spatial_z=spatial_z[train_obs_indices], # 只取震前與震後訓練日
    cal_cond=cal_features[train_obs_indices],
    spatial_sigma=spatial_sigma,
    dates=np.array(dates_str)[train_obs_indices]
)

spatial_meta = {
    'grid_w': GRID_W,
    'grid_h': GRID_H,
    'n_channels': N_CHANNELS,
    'channel_names': ['Retention_Diag', 'Outflow', 'Inflow', 'External_Exchange'],
    'od_spatial_map': [{k: v for k, v in item.items() if k not in ['raw_arr', 'b_366']} for item in od_spatial_map]
}

with open(OUT_META, 'wb') as f:
    pickle.dump(spatial_meta, f)

print(f"\n✅  spatial_diffusion_dataset.npz -> {OUT_NPZ}")
print(f"✅  spatial_meta.pkl              -> {OUT_META}")
print("=" * 75)
