"""
===============================================================================
HuMob 2026 Hybrid: Step 3 - Build 2D Spatial Residual Tensor Dataset
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from japan_calendar import JAPAN_HOLIDAYS

OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
BASE_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_base_and_gates.pkl'
OUT_NPZ   = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_dataset.npz'
OUT_META  = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_meta.pkl'

print("=" * 80)
print("[Step 3] Projecting (Raw - Baseline - Gated_psi) Residuals to 2D Spatial Tensor...")
print("=" * 80)

with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)
with open(BASE_PKL, 'rb') as f: hybrid_models = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

N_OBS_DAYS = len(dates_str)  # 292 or 306
H, W = 70, 100
spatial_tensor = np.zeros((N_OBS_DAYS, 4, H, W), dtype=np.float32)

def split_od_key(pair_key):
    if pair_key.startswith('-1_-1-'):
        o_str = '-1_-1'
        d_str = pair_key[6:]
    elif pair_key.endswith('--1_-1'):
        o_str = pair_key[:-6]
        d_str = '-1_-1'
    else:
        pts = pair_key.split('-')
        o_str, d_str = pts[0], pts[1]
    return o_str, d_str

def parse_grid_coord(grid_str):
    if grid_str == '-1_-1':
        return None
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= H and 1 <= y <= W:
            return (x - 1, y - 1)
    except:
        pass
    return None

od_spatial_map = []
for pair_key in od_ts.keys():
    o_str, d_str = split_od_key(pair_key)
    o_c = parse_grid_coord(o_str)
    d_c = parse_grid_coord(d_str)

    is_diag = (o_str == d_str and o_c is not None)
    is_ext  = (o_str == '-1_-1' or d_str == '-1_-1')

    od_spatial_map.append({
        'pair_key': pair_key,
        'o_str': o_str, 'd_str': d_str,
        'o_coord': o_c, 'd_coord': d_c,
        'is_diag': is_diag, 'is_ext': is_ext
    })

# 逐日計算殘差並投影至空間張量
for day_idx, d_str in enumerate(dates_str):
    ci = cal_date_to_idx.get(d_str)
    if ci is None:
        continue
    dow = cal_dts[ci].weekday()

    for item in od_spatial_map:
        k = item['pair_key']
        raw_val = od_ts[k][day_idx]
        if np.isnan(raw_val):
            continue

        model_info = hybrid_models.get(k)
        if model_info is not None:
            b_val = model_info['b_366'][ci]
            g_val = model_info['gate_g']
            c_func = model_info['carrier_func']
            sig_w = model_info['sigma_weekly']
            psi_val = float(c_func(14.0 + dow)) if c_func is not None else 0.0
            det_pred = max(0.0, b_val + g_val * psi_val * sig_w)
            resid = float(raw_val - det_pred)
        else:
            resid = float(raw_val)

        # 投影至 4 通道
        o_c, d_c = item['o_coord'], item['d_coord']
        if item['is_diag'] and o_c is not None:
            spatial_tensor[day_idx, 0, o_c[0], o_c[1]] += resid
        elif not item['is_ext'] and o_c is not None and d_c is not None:
            spatial_tensor[day_idx, 1, o_c[0], o_c[1]] += resid
            spatial_tensor[day_idx, 2, d_c[0], d_c[1]] += resid
        elif item['is_ext']:
            target_c = d_c if o_c is None else o_c
            if target_c is not None:
                spatial_tensor[day_idx, 3, target_c[0], target_c[1]] += resid

# 計算空間標準差圖層 σ_spatial
sigma_spatial = np.std(spatial_tensor, axis=0) # (4, H, W)
sigma_spatial = np.maximum(0.1, sigma_spatial)

# 標準化空間殘差
z_spatial = spatial_tensor / (sigma_spatial[None, :, :, :] + 1e-4)

# 構建條件特徵 (Day of week, Holiday, Month, Day of year)
cond_features = np.zeros((N_OBS_DAYS, 4), dtype=np.float32)
for i, d_str in enumerate(dates_str):
    ci = cal_date_to_idx.get(d_str, 0)
    dt = datetime.strptime(d_str, '%Y%m%d')
    cond_features[i, 0] = dt.weekday() / 6.0
    cond_features[i, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    cond_features[i, 2] = (dt.month - 1) / 11.0
    cond_features[i, 3] = ci / 365.0

OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    str(OUT_NPZ),
    z_spatial=z_spatial.astype(np.float32),
    cond_features=cond_features,
    sigma_spatial=sigma_spatial.astype(np.float32)
)

with open(OUT_META, 'wb') as f:
    pickle.dump({
        'od_spatial_map': od_spatial_map,
        'H': H, 'W': W,
        'dates_str': dates_str
    }, f)

print(f"✅ 2D Spatial Residual Tensor built: shape {z_spatial.shape}")
print(f"✅ σ_spatial Map stats: Mean = {np.mean(sigma_spatial):.3f}, Max = {np.max(sigma_spatial):.3f}")
print(f"✅ Saved to: {OUT_NPZ} and {OUT_META}")
print("=" * 80)
