"""
===============================================================================
HuMob 2026: Step 2 Spatial - Build 2D Spatial-Temporal Grid Tensor Dataset
===============================================================================
閰喟敦?詨飛璅∪??撠?隢??梧?DIFFUSION_MATHEMATICAL_MODEL.md

鞈?蝯?蝭? (Data Structure Examples)嚗?  - 頛詨 (od_time_series.pkl): 
    {
      '31_38-31_38': array([12.0, 15.0, np.nan, ...]),
      ...
    }
  - 頛詨 (spatial_diffusion_dataset.npz):
    ? 'spatial_z' ??? (N, 4, 70, 100)嚗誨銵冽?憭?4 ????2D 蝛粹??孵噩嚗?憒?
    Z[0, 0, 30, 37] = 0.5 (隞?”蝚砌?憭押?0?漣璅=30, y=37 ??皞?畾榆)

頛詨:
  - od_time_series.pkl (14,563 璇?OD 頝舐???, 292憭?
  - dates.pkl (292憭拇????
  - full_year_baseline.pkl (?典僑銝剛遘 Baseline)
??:
  1. 蝛粹?蝬脫摨扳???: X ??[1, 70], Y ??[1, 100] -> (70, 100) 蝛粹?蝬脫
  2. 瑽遣瘥 4 ?? 2D 蝛粹?撘菟? (Channels: Retention, Outflow, Inflow, External Exchange)
  3. 閮?蝛粹?銝剛遘 Baseline 撘菟? B_spatial(t) ?征??撌格?皞榆 ?_spatial
  4. 頛詨璅??征??撌桀撐??Z_spatial(t) ??R^(4, 70, 100)
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
from nine_class_baseline import compute_9class_baseline

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

print("甇?閮??典? 15,129 璇?OD 頝舐?銋?4 ??撖阡摰???Baseline...")
baselines = {}
for k, raw_arr in od_ts.items():
    y_366 = np.zeros(366, dtype=np.float64)
    for d_str, oi in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            v = raw_arr[oi]
            y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0
    b_366, _, _ = compute_9class_baseline(y_366, cal_dates, cal_date_to_idx)
    baselines[k] = b_366

BASELINE_PKL.parent.mkdir(parents=True, exist_ok=True)
with open(BASELINE_PKL, 'wb') as f:
    pickle.dump(baselines, f)
print(f"??撌脫???蝞蒂?脣? 4 ?摰???Baseline ?? {BASELINE_PKL}")

# ?? 1. 閫?????OD 頝舐?銋?(ox, oy) -> (dx, dy) 蝛粹?摨扳? ??????
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

print(f"??閫?? {len(od_spatial_map):,} 璇?OD 頝舐??征???脣???靽?")

# ?? 2. 瑽遣瘥 (292, 4, 70, 100) ?征?撐???????????????????
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
        
        # Channel 0: ??瘚? (Retention)
        if is_diag and o_c is not None:
            spatial_y[t_idx, 0, o_c[0], o_c[1]] += val_y
            spatial_b[t_idx, 0, o_c[0], o_c[1]] += val_b
            if has_y: spatial_obs_mask[t_idx, 0, o_c[0], o_c[1]] = True
            
        # ?折頝典?瘚? (Cross-grid Flow)
        elif not is_ext and not is_diag:
            # Channel 1: ?折頝典?瘚 (Outflow) at Origin
            if o_c is not None:
                spatial_y[t_idx, 1, o_c[0], o_c[1]] += val_y
                spatial_b[t_idx, 1, o_c[0], o_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 1, o_c[0], o_c[1]] = True
            # Channel 2: ?折頝典?瘚 (Inflow) at Destination
            if d_c is not None:
                spatial_y[t_idx, 2, d_c[0], d_c[1]] += val_y
                spatial_b[t_idx, 2, d_c[0], d_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 2, d_c[0], d_c[1]] = True
            
        # Channel 3: 憭?鈭箸?鈭斗? (External Exchange)
        elif is_ext:
            target_c = d_c if o_c is None else o_c
            if target_c is not None:
                spatial_y[t_idx, 3, target_c[0], target_c[1]] += val_y
                spatial_b[t_idx, 3, target_c[0], target_c[1]] += val_b
                if has_y: spatial_obs_mask[t_idx, 3, target_c[0], target_c[1]] = True

print(f"??瘥 4 ??蝛粹?瘚??湔?撱箏??? 敶Ｙ? {spatial_y.shape}")

# ?? 3. 閮?蝛粹?畾榆璅?撌??_spatial (4, 70, 100) ??????????????
train_obs_indices = [i for i, d in enumerate(dates_str) if (d < '20240101' or d >= '20240501')]

spatial_residuals = spatial_y[train_obs_indices] - spatial_b[train_obs_indices]
spatial_sigma = np.std(spatial_residuals, axis=0) # (4, 70, 100)
# ?詨潛帘摰扳??spatial_sigma = np.maximum(spatial_sigma, 0.1)

# 閮?璅??征??撌桀撐??Z(t) = (Y - B) / ?
spatial_z = (spatial_y - spatial_b) / spatial_sigma

print(f"??蝛粹?璅?撌桀?撅??_spatial 閮?摰?: ??{np.mean(spatial_sigma):.3f}, ?憭?{np.max(spatial_sigma):.3f}")
print(f"??蝛粹?璅???撌?Z_spatial ??: ??{np.mean(spatial_z):.3f}, 璅?撌?{np.std(spatial_z):.3f}")

# ?? 4. 瑽遣?交??孵噩?? (292, 4) ??????????????????????????????
cal_features = np.zeros((N_OBS, 4), dtype=np.float32)
for i, d_str in enumerate(dates_str):
    dt = datetime.strptime(d_str, '%Y%m%d')
    cal_features[i, 0] = dt.weekday() / 6.0
    cal_features[i, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    cal_features[i, 2] = (dt.month - 1) / 11.0
    cal_features[i, 3] = cal_date_to_idx[d_str] / 365.0

# ?? 5. ?脣?蝛粹?鞈?????????????????????????????????????????????
np.savez_compressed(
    str(OUT_NPZ),
    spatial_z=spatial_z[train_obs_indices], # ?芸?????敺?蝺湔
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

print(f"\n?? spatial_diffusion_dataset.npz -> {OUT_NPZ}")
print(f"?? spatial_meta.pkl              -> {OUT_META}")
print("=" * 75)

