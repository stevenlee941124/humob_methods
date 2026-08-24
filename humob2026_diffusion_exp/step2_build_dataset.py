"""
===============================================================================
HuMob 2026: Step 2 - Classification by Zero-Traffic Day Ratio (0人流日期比例)
===============================================================================
全新以「0人流日期比例 (Zero-Traffic Ratio)」為核心的分類體系：
  1. Group A (常態連續流通, 0值天數 < 35% 且 4月有值 >= 15天, 共 ~2,956 條):
     - 包括各級連續流動路線 (如 10_23-10_21, 41_47-41_47)
     - 全面計算精確 sigma，享受 1.0x 1D Diffusion 7D 週期波型
  2. Group B (高0值率偶發走廊, 0值天數 35%~75%, 如 11_28, 10_27, 共 ~220 條):
     - 具備高零值率特徵，中軸按期望值 (B * P_act) 下沉貼地，波谷歸零
  3. Group C (極度死寂孤島, 0值天數 > 75% 且 4月 < 5天, 共 ~11,387 條):
     - 直接置 0.0
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from exponential_baseline import EXCLUDED_DATES
from japan_calendar import JAPAN_HOLIDAYS

OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs'   / 'full_year_baseline.pkl'

OUT_NPZ      = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_train_dataset.npz'
OUT_SIGMA    = PACKAGE_ROOT / 'data' / 'outputs' / 'od_sigma_map.pkl'
OUT_PROFILES = PACKAGE_ROOT / 'data' / 'outputs' / 'od_profiles.pkl'
OUT_META     = PACKAGE_ROOT / 'data' / 'outputs' / 'od_train_meta.pkl'

WINDOW_LEN = 14
STRIDE     = 7

TRAIN_RANGES = [
    ('20231101', '20231231'),
    ('20240501', '20241031'),
]

with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}
apr_indices = [i for i, d in enumerate(dates_str) if d.startswith('202404')]

# ── 1. 全域路線以「0人流日期比例」精準分類 ──────────────────────
od_profiles = {}
profile_stats = {'Group_A_Continuous_Diffusion': 0, 'Group_B_ZeroInflated_Downshift': 0, 'Group_C_Dead_Zero': 0}

for pair_key, raw_arr in od_ts.items():
    n_pos = np.sum(~np.isnan(raw_arr) & (raw_arr > 0.05))
    zero_ratio = 1.0 - (n_pos / 292.0)
    
    apr_arr = raw_arr[apr_indices]
    apr_pos = np.sum(~np.isnan(apr_arr) & (apr_arr > 0.05))
    
    if pair_key not in baselines or (zero_ratio >= 0.75 and apr_pos < 5):
        prof = 'Group_C_Dead_Zero'
    elif zero_ratio < 0.35 and apr_pos >= 15:
        prof = 'Group_A_Continuous_Diffusion'
    else:
        prof = 'Group_B_ZeroInflated_Downshift'
        
    od_profiles[pair_key] = prof
    profile_stats[prof] += 1

print("=" * 75)
print(f"📊 全域 14,563 條 OD 路線『0人流日期比例』分類結果：")
print(f"  • Group A (常態連續流通, 0值天數 < 35% - 全面 Diffusion 波動) : {profile_stats['Group_A_Continuous_Diffusion']:,} 條")
print(f"  • Group B (高0值率偶發走廊, 0值天數 35%~75% - 期望值中軸貼地)  : {profile_stats['Group_B_ZeroInflated_Downshift']:,} 條")
print(f"  • Group C (極度死寂孤島, 0值天數 > 75% - 直接置 0.0)         : {profile_stats['Group_C_Dead_Zero']:,} 條")
print("=" * 75)

# ── 2. 為所有 Group A 路線計算實測殘差標準差 sigma 與訓練窗口 ────
cal_features = np.zeros((366, 4), dtype=np.float32)
for i, d_str in enumerate(cal_dates):
    dt = datetime.strptime(d_str, '%Y%m%d')
    cal_features[i, 0] = dt.weekday() / 6.0
    cal_features[i, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    cal_features[i, 2] = (dt.month - 1) / 11.0
    cal_features[i, 3] = i / 365.0

train_cal_indices = set()
for lo, hi in TRAIN_RANGES:
    for i, d in enumerate(cal_dates):
        if lo <= d <= hi and d not in EXCLUDED_DATES:
            train_cal_indices.add(i)
train_cal_indices_sorted = sorted(train_cal_indices)

all_windows    = []
all_conditions = []
od_sigma_map   = {}
od_train_meta  = []

group_a_keys = [k for k, p in od_profiles.items() if p == 'Group_A_Continuous_Diffusion']
print(f"開始為所有 {len(group_a_keys):,} 條 Group A 活躍路線計算精確殘差標準差 sigma...")

for pair_idx, pair_key in enumerate(group_a_keys):
    raw_arr = od_ts[pair_key]
    b_366   = baselines[pair_key]

    y_366 = np.full(366, np.nan, dtype=np.float64)
    for d_str, oi in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            y_366[cal_date_to_idx[d_str]] = raw_arr[oi]

    resid_vals = []
    for ci in train_cal_indices_sorted:
        y_val = y_366[ci]
        b_val = b_366[ci]
        if not np.isnan(y_val) and not np.isnan(b_val) and b_val > 0:
            resid_vals.append(float(y_val - b_val))

    if len(resid_vals) >= 10:
        sigma = float(np.std(resid_vals))
    else:
        valid_resids = [float(y_366[ci] - b_366[ci]) for ci in range(366) if not np.isnan(y_366[ci]) and not np.isnan(b_366[ci])]
        sigma = float(np.std(valid_resids)) if len(valid_resids) >= 5 else 0.5

    if sigma < 1e-3:
        sigma = 1e-3
    od_sigma_map[pair_key] = sigma

    z_366 = np.full(366, np.nan, dtype=np.float32)
    for ci in range(366):
        y_val = y_366[ci]
        b_val = b_366[ci]
        if not np.isnan(y_val) and not np.isnan(b_val):
            z_366[ci] = float(y_val - b_val) / sigma

    pair_windows_added = 0
    for start_ci in range(0, len(train_cal_indices_sorted) - WINDOW_LEN + 1, STRIDE):
        window_indices = train_cal_indices_sorted[start_ci: start_ci + WINDOW_LEN]

        if window_indices[-1] - window_indices[0] != WINDOW_LEN - 1:
            continue

        z_window = z_366[window_indices[0]: window_indices[-1] + 1]
        if np.sum(~np.isnan(z_window)) < WINDOW_LEN * 0.85:
            continue

        if np.any(np.isnan(z_window)):
            x = np.arange(WINDOW_LEN)
            valid = ~np.isnan(z_window)
            z_window = np.interp(x, x[valid], z_window[valid]).astype(np.float32)

        cond_window = cal_features[window_indices[0]: window_indices[-1] + 1]

        all_windows.append(z_window)
        all_conditions.append(cond_window)
        pair_windows_added += 1

    od_train_meta.append({
        'pair_key': pair_key,
        'sigma': sigma,
        'n_windows': pair_windows_added,
    })

windows_arr    = np.stack(all_windows,    axis=0)
conditions_arr = np.stack(all_conditions, axis=0)

print(f"✅ Group A 路線總數      : {len(od_sigma_map):,} 條 (全部享有精確 sigma!)")
print(f"✅ 提取高品質訓練窗口數 : {len(windows_arr):,} 個")
print(f"✅ 窗口形狀              : {windows_arr.shape}")

np.savez_compressed(str(OUT_NPZ), windows=windows_arr, conditions=conditions_arr)
with open(OUT_SIGMA, 'wb')    as f: pickle.dump(od_sigma_map, f)
with open(OUT_PROFILES, 'wb') as f: pickle.dump(od_profiles, f)
with open(OUT_META, 'wb')     as f: pickle.dump(od_train_meta, f)

print(f"\n✅  diffusion_train_dataset.npz  -> {OUT_NPZ}")
print(f"✅  od_profiles.pkl             -> {OUT_PROFILES}")
print(f"✅  od_sigma_map.pkl            -> {OUT_SIGMA}")
