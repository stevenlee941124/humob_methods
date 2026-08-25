"""
===============================================================================
HuMob 2026 Hybrid: Step 2 - Build Layer 1 Baseline & Layer 2 Gated Carrier psi
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from baseline_module import compute_full_baseline, FLOW_THRESHOLD
from periodic_gating import compute_regularity_and_carrier
from japan_calendar import JAPAN_HOLIDAYS

OD_PKL  = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL  = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_base_and_gates.pkl'

print("=" * 80)
print("[Step 2] Building Layer 1 (C^1 Smooth Baseline) & Layer 2 (Adaptive Gated psi)...")
print("=" * 80)

with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

hybrid_models = {}
stats = {'high_regularity': 0, 'medium_regularity': 0, 'low_regularity': 0, 'dead_zero': 0}

total_pairs = len(od_ts)

for pair_idx, (pair_key, raw_arr) in enumerate(od_ts.items()):
    y_366 = np.zeros(366, dtype=np.float64)
    for d_str, oi in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            v = raw_arr[oi]
            y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0

    b_366, lam, grid_type = compute_full_baseline(y_366, cal_dates, cal_date_to_idx)
    if grid_type == "Dead Zero":
        stats['dead_zero'] += 1
        continue

    # Layer 2: 規律度量化與自適應門控
    s_reg, gate_g, carrier_func, sigma_weekly = compute_regularity_and_carrier(y_366, b_366, cal_dates, cal_dts)

    if s_reg >= 0.45:
        stats['high_regularity'] += 1
        category = 'High_Regularity_Commuter'
    elif s_reg >= 0.20:
        stats['medium_regularity'] += 1
        category = 'Medium_Regularity_Corridor'
    else:
        stats['low_regularity'] += 1
        category = 'Low_Regularity_Noise'

    hybrid_models[pair_key] = {
        'b_366': b_366,
        's_reg': s_reg,
        'gate_g': gate_g,
        'carrier_func': carrier_func,
        'sigma_weekly': sigma_weekly,
        'category': category
    }

    if (pair_idx + 1) % 5000 == 0 or (pair_idx + 1) == total_pairs:
        print(f"  [{pair_idx+1:>5}/{total_pairs}] Processed | Qualified models: {len(hybrid_models):,}")

OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PKL, 'wb') as f:
    pickle.dump(hybrid_models, f)

print()
print("=" * 80)
print(f"✅ Saved hybrid baseline and gating models to: {OUT_PKL}")
print(f"📊 Statistics:")
print(f"  • High Regularity (S_reg ≥ 0.45, Full 7D ψ enabled): {stats['high_regularity']:,} pairs")
print(f"  • Medium Regularity (0.20 ≤ S_reg < 0.45, Gated ψ) : {stats['medium_regularity']:,} pairs")
print(f"  • Low Regularity (S_reg < 0.20, ψ Suppressed to 0) : {stats['low_regularity']:,} pairs")
print(f"  • Dead Zero Isolated routes                        : {stats['dead_zero']:,} pairs")
print("=" * 80)
