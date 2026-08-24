"""
===============================================================================
HuMob 2026: Step 1 - Build Full-Year Baseline
===============================================================================
流程：
  1. 載入 od_time_series.pkl、dates.pkl
  2. 雙重過濾：全年平均流量 < 2.0 或 4~10月平均流量 < 2.0 直接跳過
  3. 對合格 OD pair 計算全年 366 天四段嚴格連續 Baseline
  4. 輸出：
     - full_year_baseline.pkl   → {pair_key: np.ndarray shape(366,)}
     - gap90_exp_baseline.tsv   → 90天盲區 (2/1~4/30) 的官方格式預測檔
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
from exponential_baseline import compute_full_baseline, FLOW_THRESHOLD

# ── 檔案路徑 ──────────────────────────────────────────────────────
OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_PKL   = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
OUT_TSV   = PACKAGE_ROOT / 'data' / 'outputs' / 'gap90_exp_baseline.tsv'

OUTPUT_THRESHOLD = 0.05  # 輸出到 TSV 的最低值

print("=" * 80)
print(f"[Step 1] Building Full-Year Continuous Baseline (Threshold >= {FLOW_THRESHOLD})")
print("=" * 80)

# ── 載入數據 ──────────────────────────────────────────────────────
with open(OD_PKL, 'rb') as f:
    od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f:
    dates_str = pickle.load(f)

start_dt = datetime.strptime('20231101', '%Y%m%d')
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

gap90_dates = [d for d in cal_dates if '20240201' <= d <= '20240430']

all_baselines = {}
out_tsv = {d: {} for d in gap90_dates}

stats = {"Partial Recovery": 0, "Temporary Increase": 0,
         "Stable": 0, "Sparse (<2.0)": 0}
lambdas = []
total_pairs = len(od_ts)

print(f"Total OD pairs in dataset: {total_pairs:,}")
print(f"Flow threshold: >= {FLOW_THRESHOLD}")
print()

for pair_idx, (pair_key, raw_arr) in enumerate(od_ts.items()):
    y_366 = np.zeros(366, dtype=np.float64)
    # 盲區預設為 0.0 參與全年中軸
    for d_str, o_idx in obs_date_to_idx.items():
        if d_str in cal_date_to_idx:
            val = raw_arr[o_idx]
            y_366[cal_date_to_idx[d_str]] = float(val) if not np.isnan(val) else 0.0

    b_366, lam, grid_type = compute_full_baseline(y_366, cal_dates, cal_date_to_idx)

    if grid_type == "Dead Zero":
        stats["Sparse (<2.0)"] += 1
        continue

    all_baselines[pair_key] = b_366
    parts = pair_key.split('-')
    orig, dest = parts[0], parts[1]
    is_diag = (orig == dest)

    if is_diag:
        stats[grid_type] = stats.get(grid_type, 0) + 1
        lambdas.append(lam)

    # 填入 90 天 TSV 輸出
    for d_str in gap90_dates:
        c_idx = cal_date_to_idx[d_str]
        b_val = float(b_366[c_idx]) if not np.isnan(b_366[c_idx]) else 0.0
        if b_val > OUTPUT_THRESHOLD:
            if orig not in out_tsv[d_str]:
                out_tsv[d_str][orig] = {}
            out_tsv[d_str][orig][dest] = round(b_val, 4)

    if (pair_idx + 1) % 10000 == 0 or (pair_idx + 1) == total_pairs:
        pct = 100.0 * (pair_idx + 1) / total_pairs
        print(f"  [{pair_idx+1:>6}/{total_pairs}]  {pct:.1f}%  |  qualified so far: {len(all_baselines):,}")

# ── 匯出 ─────────────────────────────────────────────────────────
OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PKL, 'wb') as f:
    pickle.dump(all_baselines, f)

with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str in gap90_dates:
        f.write(f"{d_str}\t{out_tsv[d_str]}\n")

print()
print("=" * 80)
print(f"✅  full_year_baseline.pkl  -> {OUT_PKL}")
print(f"✅  gap90_exp_baseline.tsv  -> {OUT_TSV}")
print()
print("📊  Qualifying OD pairs statistics:")
print(f"    Partial Recovery  : {stats.get('Partial Recovery', 0)}")
print(f"    Temporary Increase: {stats.get('Temporary Increase', 0)}")
print(f"    Stable            : {stats.get('Stable', 0)}")
print(f"    Sparse (< {FLOW_THRESHOLD})   : {stats.get('Sparse (<2.0)', 0):,}")
print(f"    Total Qualified   : {len(all_baselines):,}")
print("=" * 80)
