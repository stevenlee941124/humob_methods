"""
===============================================================================
HuMob 2026: Diagnostic Plotting for Adaptive Regularity Diffusion (V2)
===============================================================================
Visualizes full-year Ground Truth, Physical Baseline, and Adaptive V2 Predictions
===============================================================================
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_DATA_DIR = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
DATA_DIR = PACKAGE_ROOT / 'data'
ARTIFACT_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\efa61edd-2158-4ece-9fe0-02fe777d67fb")

def get_data_path(rel_path):
    local_p = DATA_DIR / rel_path
    if local_p.exists(): return local_p
    shared_p = SHARED_DATA_DIR / rel_path
    if shared_p.exists(): return shared_p
    raise FileNotFoundError(f"Cannot find {rel_path}")

OD_PKL       = get_data_path('processed/od_time_series.pkl')
DATES_PKL    = get_data_path('processed/dates.pkl')
BASELINE_PKL = get_data_path('outputs/full_year_baseline.pkl')
PRED_TSV     = DATA_DIR / 'outputs' / 'dest1476_predictions_adaptive_v2.tsv'
OUT_PNG      = DATA_DIR / 'outputs' / 'adaptive_v2_routes_diagnostic.png'

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']

with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)

preds = {}
with open(PRED_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2:
            d_str = pts[0]
            raw_s = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw_s, {'__builtins__': {}}, {'None': None})
                preds[d_str] = od
            except: pass

TARGET_ROUTES = [
    ('41_47-41_47', '41_47-41_47 (Class 5 金澤核心大動脈)', 25.83, 633.7),
    ('34_70-34_70', '34_70-34_70 (Class 6 金澤南生活樞紐)', 19.52, 243.5),
    ('58_44-58_44', '58_44-58_44 (Class 4 能登震災重建核心)', 27.07, 350.4),
    ('30_69-30_69', '30_69-30_69 (Class 6 內灘北交通樞紐)', 28.57, 583.9)
]

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(22, 13), dpi=160)
axes = axes.flatten()

for idx, (pk, title_info, rmse_val, gt_mean) in enumerate(TARGET_ROUTES):
    ax = axes[idx]
    o_s, d_s = pk.split('-')
    raw_ts = od_ts.get(pk)
    base = baselines.get(pk)

    y_gt = np.full(366, np.nan, dtype=np.float32)
    for i, d_str_cur in enumerate(dates_str):
        c_idx = cal_date_to_idx.get(d_str_cur)
        if c_idx is not None and raw_ts is not None and i < len(raw_ts):
            val = raw_ts[i]
            if d_str_cur in EXCLUDED_DATES:
                y_gt[c_idx] = np.nan
            elif np.isnan(val) or val <= 0.0:
                y_gt[c_idx] = 0.0
            else:
                y_gt[c_idx] = float(val)

    s_series = pd.Series(y_gt)
    idx_feb01 = cal_date_to_idx['20240201']
    idx_mar31 = cal_date_to_idx['20240331']

    s_pre = s_series.iloc[:idx_feb01].interpolate(method='linear', limit=3)
    s_post = s_series.iloc[idx_mar31+1:].interpolate(method='linear', limit=3)
    y_gt_connected = np.full(366, np.nan, dtype=np.float32)
    y_gt_connected[:idx_feb01] = s_pre.values
    y_gt_connected[idx_mar31+1:] = s_post.values

    y_base = np.array(base, dtype=np.float32) if (base is not None and isinstance(base, (list, np.ndarray))) else np.zeros(366, dtype=np.float32)

    y_pred = np.full(366, np.nan, dtype=np.float32)
    for d_str_cur in blind_zone:
        c_idx = cal_date_to_idx[d_str_cur]
        y_pred[c_idx] = float(preds.get(d_str_cur, {}).get(o_s, {}).get(d_s, 0.0) or 0.0)

    x_axis = np.arange(366)

    ax.plot(x_axis, y_gt_connected, color='#f43f5e', alpha=0.85, linewidth=1.6, label='Ground Truth (真實官方人流)')
    ax.plot(x_axis, y_base, color='#fbbf24', linestyle='--', linewidth=1.4, alpha=0.75, label='Physical Baseline (物理基準線)')

    b_start = cal_date_to_idx['20240201']
    b_end   = cal_date_to_idx['20240430']
    ax.plot(x_axis[b_start:b_end+1], y_pred[b_start:b_end+1], color='#10b981', linewidth=2.4, label='Adaptive V2 (自適應雙錨點門控預測)')
    ax.axvspan(b_start, b_end, color='#0284c7', alpha=0.08, label='Blind Zone (官方盲區 2/1~4/30)')

    ax.set_title(f'{title_info}\n[ 4月評測 RMSE: {rmse_val:.2f} 人 | 真實均值: {gt_mean:.1f} 人 ]', 
                 color='#a7f3d0', fontsize=11, fontweight='bold', pad=8)

    tick_pos = [cal_date_to_idx[d] for d in ['20231101', '20240101', '20240201', '20240301', '20240401', '20240501'] if d in cal_date_to_idx]
    tick_lbl = ['11/01', '01/01', '02/01', '03/01', '04/01', '05/01']
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=9, color='#94a3b8')
    ax.tick_params(colors='#64748b')
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8.5, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')

fig.suptitle('Adaptive Regularity Diffusion (V2 Dual-Anchor) 關鍵核心路線診斷圖', 
             fontsize=16, fontweight='bold', color='#f8fafc', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=160)
plt.savefig(ARTIFACT_DIR / 'adaptive_v2_routes_diagnostic.png', bbox_inches='tight', dpi=160)
plt.close()
print(f"✅ 成功生成診斷圖: {OUT_PNG}")
