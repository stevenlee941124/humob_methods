"""
===============================================================================
High-Quality Plotting Engine for HuMob 2026:
1. Connects 0-flow points continuously (no gaps/breaks)
2. Filters out near-zero dead routes for Classes 2-9
3. Smooth aesthetic dark UI styling
===============================================================================
"""
import sys, csv, pickle, math, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize if sys.platform != 'win32' else 2147483647)

PACKAGE_ROOT = Path(r"c:\Users\User\Desktop\humob_methods\humob2026_origin_flow_matching")
SHARED_DATA  = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
OUT_DIR      = PACKAGE_ROOT / 'data' / 'outputs'
ARTIFACT_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\efa61edd-2158-4ece-9fe0-02fe777d67fb")

PRED_TSV     = OUT_DIR / 'origin_fm_predictions.tsv'
RAW_TSV      = SHARED_DATA / 'raw' / 'humob2026-dataset.tsv'
META_PKL     = SHARED_DATA / 'outputs' / 'meta_1476.pkl'
BASELINE_PKL = SHARED_DATA / 'outputs' / 'full_year_baseline.pkl'
OD_PKL       = SHARED_DATA / 'processed' / 'od_time_series.pkl'
DATES_PKL    = SHARED_DATA / 'processed' / 'dates.pkl'

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

with open(META_PKL, 'rb') as f: meta_1476 = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)

# 載入 Pred
pred_data = {}
with open(PRED_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2:
            raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od: pred_data[pts[0]] = od
            except: pass

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']

def in_eval_bbox(g_str):
    if g_str == '-1_-1': return False
    pts = g_str.split('_')
    if len(pts) != 2: return False
    try:
        gx, gy = int(pts[0]), int(pts[1])
        return (30 <= gx <= 70) and (35 <= gy <= 70)
    except:
        return False

# 篩選高品質代表路線
CLASS_NAMES = {
    1: 'Persistent Zero (零流量/微底噪)',
    2: 'Temporary Increase (震後避難激增)',
    3: 'Persistent Decrease (嚴重受災驟降)',
    4: 'Partial Recovery (震後緩步爬升)',
    5: 'Fully Recovered (中度震盪迅速復原)',
    6: 'Stable Inflow (全期穩定生活大動脈)',
    7: 'Emergent/Temporary activity (短期湧入後消退)',
    8: 'Partial Dissipation (避難消退部分外流)',
    9: 'Persistent Increase (長期增長新節奏)'
}

# 嚴選代表路線：確保 Class 2~9 均為大流量且連續的主幹路線
BEST_DIAG_ROUTES = {
    1: '60_42-60_42',  # 典型低活/零流量
    2: '39_46-39_46',  # 均值 35.0 人
    3: '58_43-58_43',  # 均值 79.1 人
    4: '58_44-58_44',  # 均值 384.8 人 (能登主動脈)
    5: '41_47-41_47',  # 均值 638.2 人 (金澤大動脈)
    6: '30_69-30_69',  # 均值 593.6 人 (金澤核心樞紐)
    7: '38_43-38_43',  # 均值 105.7 人 (避難湧入)
    8: '36_37-36_37',  # 均值 26.5 人 (消退外流)
    9: '53_37-53_37'   # 均值 39.1 人 (高頻通勤增長)
}

BEST_OFFDIAG_ROUTES = {
    1: '30_36-31_36',  # 零流量典型
    2: '34_70-33_70',  # 跨區均值 4.0 人，峰值 9.8 人
    3: '43_45-43_44',  # 跨區均值 1.5 人，受災陡降
    4: '58_44-58_43',  # 跨區均值 5.5 人，峰值 10.3 人
    5: '41_46-41_47',  # 跨區均值 8.5 人，峰值 12.3 人
    6: '30_69-31_69',  # 跨區均值 12.8 人，峰值 19.5 人
    7: '34_38-34_37',  # 跨區均值 4.7 人，峰值 11.0 人
    8: '61_63-61_62',  # 跨區代表
    9: '31_47-31_48'   # 跨區均值 1.9 人，峰值 3.1 人
}

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def render_plot(is_diagonal: bool, out_filename: str):
    fig, axes = plt.subplots(3, 3, figsize=(22, 14), dpi=160)
    fig.patch.set_facecolor('#0b0f19')
    
    route_dict = BEST_DIAG_ROUTES if is_diagonal else BEST_OFFDIAG_ROUTES
    
    for cls_id in range(1, 10):
        row = (cls_id - 1) // 3
        col = (cls_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#131b2e')
        ax.grid(True, color='#23324d', linestyle='--', alpha=0.6)
        
        pk = route_dict[cls_id]
        parts = pk.split('-')
        o_s, d_s = parts[0], parts[1]
        raw_ts = od_ts.get(pk)
        base = baselines.get(pk)
        
        # 1. 建立連貫的 Ground Truth 曲線 (0 流量也如實相連)
        # 策略：在有效觀測期內，NaN 填 0.0；官方除外異常日做局部內插相連；盲區 (2/1~3/31) 設為 NaN 顯示空缺
        y_gt = np.full(366, np.nan, dtype=np.float32)
        valid_obs_dates = set(dates_str) - EXCLUDED_DATES
        
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
                    
        # 局部微小插值（只針對單日被排除的異常日，例如 11/26、12/14 等，保持線條連續，盲區不插值）
        s_series = pd.Series(y_gt)
        # 只插值盲區外的天數
        idx_feb01 = cal_date_to_idx['20240201']
        idx_mar31 = cal_date_to_idx['20240331']
        
        s_pre = s_series.iloc[:idx_feb01].interpolate(method='linear', limit=3)
        s_post = s_series.iloc[idx_mar31+1:].interpolate(method='linear', limit=3)
        y_gt_connected = np.full(366, np.nan, dtype=np.float32)
        y_gt_connected[:idx_feb01] = s_pre.values
        y_gt_connected[idx_mar31+1:] = s_post.values
        
        # 2. 物理 Baseline (366 天完整連續)
        y_base = np.array(base, dtype=np.float32) if (base is not None and isinstance(base, (list, np.ndarray))) else np.zeros(366, dtype=np.float32)
        
        # 3. 預測值 (在 90 天盲區內 100% 連續，0 流量也是 0.0，決不斷裂！)
        y_pred = np.full(366, np.nan, dtype=np.float32)
        for d_str_cur in blind_zone:
            c_idx = cal_date_to_idx[d_str_cur]
            val = pred_data.get(d_str_cur, {}).get(o_s, {}).get(d_s, 0.0) or 0.0
            y_pred[c_idx] = float(val)
            
        x_axis = np.arange(366)
        
        # 繪製曲線
        ax.plot(x_axis, y_gt_connected, color='#f43f5e', alpha=0.85, linewidth=1.4, label='Ground Truth (含0流量連通)')
        ax.plot(x_axis, y_base, color='#fbbf24', linestyle='--', linewidth=1.6, alpha=0.85, label='Physical Baseline')
        
        b_start = cal_date_to_idx['20240201']
        b_end   = cal_date_to_idx['20240430']
        ax.plot(x_axis[b_start:b_end+1], y_pred[b_start:b_end+1], color='#10b981', linewidth=2.2, label='Pure Flow Matching (連續預測)')
        ax.axvspan(b_start, b_end, color='#0ea5e9', alpha=0.12, label='Blind Zone (官方盲區)')
        
        # 計算均值與標題
        valid_vals = [v for v in y_gt_connected if not np.isnan(v)]
        mean_v = np.mean(valid_vals) if valid_vals else 0.0
        max_v = np.max(valid_vals) if valid_vals else 0.0
        
        ax.set_title(f"Class {cls_id:02d}: {CLASS_NAMES[cls_id]}\nRoute: {pk} (均值: {mean_v:.1f} 人 | 峰值: {max_v:.1f} 人)", 
                     color='#38bdf8', fontsize=10, fontweight='bold', pad=8)
        
        # X 軸標籤
        tick_pos = [cal_date_to_idx[d] for d in ['20231101', '20240101', '20240201', '20240301', '20240401', '20240501'] if d in cal_date_to_idx]
        tick_lbl = ['11/01', '01/01 (震災)', '02/01', '03/01', '04/01', '05/01']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=8, color='#94a3b8')
        ax.tick_params(colors='#64748b')
        
        if row == 0 and col == 0:
            ax.legend(loc='upper right', fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
            
    mode_str = "Diagonal (Stay)" if is_diagonal else "Off-Diagonal (Cross Flow)"
    fig.suptitle(f"{mode_str} - 9-Class Pure Flow Matching vs. Physical Baseline (0-Flow Connected & Clean Main Routes)", 
                 color='#f8fafc', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    save_path = OUT_DIR / out_filename
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # 同步至 Artifact
    artifact_path = ARTIFACT_DIR / out_filename
    shutil.copy(str(save_path), str(artifact_path))
    print(f"✅ 高品質時序大圖已產出: {out_filename}")

render_plot(True,  'pure_fm_9class_diagonal.png')
render_plot(False, 'pure_fm_9class_offdiagonal.png')
