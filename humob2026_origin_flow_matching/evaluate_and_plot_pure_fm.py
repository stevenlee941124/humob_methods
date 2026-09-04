"""
===============================================================================
Evaluate and Plot 9-Class Results for Pure Flow Matching
===============================================================================
"""
import sys, csv, pickle, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize if sys.platform != 'win32' else 2147483647)

PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_DATA  = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
OUT_DIR      = PACKAGE_ROOT / 'data' / 'outputs'
ARTIFACT_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\efa61edd-2158-4ece-9fe0-02fe777d67fb")

PRED_TSV     = OUT_DIR / 'origin_fm_predictions.tsv'
RAW_TSV      = SHARED_DATA / 'raw' / 'humob2026-dataset.tsv'
META_PKL     = SHARED_DATA / 'outputs' / 'meta_1476.pkl'
BASELINE_PKL = SHARED_DATA / 'outputs' / 'full_year_baseline.pkl'
OD_PKL       = SHARED_DATA / 'processed' / 'od_time_series.pkl'
DATES_PKL    = SHARED_DATA / 'processed' / 'dates.pkl'

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

with open(META_PKL, 'rb') as f: meta_1476 = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)

# 載入 GT (202404)
gt_data = {}
with open(RAW_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2 and pts[0].startswith('202404') and pts[0] not in EXCLUDED_DATES:
            raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od: gt_data[pts[0]] = od
            except: pass

eval_dates = sorted(list(gt_data.keys()))

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

def in_eval_bbox(g_str):
    if g_str == '-1_-1': return False
    pts = g_str.split('_')
    if len(pts) != 2: return False
    try:
        gx, gy = int(pts[0]), int(pts[1])
        return (30 <= gx <= 70) and (35 <= gy <= 70)
    except:
        return False

route_meta = {}
for r in meta_1476['active_routes']:
    pk = r['pair_key']
    parts = pk.split('-')
    o_str = parts[0]
    d_str = parts[1]
    if in_eval_bbox(o_str) and in_eval_bbox(d_str):
        route_meta[pk] = {
            'class_id': r.get('class_id', 6),
            'class_name': r.get('class_name', 'Class 6'),
            'o_str': o_str,
            'd_str': d_str,
            'is_diag': (o_str == d_str)
        }

valid_grids = [f"{x}_{y}" for x in range(30, 71) for y in range(35, 71)]
all_diag_pairs = [(g, g) for g in valid_grids]

# 計算對角線
diag_records = []
for o_str, d_str in all_diag_pairs:
    pk = f"{o_str}-{d_str}"
    meta = route_meta.get(pk, {'class_id': 6, 'is_diag': True})
    is_active = pk in route_meta
    sq_errs = []
    for d_str_cur in eval_dates:
        yt = gt_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        yp = pred_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        sq_errs.append((yt - yp) ** 2)
    diag_records.append({
        'pair_key': pk,
        'class_id': meta['class_id'],
        'is_active': is_active,
        'mse': np.mean(sq_errs)
    })
df_diag = pd.DataFrame(diag_records)

# 計算非對角線
all_off_pairs = set()
for d_str_cur in eval_dates:
    for o_str, d_dict in gt_data[d_str_cur].items():
        if in_eval_bbox(o_str):
            for d_dest in d_dict:
                if in_eval_bbox(d_dest) and o_str != d_dest:
                    all_off_pairs.add((o_str, d_dest))
    for o_str, d_dict in pred_data.get(d_str_cur, {}).items():
        if in_eval_bbox(o_str):
            for d_dest in d_dict:
                if in_eval_bbox(d_dest) and o_str != d_dest:
                    all_off_pairs.add((o_str, d_dest))

TOTAL_OFF_PAIRS = 1476 * 1475
off_records = []
for o_str, d_str in all_off_pairs:
    pk = f"{o_str}-{d_str}"
    meta = route_meta.get(pk, {'class_id': 1, 'is_diag': False})
    is_active = pk in route_meta
    sq_errs = []
    for d_str_cur in eval_dates:
        yt = gt_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        yp = pred_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        sq_errs.append((yt - yp) ** 2)
    off_records.append({
        'pair_key': pk,
        'class_id': meta['class_id'],
        'is_active': is_active,
        'mse': np.mean(sq_errs)
    })
df_off = pd.DataFrame(off_records)

CLASS_NAMES = {
    1: 'Persistent Zero',
    2: 'Temporary Increase',
    3: 'Persistent Decrease',
    4: 'Partial Recovery',
    5: 'Fully Recovered',
    6: 'Stable Inflow',
    7: 'Emergent/Temporary activity',
    8: 'Partial Dissipation',
    9: 'Persistent Increase'
}

n_zero_off_unobserved = TOTAL_OFF_PAIRS - len(all_off_pairs)

# ── 1. 全島 2,178,576 條 (有加入 0 流量) ──
print("=" * 110)
print("【Pure Flow Matching】有加入0流量的路線 (全島 2,178,576 條)：")
print("=" * 110)
t1 = []
for c in range(1, 10):
    sub_d = df_diag[df_diag['class_id'] == c]
    sub_o = df_off[df_off['class_id'] == c]
    
    n_d = len(sub_d)
    n_o = len(sub_o) + (n_zero_off_unobserved if c == 1 else 0)
    n_tot = n_d + n_o
    if n_tot == 0: continue
    
    rmse_d = np.sqrt(sub_d['mse'].mean()) if len(sub_d) > 0 else 0.0
    nrmse_d = rmse_d / MEAN_ACTUAL_DIAG if len(sub_d) > 0 else 0.0
    
    if c == 1:
        mse_o = sub_o['mse'].sum() / n_o if n_o > 0 else 0.0
    else:
        mse_o = sub_o['mse'].mean() if len(sub_o) > 0 else 0.0
    rmse_o = np.sqrt(mse_o)
    nrmse_o = rmse_o / MEAN_ACTUAL_OFFDIAG if len(sub_o) > 0 else 0.0
    comb = 0.5 * (nrmse_d + nrmse_o)
    
    t1.append({
        'ID': c,
        'Class': CLASS_NAMES[c],
        '數量': f"{n_tot:,} 條",
        'RMSE_diag (人)': f"{rmse_d:.2f}",
        'NRMSE_diag': f"{nrmse_d:.5f}",
        'RMSE_off (人)': f"{rmse_o:.2f}",
        'NRMSE_off': f"{nrmse_o:.4f}",
        'Combined NRMSE': f"{comb:.4f}"
    })

tot_rmse_d = np.sqrt(df_diag['mse'].mean())
tot_nrmse_d = tot_rmse_d / MEAN_ACTUAL_DIAG
tot_mse_o = df_off['mse'].sum() / TOTAL_OFF_PAIRS
tot_rmse_o = np.sqrt(tot_mse_o)
tot_nrmse_o = tot_rmse_o / MEAN_ACTUAL_OFFDIAG
tot_comb = 0.5 * (tot_nrmse_d + tot_nrmse_o)

print(pd.DataFrame(t1).to_string(index=False))
print("-" * 110)
print(f"Total All Routes: {TOTAL_OFF_PAIRS+1476:,} 條 | RMSE_diag: {tot_rmse_d:.2f} | NRMSE_diag: {tot_nrmse_d:.5f} | RMSE_off: {tot_rmse_o:.2f} | NRMSE_off: {tot_nrmse_o:.5f} | Combined: {tot_comb:.5f}\n\n")

# ── 2. 活躍 1,802 條 (沒加入 0 流量) ──
print("=" * 110)
print("【Pure Flow Matching】沒加入0流量的路線 (活躍 1,802 條)：")
print("=" * 110)
active_recs = []
for r in meta_1476['active_routes']:
    pk = r['pair_key']
    parts = pk.split('-')
    o_str, d_str = parts[0], parts[1]
    if in_eval_bbox(o_str) and in_eval_bbox(d_str):
        is_diag = (o_str == d_str)
        sq = []
        for d_str_cur in eval_dates:
            yt = gt_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
            yp = pred_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
            sq.append((yt - yp) ** 2)
        active_recs.append({
            'pair_key': pk,
            'class_id': r.get('class_id', 1),
            'is_diag': is_diag,
            'mse': np.mean(sq)
        })
df_act = pd.DataFrame(active_recs)

t2 = []
for c in range(1, 10):
    sub = df_act[df_act['class_id'] == c]
    sub_d = sub[sub['is_diag']]
    sub_o = sub[~sub['is_diag']]
    n_tot = len(sub)
    if n_tot == 0: continue
    
    rmse_d = np.sqrt(sub_d['mse'].mean()) if len(sub_d) > 0 else 0.0
    nrmse_d = rmse_d / MEAN_ACTUAL_DIAG if len(sub_d) > 0 else 0.0
    
    rmse_o = np.sqrt(sub_o['mse'].mean()) if len(sub_o) > 0 else 0.0
    nrmse_o = rmse_o / MEAN_ACTUAL_OFFDIAG if len(sub_o) > 0 else 0.0
    comb = 0.5 * (nrmse_d + nrmse_o)
    
    t2.append({
        'D': c,
        'Class': CLASS_NAMES[c],
        '數量': f"{n_tot} 條",
        'RMSE_diag (人)': f"{rmse_d:.2f}",
        'NRMSE_diag': f"{nrmse_d:.5f}",
        'RMSE_off (人)': f"{rmse_o:.2f}",
        'NRMSE_off': f"{nrmse_o:.4f}",
        'Combined NRMSE': f"{comb:.4f}"
    })

tot_d_act = df_act[df_act['is_diag']]
tot_o_act = df_act[~df_act['is_diag']]
tot_rmse_d_act = np.sqrt(tot_d_act['mse'].mean())
tot_nrmse_d_act = tot_rmse_d_act / MEAN_ACTUAL_DIAG
tot_rmse_o_act = np.sqrt(tot_o_act['mse'].mean())
tot_nrmse_o_act = tot_rmse_o_act / MEAN_ACTUAL_OFFDIAG
tot_comb_act = 0.5 * (tot_nrmse_d_act + tot_nrmse_o_act)

print(pd.DataFrame(t2).to_string(index=False))
print("-" * 110)
print(f"Total All Active Routes: {len(df_act):,} 條 | RMSE_diag: {tot_rmse_d_act:.2f} | NRMSE_diag: {tot_nrmse_d_act:.5f} | RMSE_off: {tot_rmse_o_act:.2f} | NRMSE_off: {tot_nrmse_o_act:.4f} | Combined: {tot_comb_act:.4f}\n\n")

# ── 3. 繪製高品質 9-Class 時序走勢大圖 (0 流量連通 + 嚴選大流量代表路線) ──
print("🎨 正在繪製高品質 Pure Flow Matching 9-Class 時序預測大圖...", flush=True)
start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']

CLASS_TITLES = {
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

BEST_DIAG_ROUTES = {
    1: '60_42-60_42',  # 典型零流量/微底噪
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
                    
        # 盲區前與盲區後分別局部微小插值保持線條連續 (盲區 2/1~3/31 保持 NaN)
        s_series = pd.Series(y_gt)
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
        
        valid_vals = [v for v in y_gt_connected if not np.isnan(v)]
        mean_v = np.mean(valid_vals) if valid_vals else 0.0
        max_v = np.max(valid_vals) if valid_vals else 0.0
        
        ax.set_title(f"Class {cls_id:02d}: {CLASS_TITLES[cls_id]}\nRoute: {pk} (均值: {mean_v:.1f} 人 | 峰值: {max_v:.1f} 人)", 
                     color='#38bdf8', fontsize=10, fontweight='bold', pad=8)
        
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
    
    artifact_path = ARTIFACT_DIR / out_filename
    import shutil
    shutil.copy(str(save_path), str(artifact_path))
    print(f"✅ 高品質時序大圖已產出並同步: {out_filename}")

render_plot(True,  'pure_fm_9class_diagonal.png')
render_plot(False, 'pure_fm_9class_offdiagonal.png')

