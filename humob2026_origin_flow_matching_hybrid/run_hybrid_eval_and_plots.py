import sys, csv, pickle, math, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize if sys.platform != 'win32' else 2147483647)

PACKAGE_ROOT = Path(r"c:\Users\User\Desktop\humob_methods\humob2026_origin_flow_matching_hybrid")
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from evaluator import evaluate_predictions

SHARED_DATA  = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
FM_ORIGIN_DATA = PACKAGE_ROOT.parent / 'humob2026_origin_flow_matching' / 'data' / 'outputs'
OUT_DIR      = PACKAGE_ROOT / 'data' / 'outputs'
ARTIFACT_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\efa61edd-2158-4ece-9fe0-02fe777d67fb")

PRED_TSV       = OUT_DIR / 'origin_hybrid_fm_predictions.tsv'
RAW_TSV        = SHARED_DATA / 'raw' / 'humob2026-dataset.tsv'
OUT_SUBMISSION = OUT_DIR / 'origin_hybrid_submission_official.tsv'
VALIDATOR      = PACKAGE_ROOT / 'humob2026_validator.py'

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

# ── 1. 官方評估指標 ───────────────────────────────────────────────────────────
print("=" * 80)
print("📊 1/4 正在計算 Hybrid Flow Matching 官方評測分數...")
print("=" * 80)
scores = evaluate_predictions(str(RAW_TSV), str(PRED_TSV))
for k, v in scores.items():
    print(f"  • {k:<25}: {v:.5f}")

# ── 2. 官方提交檔導出與驗證 ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📦 2/4 正在導出官方提交檔案 (20240201 ~ 20240331)...")
OFFICIAL_EXCLUDE = {'20240202', '20240305'}
with open(PRED_TSV, 'r', encoding='utf-8') as fin, open(OUT_SUBMISSION, 'w', encoding='utf-8') as fout:
    for line in fin:
        pts = line.strip().split('\t')
        if len(pts) == 2 and '20240201' <= pts[0] <= '20240331' and pts[0] not in OFFICIAL_EXCLUDE:
            fout.write(f"{pts[0]}\t{pts[1]}\n")

if VALIDATOR.exists():
    print("🔍 正在執行官方 validator 驗證...")
    res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUBMISSION)], capture_output=True, text=True)
    print("Validator 輸出:")
    print(res.stdout)
    if res.returncode == 0:
        print(f"🎉 官方提交檔案 100% 通過驗證！檔案位置 → {OUT_SUBMISSION}")
    else:
        print("❌ 驗證失敗:", res.stderr)

# ── 3. 計算 9-Class 評估表格 (嚴格對齊用戶格式) ──────────────────────────────────
print("\n" + "=" * 80)
print("📋 3/4 正在產出 9-Class 評估表格...")
print("=" * 80)

with open(META_PKL,     'rb') as f: meta_1476 = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
with open(OD_PKL,       'rb') as f: od_ts     = pickle.load(f)
with open(DATES_PKL,    'rb') as f: dates_str = pickle.load(f)

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
N_EVAL_DAYS = len(eval_dates)

# 載入 Predictions (全部盲區 90 天)
pred_data = {}
with open(PRED_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2:
            d_str_k = pts[0].strip()
            raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od: pred_data[d_str_k] = od
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
    pair_key = r['pair_key']
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
    
    if in_eval_bbox(o_str) and in_eval_bbox(d_str):
        cls_id = r.get('class_id', 6)
        route_meta[pair_key] = {
            'class_id': cls_id,
            'o_str': o_str,
            'd_str': d_str,
            'is_diag': (o_str == d_str)
        }

valid_grids = [f"{x}_{y}" for x in range(30, 71) for y in range(35, 71)]
all_diag_pairs = [(g, g) for g in valid_grids]
all_off_pairs = set()

for d_str in eval_dates:
    for o_str, d_dict in gt_data[d_str].items():
        if in_eval_bbox(o_str):
            for d_dest in d_dict:
                if in_eval_bbox(d_dest) and o_str != d_dest:
                    all_off_pairs.add((o_str, d_dest))
    for o_str, d_dict in pred_data.get(d_str, {}).items():
        if in_eval_bbox(o_str):
            for d_dest in d_dict:
                if in_eval_bbox(d_dest) and o_str != d_dest:
                    all_off_pairs.add((o_str, d_dest))

diag_records = []
for o_str, d_str in all_diag_pairs:
    pair_key = f"{o_str}-{d_str}"
    meta = route_meta.get(pair_key, {'class_id': 6})
    
    sq_errs = []
    for d_str_cur in eval_dates:
        y_true = gt_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        y_pred = pred_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        sq_errs.append((y_true - y_pred) ** 2)
    
    mse = np.mean(sq_errs)
    diag_records.append({
        'pair_key': pair_key,
        'class_id': meta['class_id'],
        'is_active': pair_key in route_meta,
        'mse': mse
    })

TOTAL_OFF_PAIRS = 1476 * 1475
TOTAL_ALL_PAIRS = TOTAL_OFF_PAIRS + 1476

off_records = []
for o_str, d_str in all_off_pairs:
    pair_key = f"{o_str}-{d_str}"
    meta = route_meta.get(pair_key, {'class_id': 1})
    
    sq_errs = []
    for d_str_cur in eval_dates:
        y_true = gt_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        y_pred = pred_data.get(d_str_cur, {}).get(o_str, {}).get(d_str, 0.0) or 0.0
        sq_errs.append((y_true - y_pred) ** 2)
    
    mse = np.mean(sq_errs)
    off_records.append({
        'pair_key': pair_key,
        'class_id': meta['class_id'],
        'is_active': pair_key in route_meta,
        'mse': mse
    })

n_zero_off_unobserved = TOTAL_OFF_PAIRS - len(all_off_pairs)
df_diag = pd.DataFrame(diag_records)
df_off  = pd.DataFrame(off_records)

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

# --- 表 1: 有加入0流量的路線 ---
t1_rows = []
for c_id in range(1, 10):
    c_name = CLASS_NAMES[c_id]
    sub_d = df_diag[df_diag['class_id'] == c_id]
    sub_o = df_off[df_off['class_id'] == c_id]
    
    n_d = len(sub_d)
    n_o = len(sub_o) + (n_zero_off_unobserved if c_id == 1 else 0)
    n_tot = n_d + n_o
    
    mse_d = sub_d['mse'].mean() if len(sub_d) > 0 else 0.0
    rmse_d = np.sqrt(mse_d)
    nrmse_d = rmse_d / MEAN_ACTUAL_DIAG if len(sub_d) > 0 else 0.0
    
    if c_id == 1:
        sum_sq_o = sub_o['mse'].sum() if len(sub_o) > 0 else 0.0
        mse_o = sum_sq_o / n_o if n_o > 0 else 0.0
    else:
        mse_o = sub_o['mse'].mean() if len(sub_o) > 0 else 0.0
        
    rmse_o = np.sqrt(mse_o)
    nrmse_o = rmse_o / MEAN_ACTUAL_OFFDIAG if len(sub_o) > 0 else 0.0
    comb = 0.5 * (nrmse_d + nrmse_o)
    
    t1_rows.append({
        'ID': c_id,
        'Class': c_name,
        '數量': f"{n_tot:,} 條",
        'RMSE_diag (人)': f"{rmse_d:.2f}",
        'NRMSE_diag': f"{nrmse_d:.5f}",
        'RMSE_off (人)': f"{rmse_o:.2f}",
        'NRMSE_off': f"{nrmse_o:.4f}",
        'Combined NRMSE': f"{comb:.4f}"
    })

mse_diag_tot = df_diag['mse'].mean()
rmse_diag_tot = np.sqrt(mse_diag_tot)
nrmse_diag_tot = rmse_diag_tot / MEAN_ACTUAL_DIAG

mse_off_tot = df_off['mse'].sum() / TOTAL_OFF_PAIRS
rmse_off_tot = np.sqrt(mse_off_tot)
nrmse_off_tot = rmse_off_tot / MEAN_ACTUAL_OFFDIAG

comb_tot = 0.5 * (nrmse_diag_tot + nrmse_off_tot)

print("\n有加入0流量的路線：\n")
print(pd.DataFrame(t1_rows).to_string(index=False))
print("-" * 105)
print(f"Total All Routes: {TOTAL_ALL_PAIRS:,} 條 | RMSE_diag: {rmse_diag_tot:.2f} | NRMSE_diag: {nrmse_diag_tot:.5f} | RMSE_off: {rmse_off_tot:.2f} | NRMSE_off: {nrmse_off_tot:.5f} | Combined: {comb_tot:.5f}")

# --- 表 2: 沒加入0流量的路線 ---
t2_rows = []
for c_id in range(1, 10):
    c_name = CLASS_NAMES[c_id]
    if c_id == 1:
        c_name = 'Persistent Zero (微量底噪)'
        
    sub_d = df_diag[(df_diag['class_id'] == c_id) & df_diag['is_active']]
    sub_o = df_off[(df_off['class_id'] == c_id) & df_off['is_active']]
    
    n_d = len(sub_d)
    n_o = len(sub_o)
    n_tot = n_d + n_o
    
    if n_tot == 0:
        continue
        
    mse_d = sub_d['mse'].mean() if len(sub_d) > 0 else 0.0
    rmse_d = np.sqrt(mse_d)
    nrmse_d = rmse_d / MEAN_ACTUAL_DIAG if len(sub_d) > 0 else 0.0
    
    mse_o = sub_o['mse'].mean() if len(sub_o) > 0 else 0.0
    rmse_o = np.sqrt(mse_o)
    nrmse_o = rmse_o / MEAN_ACTUAL_OFFDIAG if len(sub_o) > 0 else 0.0
    comb = 0.5 * (nrmse_d + nrmse_o)
    
    t2_rows.append({
        'D': c_id,
        'Class': c_name,
        '數量': f"{n_tot:,} 條",
        'RMSE_diag (人)': f"{rmse_d:.2f}",
        'NRMSE_diag': f"{nrmse_d:.5f}",
        'RMSE_off (人)': f"{rmse_o:.2f}",
        'NRMSE_off': f"{nrmse_o:.4f}",
        'Combined NRMSE': f"{comb:.4f}"
    })

df_diag_act = df_diag[df_diag['is_active']]
df_off_act = df_off[df_off['is_active']]
tot_act_cnt = len(df_diag_act) + len(df_off_act)

mse_d_act = df_diag_act['mse'].mean()
rmse_d_act = np.sqrt(mse_d_act)
nrmse_d_act = rmse_d_act / MEAN_ACTUAL_DIAG

mse_o_act = df_off_act['mse'].mean()
rmse_o_act = np.sqrt(mse_o_act)
nrmse_o_act = rmse_o_act / MEAN_ACTUAL_OFFDIAG
comb_act = 0.5 * (nrmse_d_act + nrmse_o_act)

print("\n\n沒加入0流量的路線：\n")
print(pd.DataFrame(t2_rows).to_string(index=False))
print("-" * 105)
print(f"Total All Active Routes: {tot_act_cnt:,} 條 | RMSE_diag: {rmse_d_act:.2f} | NRMSE_diag: {nrmse_d_act:.5f} | RMSE_off: {rmse_o_act:.2f} | NRMSE_off: {nrmse_o_act:.4f} | Combined: {comb_act:.4f}")

# ── 4. 繪製 9-Class 時序大圖 (含三條曲線) ──────────────────────────────────────
print("\n" + "=" * 80)
print("🎨 4/4 正在繪製 9-Class 時序預測大圖...")
print("=" * 80)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']

diag_routes_by_cls = {c: [] for c in range(1, 10)}
offdiag_routes_by_cls = {c: [] for c in range(1, 10)}

for r in meta_1476['active_routes']:
    pk = r['pair_key']
    cls_id = r.get('class_id', 6)
    o, d = pk.split('-')[0], pk.split('-')[1]
    is_diag = (o == d)
    if is_diag:
        diag_routes_by_cls[cls_id].append(r)
    else:
        offdiag_routes_by_cls[cls_id].append(r)

CLASS_LABELS = {
    1: "Class 01: Low Activity (低活性/雜訊)",
    2: "Class 02: Sudden Plunge & Slow (劇烈暴跌緩慢復甦)",
    3: "Class 03: Fast Bounce (快速反彈型)",
    4: "Class 04: Linear Recovery (階梯線性爬升)",
    5: "Class 05: Mid Shock (中度震盪微幅受創)",
    6: "Class 06: Stable Unaffected (全期穩定無受災)",
    7: "Class 07: Evacuee Influx Surge (避難湧入先升後降)",
    8: "Class 08: Partial Dissipation (避難消退部分外流)",
    9: "Class 09: Dynamic Commuter (高頻穩定通勤)"
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

def render_9class_grid(is_diag_flag, save_png_path, title_prefix):
    fig, axes = plt.subplots(3, 3, figsize=(22, 14), dpi=160)
    fig.patch.set_facecolor('#0b0f19')
    
    route_dict = BEST_DIAG_ROUTES if is_diag_flag else BEST_OFFDIAG_ROUTES
    
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
        
        # 1. Ground Truth (含 0 流量連通)
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
                    
        # 盲區外做局部微小插值保持線條連續 (盲區 2/1~3/31 保持 NaN)
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
        
        # 3. 預測值 (在 90 天盲區內 100% 連續)
        y_pred = np.full(366, np.nan, dtype=np.float32)
        for d_str_cur in blind_zone:
            c_idx = cal_date_to_idx[d_str_cur]
            val = pred_data.get(d_str_cur, {}).get(o_s, {}).get(d_s, 0.0) or 0.0
            y_pred[c_idx] = float(val)
            
        x_axis = np.arange(366)
        
        ax.plot(x_axis, y_gt_connected, color='#f43f5e', alpha=0.85, linewidth=1.4, label='Ground Truth (含0流量連通)')
        ax.plot(x_axis, y_base, color='#fbbf24', linestyle='--', linewidth=1.6, alpha=0.85, label='Physical Baseline')
        
        b_start = cal_date_to_idx['20240201']
        b_end   = cal_date_to_idx['20240430']
        ax.plot(x_axis[b_start:b_end+1], y_pred[b_start:b_end+1], color='#10b981', linewidth=2.2, label='Hybrid Flow Matching (連續預測)')
        ax.axvspan(b_start, b_end, color='#0ea5e9', alpha=0.12, label='Blind Zone (官方盲區)')
        
        valid_vals = [v for v in y_gt_connected if not np.isnan(v)]
        mean_v = np.mean(valid_vals) if valid_vals else 0.0
        max_v = np.max(valid_vals) if valid_vals else 0.0
        
        ax.set_title(f"{CLASS_LABELS[cls_id]}\nRoute: {pk} (均值: {mean_v:.1f} 人 | 峰值: {max_v:.1f} 人)", 
                     color='#38bdf8', fontsize=10, fontweight='bold', pad=8)
        
        tick_pos = [cal_date_to_idx[d] for d in ['20231101', '20240101', '20240201', '20240301', '20240401', '20240501'] if d in cal_date_to_idx]
        tick_lbl = ['11/01', '01/01 (震災)', '02/01', '03/01', '04/01', '05/01']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=8, color='#94a3b8')
        ax.tick_params(colors='#64748b')
        
        if row == 0 and col == 0:
            ax.legend(loc='upper right', fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
            
    fig.suptitle(f"{title_prefix} - 9-Class Hybrid Flow Matching vs. Baseline (0-Flow Connected & Clean Main Routes)", 
                 color='#f8fafc', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✅ 圖表已產出: {save_png_path}")

diag_png = OUT_DIR / 'hybrid_9class_diagonal.png'
offdiag_png = OUT_DIR / 'hybrid_9class_offdiagonal.png'

render_9class_grid(True, diag_png, "Diagonal (Stay)")
render_9class_grid(False, offdiag_png, "Off-Diagonal (Cross Flow)")

shutil.copy(diag_png, ARTIFACT_DIR / 'hybrid_9class_diagonal.png')
shutil.copy(offdiag_png, ARTIFACT_DIR / 'hybrid_9class_offdiagonal.png')
print("✅ 圖表已成功同步至 Artifact 目錄！")
