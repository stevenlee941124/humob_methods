"""
===============================================================================
HuMob 2026: Step 4c - Hybrid Flow Matching Sampling with 60-Day Anchor Gate
===============================================================================
雙軌混成解碼 (Hybrid Dual-Track Decoding):
  1. 60 天雙錨點接續門控 Gate_60(t):
     - 左錨點 (1月底最後有數據): 提取 1/15~1/31 之振幅恢復率 G_start
     - 60 天過渡 (2/1~3/31): 平滑過渡至 1.0 (tau_60 = j / 59)
     - 右錨點 (4/1起已有數據): Gate 保持 1.0，無縫對齊真實觀測
  2. 對角線 (Stay) : 全面注入 7 天通勤波形 psi_7(t) * Gate_60(t) * W_psi
  3. 非對角線 (Cross): 零底噪路線 (reg<=0.05 & mean<=1.20) 歸零過濾
                    高規律跨區路線 (reg>=0.25) 注入適度 psi_7
                    其餘採用純淨 Flow Matching 空間向量場
===============================================================================
"""
import sys
import gc
import math
import pickle
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from origin_flow_matching import OriginDestFlowUNet, OriginFlowMatching
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

SHARED_DATA     = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
FM_ORIGIN_DATA  = PACKAGE_ROOT.parent / 'humob2026_origin_flow_matching' / 'data' / 'outputs'

CHECKPOINT_EP5 = FM_ORIGIN_DATA / 'origin_fm_checkpoint_ep5.pt'
CHECKPOINT     = CHECKPOINT_EP5 if CHECKPOINT_EP5.exists() else (FM_ORIGIN_DATA / 'origin_fm_checkpoint.pt')
META_PKL     = FM_ORIGIN_DATA / 'origin_fm_meta.pkl'
DEST_META_PKL= SHARED_DATA / 'outputs' / 'meta_1476.pkl'
BASELINE_PKL = SHARED_DATA / 'outputs' / 'full_year_baseline.pkl'
OD_PKL       = SHARED_DATA / 'processed' / 'od_time_series.pkl'
DATES_PKL    = SHARED_DATA / 'processed' / 'dates.pkl'
RAW_TSV      = SHARED_DATA / 'raw' / 'humob2026-dataset.tsv'

OUT_DIR      = PACKAGE_ROOT / 'data' / 'outputs'
OUT_TSV      = OUT_DIR / 'origin_hybrid_fm_predictions.tsv'
OUT_SUBMISSION = OUT_DIR / 'origin_hybrid_submission_official.tsv'
VALIDATOR    = PACKAGE_ROOT / 'humob2026_validator.py'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_ORIGINS = 4
ODE_STEPS = 20

print("=" * 80)
print(f"[Step 4c] Hybrid Flow Matching with 60-Day Two-Sided Anchor Gate on {DEVICE.upper()}")
print("=" * 80)

# 日曆索引建立
start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
N_BLIND = len(blind_zone) # 90 days
blind_wds = [datetime.strptime(d, '%Y%m%d').weekday() for d in blind_zone]
blind_idxs = np.array([cal_date_to_idx[d] for d in blind_zone], dtype=np.int32)

# 60 天盲區時間進度 (2/1 至 3/31 共 60 天，4/1 起進度為 1.0)
N_GAP_DAYS = 60
tau_60 = np.array([
    min(1.0, float(j) / (N_GAP_DAYS - 1)) if j < N_GAP_DAYS else 1.0
    for j in range(N_BLIND)
], dtype=np.float32)

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def build_cond_8d(d_str: str, ox_norm: float, oy_norm: float) -> np.ndarray:
    dt    = datetime.strptime(d_str, '%Y%m%d')
    wd    = dt.weekday()
    c_idx = cal_date_to_idx.get(d_str, 0)
    tau   = c_idx / 365.0
    return np.array([
        math.sin(2 * math.pi * wd / 7),
        math.cos(2 * math.pi * wd / 7),
        1.0 if d_str in JAPAN_HOLIDAYS else 0.0,
        math.sin(2 * math.pi * tau),
        math.cos(2 * math.pi * tau),
        tau,
        ox_norm,
        oy_norm,
    ], dtype=np.float32)

print("1/4 載入模型與中介資料...", flush=True)
with open(META_PKL,      'rb') as f: meta_fm   = pickle.load(f)
with open(DEST_META_PKL, 'rb') as f: meta_dest = pickle.load(f)
with open(BASELINE_PKL,  'rb') as f: baselines = pickle.load(f)
with open(OD_PKL,        'rb') as f: od_ts     = pickle.load(f)
with open(DATES_PKL,     'rb') as f: dates_str = pickle.load(f)

date_idx_map = {d: i for i, d in enumerate(dates_str)}
normal_pre_dates = [d for d in dates_str if '20231101' <= d < '20231225' and d not in EXCLUDED_DATES]
late_jan_dates   = [d for d in dates_str if '20240115' <= d <= '20240131' and d not in EXCLUDED_DATES]

eval_origins      = meta_fm['eval_origins']
eval_destinations = meta_fm['eval_destinations']
origin_meta_list  = meta_fm['origin_meta_list']
origin_meta_dict  = {m['o_str']: m for m in origin_meta_list}

GRID_W, GRID_H = meta_fm['grid_w'], meta_fm['grid_h']
COND_DIM       = meta_fm['cond_dim']
N_ORIGINS      = len(eval_origins)

def parse_coord(grid_str):
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
            return (x - 1, y - 1)
    except: pass
    return None

dest_coords = {d: parse_coord(d) for d in eval_destinations}
route_class_map = {r['pair_key']: r.get('class_id', 6) for r in meta_dest['active_routes']}

normal_dates = [d for d in dates_str if (d < '20240101' or d >= '20240501')]

print("2/4 提煉各路線 7 天通勤齒波 psi_7 與 60 天接續 Gate...", flush=True)
route_profiles = {}
for pk, raw in od_ts.items():
    if raw is None: continue
    valid_v = [x for x in raw if not np.isnan(x)]
    mean_v  = np.mean(valid_v) if valid_v else 0.0
    p_act   = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
    
    # 7 天週內波形
    weekday_vals = {w: [] for w in range(7)}
    for di, d_str in enumerate(dates_str):
        if d_str in normal_dates and di < len(raw) and not np.isnan(raw[di]):
            wd = datetime.strptime(d_str, '%Y%m%d').weekday()
            weekday_vals[wd].append(raw[di])
            
    wd_means = [np.mean(weekday_vals[w]) if len(weekday_vals[w]) > 0 else mean_v for w in range(7)]
    mean_of_wds = np.mean(wd_means)
    psi_7 = np.array([wd_means[w] - mean_of_wds for w in range(7)], dtype=np.float32)
    
    # 34 週規律度
    date_val_map = {dates_str[oi]: raw[oi] for oi in range(len(raw)) if oi < len(dates_str) and not np.isnan(raw[oi])}
    mondays = [d for d in normal_dates if datetime.strptime(d, '%Y%m%d').weekday() == 0]
    weeks_list = []
    for m_str in mondays:
        m_dt = datetime.strptime(m_str, '%Y%m%d')
        w_vals = []
        for di in range(7):
            cur_d = (m_dt + timedelta(days=di)).strftime('%Y%m%d')
            if cur_d in date_val_map:
                w_vals.append(date_val_map[cur_d])
        if len(w_vals) == 7 and np.std(w_vals) > 1e-5:
            w_arr = np.array(w_vals, dtype=np.float32)
            weeks_list.append((w_arr - np.mean(w_arr)) / (np.std(w_arr) + 1e-6))
            
    if len(weeks_list) >= 2:
        norm_matrix = np.stack(weeks_list, axis=0)
        var_per_day = np.var(norm_matrix, axis=0)
        mean_var = np.mean(var_per_day)
        regularity = float(np.clip(1.0 - mean_var / 1.0, 0.0, 1.0))
    else:
        regularity = 0.0
        
    # 計算 1 月底真實恢復起點 G_start
    pre_vals = [raw[date_idx_map[d]] for d in normal_pre_dates if date_idx_map[d] < len(raw) and not np.isnan(raw[date_idx_map[d]])]
    late_vals = [raw[date_idx_map[d]] for d in late_jan_dates if date_idx_map[d] < len(raw) and not np.isnan(raw[date_idx_map[d]])]
    
    std_pre = np.std(pre_vals) if len(pre_vals) >= 7 else 0.0
    std_late = np.std(late_vals) if len(late_vals) >= 5 else 0.0
    mean_pre = np.mean(pre_vals) if pre_vals else 0.0
    mean_late = np.mean(late_vals) if late_vals else 0.0
    
    if std_pre > 0.5:
        g_start = float(np.clip(std_late / std_pre, 0.25, 1.15))
    elif mean_pre > 1.0:
        g_start = float(np.clip(mean_late / mean_pre, 0.25, 1.15))
    else:
        g_start = 1.0
        
    # 60 天無縫接續門控曲線
    gate_60 = g_start + (1.0 - g_start) * (tau_60 ** 1.5)
        
    route_profiles[pk] = {
        'psi_7': psi_7,
        'regularity': regularity,
        'mean_v': mean_v,
        'p_act': p_act,
        'g_start': g_start,
        'gate_60': gate_60
    }

print(f"✅ 成功提煉 {len(route_profiles):,} 條路線之 60 天接續特徵 (平均 G_start: {np.mean([p['g_start'] for p in route_profiles.values()]):.2f})")

# 載入 Flow Matching 模型
model = OriginDestFlowUNet(cond_dim=COND_DIM, base_ch=32, time_dim=128).to(DEVICE)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"✅ 載入 Flow Matching Checkpoint: {CHECKPOINT.name} (Epoch: {ckpt.get('epoch')}, Loss: {ckpt.get('best_loss'):.6f})")

print("3/4 開始 Per-Origin Flow Matching 採樣與 60 天雙錨點混成解碼...", flush=True)
origin_batches = [eval_origins[i: i + BATCH_ORIGINS] for i in range(0, N_ORIGINS, BATCH_ORIGINS)]
output_rows = {d: {} for d in blind_zone}
n_written = 0

for b_idx, batch_origins in enumerate(origin_batches):
    B_curr = len(batch_origins)
    total_maps = B_curr * N_BLIND
    
    cond_batch = np.zeros((total_maps, COND_DIM), dtype=np.float32)
    idx = 0
    for o_str in batch_origins:
        om = origin_meta_dict.get(o_str)
        ox_norm = om['ox_norm'] if om else 0.0
        oy_norm = om['oy_norm'] if om else 0.0
        for d_str in blind_zone:
            cond_batch[idx] = build_cond_8d(d_str, ox_norm, oy_norm)
            idx += 1
            
    cond_batch_t = torch.from_numpy(cond_batch).to(DEVICE)
    shape = (total_maps, 1, GRID_W, GRID_H)
    
    with torch.inference_mode():
        z_pred_t = OriginFlowMatching.sample(model, cond_batch_t, shape, DEVICE, n_steps=ODE_STEPS)
        z_pred = z_pred_t.cpu().numpy()
        del cond_batch_t, z_pred_t
        torch.cuda.empty_cache()
        
    for oi, o_str in enumerate(batch_origins):
        om = origin_meta_dict.get(o_str)
        if om is None: continue
        
        z_o = z_pred[oi * N_BLIND: (oi + 1) * N_BLIND]
        sigma_o = om['sigma']
        
        for d_str_k in eval_destinations:
            pair_key = f"{o_str}-{d_str_k}"
            d_coord  = dest_coords.get(d_str_k)
            if d_coord is None: continue
            dx, dy = d_coord
            
            is_diag = (o_str == d_str_k)
            cls_id  = route_class_map.get(pair_key, 6)
            b_366   = baselines.get(pair_key)
            prof    = route_profiles.get(pair_key, {'psi_7': np.zeros(7), 'regularity': 0.0, 'mean_v': 0.0, 'p_act': 0.0, 'gate_60': np.ones(N_BLIND)})
            
            reg    = prof['regularity']
            mean_v = prof['mean_v']
            p_act  = prof['p_act']
            
            # ── 1. 底噪過濾 (Zero Pruning) ──────────────────────────────────
            if reg <= 0.05 and mean_v <= 1.20:
                continue
            if mean_v < 0.15 and p_act < 0.15:
                continue
                
            base_90 = np.array(b_366[blind_idxs], dtype=np.float32) if (b_366 is not None and isinstance(b_366, (list, np.ndarray))) else np.zeros(N_BLIND, dtype=np.float32)
                
            # ── 2. 取出 Flow Matching 空間殘差 ─────────────────────────────
            z_raw   = z_o[:, 0, dx, dy]
            sig_val = float(sigma_o[0, dx, dy])
            z_std   = np.std(z_raw)
            if z_std > 1e-6:
                z_norm = (z_raw - np.mean(z_raw)) / z_std
            else:
                z_norm = np.zeros_like(z_raw)
            z_norm = np.clip(z_norm, -2.5, 2.5)
            
            # ── 3. 取出 60 天雙錨點接續 Gate ──────────────────────────────
            gate_curve = prof['gate_60']
            psi_90     = np.array([prof['psi_7'][w] for w in blind_wds], dtype=np.float32)
            
            # ── 4. 雙軌混成還原公式 (先加總 psi 與 FM 再整體乘上 Gate) ───
            if is_diag:
                # 對角線自身停留：(psi + Flow Matching) 先加再整體乘 Gate
                w_psi  = 0.50 + 0.50 * reg
                w_diff = max(0.10, 1.0 - 0.70 * reg)
                resid_total = (psi_90 * w_psi + z_norm * sig_val * 0.35 * w_diff) * gate_curve
                y_pred = np.maximum(0.0, base_90 + resid_total)
            else:
                # 非對角線跨區流動：
                if reg >= 0.25:
                    resid_total = (psi_90 * 0.40 + z_norm * sig_val * 0.35) * gate_curve
                else:
                    resid_total = (z_norm * sig_val * 0.35) * gate_curve
                y_pred = np.maximum(0.0, base_90 + resid_total)
                    
                # 非對角線微量噪聲截斷
                if np.mean(y_pred) < 0.10:
                    continue
                    
            for j, d_date in enumerate(blind_zone):
                val = float(y_pred[j])
                if val > 0.05:
                    if o_str not in output_rows[d_date]:
                        output_rows[d_date][o_str] = {}
                    output_rows[d_date][o_str][d_str_k] = round(val, 4)
                    n_written += 1

    if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(origin_batches):
        print(f"  批次 {b_idx+1}/{len(origin_batches)} ({((b_idx+1)/len(origin_batches)*100):.1f}%) | 已寫入有效點: {n_written:,}", flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        f.write(f"{d_str_cur}\t{output_rows[d_str_cur]}\n")

print(f"\n✅ 預測完畢！共寫入有效預測值 {n_written:,} 筆")
print(f"✅ 輸出 TSV → {OUT_TSV}")

# ── 4/4 評估結果 ─────────────────────────────────────────────────────────────
if RAW_TSV.exists():
    scores = evaluate_predictions(str(RAW_TSV), str(OUT_TSV))
    print("\n" + "=" * 80)
    print("📊 混成流匹配 (Hybrid Flow Matching) 評估結果:")
    print("=" * 80)
    for k, v in scores.items():
        print(f"  • {k:<25}: {v:.5f}")

# ── 導出官方提交檔並執行官方驗證 ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("📦 正在導出官方提交檔案 (20240201 ~ 20240331)...")
OFFICIAL_EXCLUDE = {'20240202', '20240305'}
with open(OUT_TSV, 'r', encoding='utf-8') as fin, open(OUT_SUBMISSION, 'w', encoding='utf-8') as fout:
    for line in fin:
        pts = line.strip().split('\t')
        if len(pts) == 2 and '20240201' <= pts[0] <= '20240331' and pts[0] not in OFFICIAL_EXCLUDE:
            fout.write(f"{pts[0]}\t{pts[1]}\n")

if VALIDATOR.exists():
    import subprocess
    print("🔍 正在執行官方 validator 驗證...")
    res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUBMISSION)], capture_output=True, text=True)
    print("Validator 輸出:")
    print(res.stdout)
    if res.returncode == 0:
        print(f"🎉 官方提交檔案 100% 通過驗證！檔案位置 → {OUT_SUBMISSION}")
    else:
        print("❌ 驗證失敗:", res.stderr)
print("=" * 80)
