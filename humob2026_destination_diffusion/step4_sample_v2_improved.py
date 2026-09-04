"""
===============================================================================
HuMob 2026: Step 4 - Improved ψ + Recovery Gate + Regularity-Adaptive Weights
===============================================================================
改進點（vs step4_sample_1476_diffusion.py）：
  1. Recovery Gate：τ² power law，模擬「震後人流由小到大恢復」
  2. 規律性自動調權：ψ_scale 依路線的週間規律性動態調整（0.3~0.85）
     → 高規律路線：重 ψ，輕 diffusion（防止 diffusion 暴走）
     → 低規律路線：重 diffusion，輕 ψ
  3. 輸出儲存為不同檔名，方便直接對比分數
===============================================================================
"""
import sys, pickle, torch, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_1476_checkpoint.pt'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_TSV      = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions_v2.tsv'  # 新檔名

# ── Recovery Gate 設定 ────────────────────────────────────────────────────────
# 盲區 = 2024/02/01 ~ 04/30（2024 能登半島地震後的恢復期）
# gate(τ) = τ^GATE_ALPHA：α>1 → 慢起快收，符合「震幅由小到大」
GATE_ALPHA = 2.0   # 可調：1.0=線性, 2.0=二次, 0.5=快起慢收

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
N_BLIND = len(blind_zone)

# ── 60 天雙錨點連續接續門控設定 ────────────────────────────────────────────────
# 盲區缺漏共 60 天 (2024/02/01 ~ 03/31: 29天 + 31天 = 60天)
# 4/01 起已有真實觀測，門控維持 1.00
N_GAP_DAYS = 60
tau_60 = np.array([
    min(1.0, float(j) / (N_GAP_DAYS - 1)) if j < N_GAP_DAYS else 1.0
    for j in range(N_BLIND)
], dtype=np.float32)

with open(META_PKL,     'rb') as f: meta_1476 = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
with open(OD_PKL,       'rb') as f: od_ts     = pickle.load(f)
with open(DATES_PKL,    'rb') as f: dates_str = pickle.load(f)

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

DISASTER_PERTURBED_CLASSES = {2, 3, 4, 5, 7, 8}

late_jan_dates = [d for d in dates_str if '20240115' <= d <= '20240131' and d not in EXCLUDED_DATES]
normal_pre_dates = [d for d in dates_str if '20231101' <= d < '20231225' and d not in EXCLUDED_DATES]
date_idx_map = {d: i for i, d in enumerate(dates_str)}

CACHE_Z_FILE = PACKAGE_ROOT / 'data' / 'outputs' / 'z_pred_cache.npy'

if CACHE_Z_FILE.exists():
    print(f"⚡ 載入現有 DDIM 預測快取: {CACHE_Z_FILE}", flush=True)
    z_pred_all = np.load(CACHE_Z_FILE)
else:
    model = MultiChannelSpatialUNet(in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

    print("正在執行 DDIM 採樣...", flush=True)
    cond_tensor = torch.tensor(blind_cond, device=DEVICE)

    z_pred_list = []
    batch_days = 15
    with torch.no_grad():
        for start_b in range(0, N_BLIND, batch_days):
            end_b = min(N_BLIND, start_b + batch_days)
            sub_shape = (end_b - start_b, 1476, 70, 100)
            sub_cond  = cond_tensor[start_b:end_b]
            sub_z     = ddpm.ddim_sample(model, sub_shape, c_cond=sub_cond, n_steps=50).cpu()
            z_pred_list.append(sub_z)

    z_pred_all = torch.cat(z_pred_list, dim=0).numpy()
    np.save(CACHE_Z_FILE, z_pred_all)

train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101']
output_rows = {d: {} for d in blind_zone}
n_written = 0
active_keys = set()

for r in meta_1476['active_routes']:
    pair_key = r['pair_key']
    active_keys.add(pair_key)
    parts = pair_key.split('-')
    o_str    = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str    = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]

    c_idx = r['c_idx']
    ox, oy = r['ox'], r['oy']
    sig_i  = r['sigma_i']
    b_366  = baselines.get(pair_key)
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
        continue
    base_90 = np.copy(b_366[blind_idxs])

    raw     = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v  = np.mean(valid_v) if valid_v else 0.0
    p_act   = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0

    cls_id = r.get('class_id', 6)
    if cls_id == 1 or (mean_v < 0.10 and p_act < 0.10):
        continue

    # ── 🌟 改進 1：萃取歷史 ψ₇ 與 34 週常態期波形疊加重合度 ──────────────────
    pre_obs = [(dates_str[oi], raw[oi])
               for oi in train_days_idx
               if oi < len(raw) and not np.isnan(raw[oi])]

    psi_comp  = np.zeros(N_BLIND)
    psi_scale = 0.30
    regularity = 0.0

    if len(pre_obs) >= 14 and mean_v >= 1.0:
        overall_m = np.mean([v for _, v in pre_obs])
        wd_map = {w: [] for w in range(7)}
        for d_str_k, v in pre_obs:
            wd_map[datetime.strptime(d_str_k, '%Y%m%d').weekday()].append(v)

        psi_7 = np.array([
            (np.mean(wd_map[w]) - overall_m) if wd_map[w] else 0.0
            for w in range(7)
        ])
    else:
        psi_7 = np.zeros(7, dtype=np.float32)

    # 34 週常態期 (11~12月 + 5~10月) 波形重合離散度
    date_val_map = {dates_str[oi]: raw[oi] for oi in range(len(raw)) if oi < len(dates_str) and not np.isnan(raw[oi])}
    normal_dates = [d for d in dates_str if (d < '20240101' or d >= '20240501')]
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
        # 🌟 正確理論無偏規律度公式：1.0 - mean_var
        regularity = float(np.clip(1.0 - mean_var / 1.0, 0.0, 1.0))
    else:
        regularity = 0.0

    # 路線獨立 Z-Score 標準化（防 U-Net 整體偏移）
    z_i_raw = z_pred_all[:, c_idx, ox, oy]
    z_std   = np.std(z_i_raw)
    if z_std > 1e-6:
        z_i = (z_i_raw - np.mean(z_i_raw)) / z_std
    else:
        z_i = np.zeros_like(z_i_raw)
    z_i = np.clip(z_i, -2.5, 2.5)

    is_diag = (o_str == d_str)

    # 🌟 60 天雙錨點自適應接續門控 (1月底無縫接續至4月)
    if cls_id in DISASTER_PERTURBED_CLASSES:
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
        cur_gate = g_start + (1.0 - g_start) * (tau_60 ** 1.5)
    else:
        cur_gate = np.ones(N_BLIND, dtype=np.float32)

    # ── 🌟 改進 2 & 4：雙軌四象限自適應融合解碼 ──────────────────────────
    if not is_diag:
        # 【非對角線 (跨區流動)】：適度放寬底噪門檻至 reg<=0.05 & mean<=1.20，保留實質低流量跨區路線 (如 Class 8)
        if regularity <= 0.05 and mean_v <= 1.20:
            y_pred = np.zeros(N_BLIND, dtype=np.float32)
        else:
            # 高規律跨區通勤：先加總再整體乘上 cur_gate
            diff_w = max(0.0, 1.0 - regularity / 0.25) if regularity < 0.25 else 0.0
            psi_sc = 0.50 + 0.60 * regularity
            psi_raw = np.array([psi_7[cal_dts[blind_idxs[j]].weekday()] * psi_sc for j in range(N_BLIND)], dtype=np.float32)
            resid_total = (psi_raw + z_i * (sig_i * 0.35 * diff_w)) * cur_gate
            y_pred = np.maximum(0.0, base_90 + resid_total)
            # 動態微量過濾
            y_pred[y_pred < 0.05] = 0.0
    else:
        # 【對角線 (自身停留)】：分母 26.57，大流量波形保護
        if regularity <= 0.05 and mean_v <= 1.20:
            y_pred = np.zeros(N_BLIND, dtype=np.float32)
        else:
            # 核心樞紐停留：先加總 (ψ7 + Diffusion) 再整體乘上 cur_gate
            psi_sc = 0.35 + 0.85 * regularity
            diff_w = max(0.05, 1.0 - regularity * 0.90)
            psi_raw = np.array([psi_7[cal_dts[blind_idxs[j]].weekday()] * psi_sc for j in range(N_BLIND)], dtype=np.float32)
            resid_total = (psi_raw + z_i * (sig_i * 0.40 * diff_w)) * cur_gate
            y_pred = np.maximum(0.0, base_90 + resid_total)

    for j, d_str_cur in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]:
                output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

def parse_od_pair(pair_key):
    if pair_key.startswith('-1_-1--1_-1'):
        return '-1_-1', '-1_-1'
    elif pair_key.startswith('-1_-1-'):
        return '-1_-1', pair_key[6:]
    elif pair_key.endswith('--1_-1'):
        return pair_key[:-6], '-1_-1'
    else:
        parts = pair_key.split('-')
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, None

# Non-active routes：純 Baseline
for pair_key, b_366 in baselines.items():
    if pair_key in active_keys: continue
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)): continue
    raw     = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v  = np.mean(valid_v) if valid_v else 0.0
    p_act   = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
    if mean_v < 0.25 or p_act < 0.20: continue
    
    o_str, d_str = parse_od_pair(pair_key)
    if not o_str or not d_str: continue

    base_90 = b_366[blind_idxs]
    for j, d_str_cur in enumerate(blind_zone):
        val = float(base_90[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]:
                output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        f.write(f"{d_str_cur}\t{output_rows[d_str_cur]}\n")

print(f"\n✅ 解碼完成！有效寫入非零點數: {n_written:,}")
print(f"✅ 輸出 → {OUT_TSV}")
print(f"\n📊 [v2] Recovery Gate α={GATE_ALPHA} + 規律性自動調權 評估結果：")
scores = evaluate_predictions(
    str(PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'),
    str(OUT_TSV)
)
for k, v in scores.items():
    print(f"  • {k:<25}: {v:.5f}")
