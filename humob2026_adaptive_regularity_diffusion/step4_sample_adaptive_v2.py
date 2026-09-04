"""
===============================================================================
HuMob 2026: Step 4 - Adaptive Regularity Diffusion Sampling (V2 Dual-Anchor)
===============================================================================
Features:
1. 34-Week Continuous Regularity Metric: 1.0 - mean_var
2. Dual-Track Continuous Weights:
   - Diagonal: W_psi = 0.35 + 0.85 * Reg, W_diff = max(0.05, 1.0 - 0.90 * Reg)
   - Off-diagonal: W_psi = 0.50 + 0.60 * Reg, W_diff = max(0.0, 1.0 - Reg / 0.25)
3. Per-Route Dual-Anchor Adaptive Gate (Spring Expansion):
   - g_start = clip(std_Jan / std_pre, 0.25, 1.15) (Disaster Shock Left Anchor)
   - g_end   = clip(std_May / std_pre, 0.35, 1.80) (Spring Expansion Right Anchor)
   - Gate(t) = g_start + (g_end - g_start) * (tau ^ 1.5)
4. Route-wise Z-Score Normalization & Micro-Noise Filtering
===============================================================================
"""
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_DATA_DIR = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
DATA_DIR = PACKAGE_ROOT / 'data'

sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM
from japan_calendar import JAPAN_HOLIDAYS

def get_data_path(rel_path):
    local_p = DATA_DIR / rel_path
    if local_p.exists():
        return local_p
    shared_p = SHARED_DATA_DIR / rel_path
    if shared_p.exists():
        return shared_p
    raise FileNotFoundError(f"Cannot find {rel_path} in {DATA_DIR} or {SHARED_DATA_DIR}")

CHECKPOINT   = get_data_path('outputs/ddpm_1476_checkpoint.pt')
META_PKL     = get_data_path('outputs/meta_1476.pkl')
BASELINE_PKL = get_data_path('outputs/full_year_baseline.pkl')
OD_PKL       = get_data_path('processed/od_time_series.pkl')
DATES_PKL    = get_data_path('processed/dates.pkl')
RAW_TSV      = get_data_path('raw/humob2026-dataset.tsv')

OUT_DIR = DATA_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TSV = OUT_DIR / 'dest1476_predictions_adaptive_v2.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用設備: {DEVICE}")

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
N_BLIND = len(blind_zone)

blind_cond = np.zeros((N_BLIND, 4), dtype=np.float32)
for j, d_str in enumerate(blind_zone):
    dt = datetime.strptime(d_str, '%Y%m%d')
    blind_cond[j, 0] = dt.weekday() / 6.0
    blind_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    blind_cond[j, 2] = (dt.month - 1) / 11.0
    blind_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

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

# Anchors definition
late_jan_dates   = [d for d in dates_str if '20240115' <= d <= '20240131' and d not in EXCLUDED_DATES]
normal_pre_dates = [d for d in dates_str if '20231101' <= d < '20231225' and d not in EXCLUDED_DATES]
early_may_dates  = [d for d in dates_str if '20240501' <= d <= '20240520' and d not in EXCLUDED_DATES]
date_idx_map     = {d: i for i, d in enumerate(dates_str)}

CACHE_Z_FILE = SHARED_DATA_DIR / 'outputs' / 'z_pred_cache.npy'
if not CACHE_Z_FILE.exists():
    CACHE_Z_FILE = OUT_DIR / 'z_pred_cache.npy'

if CACHE_Z_FILE.exists():
    print(f"📦 載入快取的 Diffusion 去噪預測殘差: {CACHE_Z_FILE}")
    z_pred_all = np.load(CACHE_Z_FILE)
else:
    print(f"⚡ 執行 DDIM 50 步反向去噪採樣...")
    model = MultiChannelSpatialUNet(in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

    cond_tensor = torch.tensor(blind_cond, device=DEVICE)
    z_pred_list = []
    batch_days = 15
    with torch.no_grad():
        for start_b in range(0, N_BLIND, batch_days):
            end_b = min(N_BLIND, start_b + batch_days)
            sub_shape = (end_b - start_b, 1476, 70, 100)
            sub_cond = cond_tensor[start_b:end_b]
            sub_z = ddpm.ddim_sample(model, sub_shape, sub_cond, n_steps=50, eta=0.0).cpu()
            z_pred_list.append(sub_z)
    z_pred_all = torch.cat(z_pred_list, dim=0).numpy()
    np.save(OUT_DIR / 'z_pred_cache.npy', z_pred_all)

print(f"✅ Diffusion 殘差形狀: {z_pred_all.shape}")

train_obs_dates = meta_1476['train_obs_dates']
train_days_idx = [date_idx_map[d] for d in train_obs_dates if d in date_idx_map]

output_rows = {d: {} for d in blind_zone}
n_written = 0

print("🌟 正在執行【全自適應 Regularity + 雙錨點春季振幅門控】解碼合成...")

tau_90 = np.linspace(0.0, 1.0, N_BLIND, dtype=np.float32)

for r in meta_1476['active_routes']:
    pair_key = r['pair_key']
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]

    c_idx, ox, oy = r['c_idx'], r['ox'], r['oy']
    sig_i = r['sigma_i']
    mean_v = r.get('mean_v', 1.0)
    cls_id = r.get('class_id', 6)

    b_366 = baselines.get(pair_key)
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
        continue
    base_90 = np.copy(b_366[blind_idxs])

    raw = od_ts.get(pair_key)
    if raw is None:
        continue

    # 1. 萃取歷史 ψ7 週波
    pre_obs = [(dates_str[oi], raw[oi])
               for oi in train_days_idx
               if oi < len(raw) and not np.isnan(raw[oi])]

    if len(pre_obs) >= 14 and mean_v >= 1.0:
        overall_m = np.mean([v for _, v in pre_obs])
        wd_map = {w: [] for w in range(7)}
        for d_str_k, v in pre_obs:
            wd_map[datetime.strptime(d_str_k, '%Y%m%d').weekday()].append(v)

        psi_7 = np.array([
            (np.mean(wd_map[w]) - overall_m) if wd_map[w] else 0.0
            for w in range(7)
        ], dtype=np.float32)
    else:
        psi_7 = np.zeros(7, dtype=np.float32)

    # 2. 34 週常態期波形重合離散度 (無偏理論規律度)
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
        regularity = float(np.clip(1.0 - mean_var, 0.0, 1.0))
    else:
        regularity = 0.0

    # 3. 路線獨立 Z-Score 標準化
    z_i_raw = z_pred_all[:, c_idx, ox, oy]
    z_std   = np.std(z_i_raw)
    if z_std > 1e-6:
        z_i = (z_i_raw - np.mean(z_i_raw)) / z_std
    else:
        z_i = np.zeros_like(z_i_raw)
    z_i = np.clip(z_i, -2.5, 2.5)

    is_diag = (o_str == d_str)

    # 4. 🌟 每路線專屬【雙錨點自適應振幅過渡門控】
    pre_vals  = [raw[date_idx_map[d]] for d in normal_pre_dates if date_idx_map[d] < len(raw) and not np.isnan(raw[date_idx_map[d]])]
    late_vals = [raw[date_idx_map[d]] for d in late_jan_dates if date_idx_map[d] < len(raw) and not np.isnan(raw[date_idx_map[d]])]
    may_vals  = [raw[date_idx_map[d]] for d in early_may_dates if date_idx_map[d] < len(raw) and not np.isnan(raw[date_idx_map[d]])]

    std_pre  = np.std(pre_vals) if len(pre_vals) >= 7 else 0.0
    std_late = np.std(late_vals) if len(late_vals) >= 5 else 0.0
    std_may  = np.std(may_vals) if len(may_vals) >= 5 else 0.0

    if std_pre > 0.5:
        # 左錨點 (受災壓抑) 與 右錨點 (春季擴張自適應)
        g_start = float(np.clip(std_late / std_pre, 0.25, 1.15))
        g_end   = float(np.clip(std_may / std_pre, 0.35, 1.80))
    else:
        g_start = 1.0
        g_end   = 1.0

    cur_gate = g_start + (g_end - g_start) * (tau_90 ** 1.5)

    # 5. 雙軌四象限自適應融合解碼
    if not is_diag:
        # 【非對角線 (跨區流動)】
        if regularity <= 0.05 and mean_v <= 1.20:
            y_pred = np.zeros(N_BLIND, dtype=np.float32)
        else:
            diff_w = max(0.0, 1.0 - regularity / 0.25) if regularity < 0.25 else 0.0
            psi_sc = 0.50 + 0.60 * regularity
            psi_raw = np.array([psi_7[cal_dts[blind_idxs[j]].weekday()] * psi_sc for j in range(N_BLIND)], dtype=np.float32)
            resid_total = (psi_raw + z_i * (sig_i * 0.35 * diff_w)) * cur_gate
            y_pred = np.maximum(0.0, base_90 + resid_total)
            y_pred[y_pred < 0.05] = 0.0
    else:
        # 【對角線 (核心停留)】
        if regularity <= 0.05 and mean_v <= 1.20:
            y_pred = np.zeros(N_BLIND, dtype=np.float32)
        else:
            psi_sc = 0.35 + 0.85 * regularity
            diff_w = max(0.05, 1.0 - regularity * 0.90)
            psi_raw = np.array([psi_7[cal_dts[blind_idxs[j]].weekday()] * psi_sc for j in range(N_BLIND)], dtype=np.float32)
            resid_total = (psi_raw + z_i * (sig_i * 0.40 * diff_w)) * cur_gate
            y_pred = np.maximum(0.0, base_90 + resid_total)

    # 寫入預測
    for j, d_str_cur in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]:
                output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

print(f"💾 正在寫入預測結果 TSV: {OUT_TSV} ...")
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        od_dict = output_rows[d_str_cur]
        f.write(f"{d_str_cur}\t{od_dict}\n")

print(f"✅ 成功完成採樣！共寫入 {n_written:,} 個有效 OD 預測格點。")
