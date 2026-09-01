"""
===============================================================================
HuMob 2026: Step 4 - Ultimate Smooth Momentum + Volume Classification
===============================================================================
"""
import sys, pickle, torch, numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_1476_checkpoint.pt'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476_reg.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline_reg.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_TSV      = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions_progressive.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

with open(META_PKL, 'rb')     as f: meta_1476 = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)

train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101']

# 🌟 1. Precompute Z_TARGET and CORR_MAP for Progressive Unmasking
Z_TARGET = torch.zeros((N_BLIND, 1476, 70, 100), device=DEVICE)
CORR_MAP = torch.full((1, 1476, 70, 100), -2.0, device=DEVICE)
active_keys = set()

print("Precomputing Z_TARGET for confident routes...", flush=True)
for r in meta_1476['active_routes']:
    pair_key = r['pair_key']
    active_keys.add(pair_key)
    c_idx, ox, oy = r['c_idx'], r['ox'], r['oy']
    sig_i = r['sigma_i']
    correlation = r.get('correlation', 0.0)
    CORR_MAP[0, c_idx, ox, oy] = correlation
    
    b_366 = baselines.get(pair_key)
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
        continue
    base_90 = np.copy(b_366[blind_idxs])
    
    raw = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v = np.mean(valid_v) if valid_v else 0.0
    
    pre_obs = [(dates_str[oi], raw[oi]) for oi in train_days_idx if oi < len(raw) and not np.isnan(raw[oi])]
    psi_comp = np.zeros(N_BLIND)
    if len(pre_obs) >= 14 and mean_v >= 1.0:
        overall_m = np.mean([v for _, v in pre_obs])
        wd_map = {w: [] for w in range(7)}
        for d_str_k, v in pre_obs:
            wd_map[datetime.strptime(d_str_k, '%Y%m%d').weekday()].append(v)
        psi_7 = np.zeros(7)
        for w in range(7):
            psi_7[w] = (np.mean(wd_map[w]) - overall_m) if wd_map[w] else 0.0
        
        for j, ci in enumerate(blind_idxs):
            gate = min(max(float(base_90[j]) / max(overall_m, 1e-5), 0.0), 1.5)
            psi_comp[j] = psi_7[cal_dts[ci].weekday()] * gate
            
    # Z_TARGET is the normalized residual
    if sig_i > 1e-5:
        z_target_i = psi_comp / sig_i
    else:
        z_target_i = np.zeros(N_BLIND)
    Z_TARGET[:, c_idx, ox, oy] = torch.tensor(z_target_i, device=DEVICE, dtype=torch.float32)

# 🌟 2. Custom Progressive DDIM Sampler
def progressive_ddim_sample(ddpm, model, shape, c_cond, z_targets_batch, corr_map, n_steps=50):
    step_size = ddpm.T // n_steps
    time_steps = list(range(0, ddpm.T, step_size))
    x = torch.randn(shape, device=ddpm.device)

    for i in reversed(range(len(time_steps))):
        t_cur = time_steps[i]
        t_tensor = torch.full((shape[0],), t_cur, device=ddpm.device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t_tensor, c_cond)

        ab_cur = ddpm.alphas_bar[t_cur]
        ab_prev = ddpm.alphas_bar[time_steps[i - 1]] if i > 0 else torch.tensor(1.0, device=ddpm.device)

        x_0_pred = (x - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur)
        
        # --- PROGRESSIVE UNMASKING (INPAINTING) ---
        # 閾值從 1.0 (全不解開) 緩降至 0.2 (完全解開)
        progress = (n_steps - 1 - i) / max(1, (n_steps - 1))
        r_thresh = 1.0 - 0.8 * progress
        
        mask = (corr_map > r_thresh).float()
        
        # 用理論目標 (Z_TARGET) 強制覆寫高信心預測
        x_0_pred = x_0_pred * (1 - mask) + z_targets_batch * mask
        
        eps_recomputed = (x - torch.sqrt(ab_cur) * x_0_pred) / torch.sqrt(1.0 - ab_cur)
        dir_xt = torch.sqrt(1.0 - ab_prev) * eps_recomputed
        x = torch.sqrt(ab_prev) * x_0_pred + dir_xt

    return x

model = MultiChannelSpatialUNet(in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

print("🚀 執行 Progressive DDIM Inpainting 採樣...", flush=True)
cond_tensor = torch.tensor(blind_cond, device=DEVICE)

z_pred_list = []
batch_days = 15
with torch.no_grad():
    for start_b in range(0, N_BLIND, batch_days):
        end_b = min(N_BLIND, start_b + batch_days)
        sub_shape = (end_b - start_b, 1476, 70, 100)
        sub_cond = cond_tensor[start_b:end_b]
        sub_targets = Z_TARGET[start_b:end_b]
        
        sub_z = progressive_ddim_sample(ddpm, model, sub_shape, sub_cond, sub_targets, CORR_MAP, n_steps=50).cpu()
        z_pred_list.append(sub_z)

z_pred_all = torch.cat(z_pred_list, dim=0).numpy()

output_rows = {d: {} for d in blind_zone}
n_written = 0

for r in meta_1476['active_routes']:
    pair_key = r['pair_key']
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
    
    c_idx, ox, oy = r['c_idx'], r['ox'], r['oy']
    sig_i = r['sigma_i']
    b_366 = baselines.get(pair_key)
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
        continue
    base_90 = np.copy(b_366[blind_idxs])
    
    raw = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v = np.mean(valid_v) if valid_v else 0.0
    p_act = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
    
    cls_id = r.get('class_id', 6)
    if cls_id == 1 or (mean_v < 0.10 and p_act < 0.10):
        continue

    correlation = r.get('correlation', 0.0)
    
    if correlation < 0.0:
        psi_scale = 0.26
        diff_scale = sig_i * 0.00
    elif correlation < 0.2:
        psi_scale = 0.69
        diff_scale = sig_i * 0.03
    elif correlation < 0.4:
        psi_scale = 1.02
        diff_scale = sig_i * 0.07
    elif correlation < 0.6:
        psi_scale = 0.75
        diff_scale = sig_i * 0.04
    elif correlation < 0.8:
        psi_scale = 0.80
        diff_scale = sig_i * 0.02
    else:
        psi_scale = 1.03
        diff_scale = sig_i * 0.09

    # 🌟 3. 精確萃取並疊加「真實歷史 7 天通勤鋸齒波」
    pre_obs = [(dates_str[oi], raw[oi]) for oi in train_days_idx if oi < len(raw) and not np.isnan(raw[oi])]
    psi_comp = np.zeros(N_BLIND)
    if len(pre_obs) >= 14 and mean_v >= 1.0:
        overall_m = np.mean([v for _, v in pre_obs])
        wd_map = {w: [] for w in range(7)}
        for d_str_k, v in pre_obs:
            wd_map[datetime.strptime(d_str_k, '%Y%m%d').weekday()].append(v)
        psi_7 = np.zeros(7)
        for w in range(7):
            psi_7[w] = (np.mean(wd_map[w]) - overall_m) if wd_map[w] else 0.0
        
        for j, ci in enumerate(blind_idxs):
            gate = min(max(float(base_90[j]) / max(overall_m, 1e-5), 0.0), 1.5)
            psi_comp[j] = psi_7[cal_dts[ci].weekday()] * gate

    z_i_raw = z_pred_all[:, c_idx, ox, oy]
    # 🌟 路線獨立 Z-Score 標準化：保證分佈呈完美的 N(0,1)，徹底解決 U-Net 爆掉或扁平的問題！
    z_std = np.std(z_i_raw)
    if z_std > 1e-6:
        z_i = (z_i_raw - np.mean(z_i_raw)) / z_std
    else:
        z_i = np.zeros_like(z_i_raw)
    # 輕微 Clip 防極端防單日暴走
    z_i = np.clip(z_i, -2.5, 2.5)
    
    # 🌟 4. 完美還原 (套用最佳權重)
    if mean_v >= 1.0:
        y_pred = np.maximum(0.0, base_90 + psi_comp * psi_scale + z_i * diff_scale)
    elif mean_v >= 0.30 and p_act >= 0.25:
        y_pred = np.maximum(0.0, base_90 + psi_comp * psi_scale + z_i * diff_scale)
    elif mean_v >= 0.15 and p_act >= 0.15:
        y_pred = np.maximum(0.0, base_90)
    else:
        y_pred = np.zeros(N_BLIND, dtype=np.float32)
        
    for j, d_str_cur in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]: output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

for pair_key, b_366 in baselines.items():
    if pair_key in active_keys: continue
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)): continue
    raw = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v = np.mean(valid_v) if valid_v else 0.0
    p_act = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
    if mean_v < 0.25 or p_act < 0.20: continue
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
    base_90 = b_366[blind_idxs]
    for j, d_str_cur in enumerate(blind_zone):
        val = float(base_90[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]: output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        f.write(f"{d_str_cur}\t{output_rows[d_str_cur]}\n")

print(f"✅ 解碼完成！有效寫入非零點數: {n_written:,}")
scores = evaluate_predictions(str(PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'), str(OUT_TSV))
for k, v in scores.items():
    print(f"  • {k:<25}: {v:.5f}")
