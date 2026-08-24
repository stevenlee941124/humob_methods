"""
===============================================================================
HuMob 2026: Step 4 - Sample Predictions with Zero-Traffic Ratio Decoupled Models
===============================================================================
分流合成核心：
  1. Group A (常態連續流通, 0值天數 < 35%):
     - 4 段物理連續宏觀 Baseline + 1D Conditional DDPM (7D 週期波型)
     - 捕捉真實都會與連續流通的平日/週末上下起伏
  2. Group B (高0值率偶發走廊, 0值天數 35%~75%, 如 11_28, 10_27):
     - 中軸按活躍概率校準 (B * P_act) 大幅下沉貼地 + 微幅波型 + 門檻歸零 (max(0, y))
  3. Group C (極度死寂孤島, 0值天數 > 75%):
     - 預測 = 0.0 (杜絕空中插值懲罰)
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import torch

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from diffusion_model import ConditionalUNet1D, DDPM
from japan_calendar import JAPAN_HOLIDAYS

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_checkpoint.pt'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
SIGMA_PKL    = PACKAGE_ROOT / 'data' / 'outputs' / 'od_sigma_map.pkl'
PROFILES_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'od_profiles.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
OUT_TSV      = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_predictions.tsv'

BLIND_START = '20240201'
BLIND_END   = '20240430'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device.upper()}")

start_dt        = datetime(2023, 11, 1)
cal_dates       = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if BLIND_START <= d <= BLIND_END]
blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
N_BLIND    = len(blind_zone)

blind_cond = np.zeros((N_BLIND, 4), dtype=np.float32)
for j, d_str in enumerate(blind_zone):
    dt = datetime.strptime(d_str, '%Y%m%d')
    blind_cond[j, 0] = dt.weekday() / 6.0
    blind_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    blind_cond[j, 2] = (dt.month - 1) / 11.0
    blind_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

with open(BASELINE_PKL, 'rb') as f: baselines   = pickle.load(f)
with open(SIGMA_PKL,    'rb') as f: sigma_map   = pickle.load(f)
with open(PROFILES_PKL, 'rb') as f: od_profiles = pickle.load(f)
with open(OD_PKL,       'rb') as f: od_ts       = pickle.load(f)

model_14 = ConditionalUNet1D(seq_len=14, cond_dim=4, base_ch=64, time_dim=128).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model_14.load_state_dict(ckpt['model'])
model_14.eval()
ddpm = DDPM(T=1000, device=device)

L, stride = 14, 7
starts = list(range(0, N_BLIND - L + 1, stride))
if starts[-1] + L < N_BLIND: starts.append(N_BLIND - L)

batch_cond = np.zeros((len(starts), L, 4), dtype=np.float32)
for idx, s in enumerate(starts): batch_cond[idx] = blind_cond[s: s + L]
tensor_cond = torch.tensor(batch_cond, device=device)
weights = np.hanning(L) + 0.05

all_z = []
for _ in range(16):
    with torch.no_grad():
        z_p = ddpm.ddim_sample(model_14, (len(starts), L), tensor_cond, n_steps=50).cpu().numpy()
    p_sum, p_cnt = np.zeros(N_BLIND), np.zeros(N_BLIND)
    for idx, s in enumerate(starts):
        e = s + L
        p_sum[s:e] += z_p[idx] * weights
        p_cnt[s:e] += weights
    all_z.append(p_sum / np.maximum(p_cnt, 1e-8))

z_pred = np.mean(all_z, axis=0)
z_pred = (z_pred - np.mean(z_pred)) / np.std(z_pred)

output_rows = {d: {} for d in blind_zone}
counts = {'Group A (常態連續 0值率<35%, Diffusion)': 0,
          'Group B (高0值率偶發 0值率35~75%, 期望值貼地)': 0,
          'Group C (極度死寂 0值率>75%, 置0)': 0}
n_written = 0

for pair_key, b_366 in baselines.items():
    prof = od_profiles.get(pair_key, 'Group_C_Dead_Zero')
    if prof == 'Group_C_Dead_Zero':
        counts['Group C (極度死寂 0值率>75%, 置0)'] += 1
        continue

    baseline_90 = b_366[blind_idxs]
    sigma       = sigma_map.get(pair_key, 0.0)
    raw_arr     = od_ts.get(pair_key)
    n_pos       = np.sum(~np.isnan(raw_arr) & (raw_arr > 0.05)) if raw_arr is not None else 0
    p_act       = n_pos / 292.0

    # 1. Group A (常態連續流通, 0值天數 < 35%): 宏觀 Baseline + 1.0x Diffusion 7D 週期波型
    if prof == 'Group_A_Continuous_Diffusion':
        r_pred = z_pred * sigma * 1.0
        y_pred = np.maximum(0.0, baseline_90 + r_pred)
        counts['Group A (常態連續 0值率<35%, Diffusion)'] += 1

    # 2. Group B (高0值率偶發走廊, 0值天數 35%~75%, 如 11_28, 10_27): 中軸下沉貼地 + 門檻截斷
    else:
        b_adj = baseline_90 * p_act
        r_pred = z_pred * sigma * 0.2
        y_pred = np.maximum(0.0, b_adj + r_pred)
        y_pred[y_pred < 0.5] = 0.0
        counts['Group B (高0值率偶發 0值率35~75%, 期望值貼地)'] += 1

    parts = pair_key.split('-')
    orig, dest = parts[0], parts[1]

    for j, d_str in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if orig not in output_rows[d_str]:
                output_rows[d_str][orig] = {}
            output_rows[d_str][orig][dest] = round(val, 4)
            n_written += 1

with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str in blind_zone:
        f.write(f"{d_str}\t{output_rows[d_str]}\n")

print(f"\nPredictions Synthesis Summary:")
for k, v in counts.items():
    print(f"  • {k:<55}: {v:,} pairs")
print(f"Done broadcasting. Total non-zero predictions written: {n_written:,}")
print(f"\n✅  Predictions saved to: {OUT_TSV}")
