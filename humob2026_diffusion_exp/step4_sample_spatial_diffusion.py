"""
===============================================================================
HuMob 2026: Step 4 Spatial - Sample 2D Spatial Tensor & Decode to 14,563 OD Pairs
===============================================================================
詳細數學模型與反正規化還原邏輯，請參閱：DIFFUSION_MATHEMATICAL_MODEL.md

資料結構範例 (Data Structure Examples)：
  - 模型生成輸出 (spatial_field_90d.npz):
    'z_spatial_pred' -> 大小為 (90, 4, 70, 100) 的預測殘差張量
  - 最終反正規化解碼結果 (diffusion_predictions.tsv):
    20240201 \t {'31_38': {'31_38': 13.5, '32_38': 0.8}, ...}
    20240202 \t {'31_38': {'31_38': 14.1, ...}, ...}
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from spatial_diffusion_model import SpatialUNet2D, SpatialDDPM
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_ddpm_checkpoint.pt'
DATASET_NPZ  = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_diffusion_dataset.npz'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_meta.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
PROFILES_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'od_profiles.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'

OUT_TSV      = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_predictions.tsv'
OUT_FIELD_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_field_90d.npz'

BLIND_START = '20240201'
BLIND_END   = '20240430'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 75)
print(f"[Step 4 Spatial] Sampling 2D Spatial Tensor & Decoding to OD Pairs on {DEVICE.upper()}")
print("=" * 75)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if BLIND_START <= d <= BLIND_END]
blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
N_BLIND    = len(blind_zone)

# ── 1. 構建 90 天日曆條件 ─────────────────────────────────────
blind_cond = np.zeros((N_BLIND, 4), dtype=np.float32)
for j, d_str in enumerate(blind_zone):
    dt = datetime.strptime(d_str, '%Y%m%d')
    blind_cond[j, 0] = dt.weekday() / 6.0
    blind_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    blind_cond[j, 2] = (dt.month - 1) / 11.0
    blind_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

# ── 2. 載入模型與權重 ──────────────────────────────────────────
with open(META_PKL, 'rb')     as f: spatial_meta = pickle.load(f)
with open(BASELINE_PKL, 'rb') as f: baselines    = pickle.load(f)
with open(PROFILES_PKL, 'rb') as f: od_profiles  = pickle.load(f)
with open(OD_PKL, 'rb')       as f: od_ts        = pickle.load(f)
with open(DATES_PKL, 'rb')    as f: dates_str    = pickle.load(f)

# 計算每條 OD 路線的真實殘差標準差 sigma_i
train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101' or d >= '20240501']
real_sigmas = {}
for k, raw in od_ts.items():
    b = baselines.get(k)
    if b is None: 
        real_sigmas[k] = 0.5
        continue
    resids = []
    for oi in train_days_idx:
        d = dates_str[oi]
        ci = cal_date_to_idx.get(d)
        y_val = raw[oi]
        b_val = b[ci]
        if not np.isnan(y_val) and not np.isnan(b_val):
            resids.append(float(y_val - b_val))
    if len(resids) >= 5:
        real_sigmas[k] = max(0.1, float(np.std(resids)))
    else:
        real_sigmas[k] = 0.5

model = SpatialUNet2D(in_ch=4, cond_dim=4, base_ch=32, time_dim=128).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
ddpm = SpatialDDPM(T=1000, device=DEVICE)

# ── 3. 2D DDIM 空間採樣 ────────────────────────────────────────
print(f"正在執行 2D Spatial DDIM 採樣 (生成 90 天 × 4 通道 × 70 × 100 全域空間流量場)...")
cond_tensor = torch.tensor(blind_cond, device=DEVICE)

with torch.no_grad():
    z_spatial_pred = ddpm.ddim_sample(model, (N_BLIND, 4, 70, 100), c_cond=cond_tensor, n_steps=50).cpu().numpy()

# 🌟 動態截斷：將空間殘差嚴格限制在 [-2.5, +2.5] 標準常態分佈範圍內，徹底杜絕單日暴走與極端離群值！
z_spatial_pred = np.clip(z_spatial_pred, -2.5, 2.5)

np.savez_compressed(str(OUT_FIELD_NPZ), z_spatial_pred=z_spatial_pred, dates=np.array(blind_zone))

# ── 4. 空間場精確解碼至 14,563 條 OD 路線 ─────────────────────
print(f"正在將 2D 空間流量場精確解碼至全域 14,563 條 OD 路線 (嚴格圍繞 Baseline 波動)...")

output_rows = {d: {} for d in blind_zone}
n_written = 0

for item in spatial_meta['od_spatial_map']:
    pair_key = item['pair_key']
    o_str    = item['o_str']
    d_str    = item['d_str']
    o_c      = item['o_coord']
    d_c      = item['d_coord']
    is_diag  = item['is_diag']
    is_ext   = item['is_ext']
    
    prof = od_profiles.get(pair_key, 'Group_C_Dead_Zero')
    if prof == 'Group_C_Dead_Zero':
        continue
        
    b_366 = baselines.get(pair_key)
    if b_366 is None:
        continue
    baseline_90 = b_366[blind_idxs]
    sigma_i = real_sigmas.get(pair_key, 0.5)
    
    # 提取該路線對應空間位置的標準化波動波型 z_i(t)
    if is_diag and o_c is not None:
        z_i = z_spatial_pred[:, 0, o_c[0], o_c[1]]
    elif not is_ext and o_c is not None and d_c is not None:
        z_i = 0.5 * (z_spatial_pred[:, 1, o_c[0], o_c[1]] + z_spatial_pred[:, 2, d_c[0], d_c[1]])
    elif is_ext:
        target_c = d_c if o_c is None else o_c
        z_i = z_spatial_pred[:, 3, target_c[0], target_c[1]] if target_c is not None else np.zeros(N_BLIND)
    else:
        z_i = np.zeros(N_BLIND)
        
    # 🌟 正確反正規化：100% 圍繞 Baseline(t) 波動！
    # Y_pred = max(0.0, Baseline(t) + z_i(t) * sigma_i)
    y_raw = baseline_90 + z_i * sigma_i
    y_pred = np.maximum(0.0, y_raw)

    for j, d_str_cur in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]:
                output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        f.write(f"{d_str_cur}\t{output_rows[d_str_cur]}\n")

print(f"✅ 解碼完成！共寫入非零預測點: {n_written:,}")
print(f"✅ 預測結果儲存至: {OUT_TSV}")

# ── 5. 官方真實驗證集評估 ─────────────────────────────────────
print("\n" + "=" * 75)
print("📊 [Evaluation] 2D 時空網格交互擴散模型官方評測結果")
print("=" * 75)
scores = evaluate_predictions(str(PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'), str(OUT_TSV))
for k, v in scores.items():
    print(f"  • {k:<25}: {v:.5f}")
print("=" * 75)
