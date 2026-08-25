"""
===============================================================================
HuMob 2026 Hybrid: Step 5 - Sample 2D Spatial Diffusion & Synthesize 3 Layers
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from spatial_diffusion import SpatialUNet2D, SpatialDDPM
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_checkpoint.pt'
DATASET_NPZ  = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_dataset.npz'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_meta.pkl'
BASE_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_base_and_gates.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'

OUT_TSV      = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_predictions.tsv'
OUT_FIELD_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_field_90d.npz'

BLIND_START = '20240201'
BLIND_END   = '20240430'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print(f"[Step 5] Sampling 2D Spatial Tensor & Synthesizing 3-Layer Predictions on {DEVICE.upper()}")
print("=" * 80)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

blind_zone = [d for d in cal_dates if BLIND_START <= d <= BLIND_END]
blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
N_BLIND    = len(blind_zone)

# 1. 構建 90 天條件特徵
blind_cond = np.zeros((N_BLIND, 4), dtype=np.float32)
for j, d_str in enumerate(blind_zone):
    dt = datetime.strptime(d_str, '%Y%m%d')
    blind_cond[j, 0] = dt.weekday() / 6.0
    blind_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    blind_cond[j, 2] = (dt.month - 1) / 11.0
    blind_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

# 2. 載入模型與前置資訊
with open(META_PKL, 'rb') as f: meta_data = pickle.load(f)
with open(BASE_PKL, 'rb') as f: hybrid_models = pickle.load(f)
with open(OD_PKL, 'rb')   as f: od_ts = pickle.load(f)
with open(DT_PKL, 'rb')   as f: dates_str = pickle.load(f)

# 計算每條 OD 路線歷史真實活躍度 P_active (活躍天數 / 總訓練天數)
train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101' or d >= '20240501']
total_train_days = len(train_days_idx)

activity_stats = {}
for k, raw in od_ts.items():
    active_count = sum(not np.isnan(raw[oi]) and raw[oi] > 0.05 for oi in train_days_idx)
    true_p_act = active_count / total_train_days
    y_train_all = [raw[oi] if not np.isnan(raw[oi]) else 0.0 for oi in train_days_idx]
    true_mean = float(np.mean(y_train_all))
    activity_stats[k] = (true_p_act, true_mean)

model = SpatialUNet2D(in_ch=4, cond_dim=4, base_ch=32, time_dim=128).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
ddpm = SpatialDDPM(T=1000, device=DEVICE)

# 3. 2D DDIM 空間採樣
print("Executing 2D Spatial DDIM Sampling...")
cond_tensor = torch.tensor(blind_cond, device=DEVICE)

with torch.no_grad():
    z_spatial_pred = ddpm.ddim_sample(model, (N_BLIND, 4, 70, 100), c_cond=cond_tensor, n_steps=50).cpu().numpy()

# 動態截斷：嚴格限制在 [-2.5, +2.5]
z_spatial_pred = np.clip(z_spatial_pred, -2.5, 2.5)
np.savez_compressed(str(OUT_FIELD_NPZ), z_spatial_pred=z_spatial_pred, dates=np.array(blind_zone))

# 4. 三層物理合成解碼 (結合活躍度門控與稀疏分位過濾)
print("Synthesizing Layer 1 (Baseline) + Layer 2 (Gated ψ) + Layer 3 (Spatial Field)...")
output_rows = {d: {} for d in blind_zone}
n_written = 0

for item in meta_data['od_spatial_map']:
    pair_key = item['pair_key']
    o_str    = item['o_str']
    d_str    = item['d_str']
    o_c      = item['o_coord']
    d_c      = item['d_coord']
    is_diag  = item['is_diag']
    is_ext   = item['is_ext']

    m = hybrid_models.get(pair_key)
    if m is None:
        continue

    p_act, mean_val = activity_stats.get(pair_key, (0.0, 0.0))

    # 🌟 核心物理分離：超低流量/散粒噪聲路線 (活躍天數 < 20% 或 均值 < 0.40人)：
    # 物理上大多為極度偶發的單次偶然雜訊，直接完全不預測，保持 100% 純淨 0.0 貼地！
    if p_act < 0.20 or mean_val < 0.40:
        continue

    b_90   = m['b_366'][blind_idxs]
    g_val  = m['gate_g']
    c_func = m['carrier_func']
    sig_w  = m['sigma_weekly']
    s_reg  = m['s_reg']

    # 2. Layer 2: 確定性通勤載波
    psi_90 = np.zeros(N_BLIND, dtype=np.float64)
    if c_func is not None and g_val > 0.01:
        for j, ci in enumerate(blind_idxs):
            dow = cal_dts[ci].weekday()
            psi_90[j] = float(c_func(14.0 + dow))

    # 3. Layer 3: 空間擴散場
    if is_diag and o_c is not None:
        z_i = z_spatial_pred[:, 0, o_c[0], o_c[1]]
    elif not is_ext and o_c is not None and d_c is not None:
        z_i = 0.5 * (z_spatial_pred[:, 1, o_c[0], o_c[1]] + z_spatial_pred[:, 2, d_c[0], d_c[1]])
    elif is_ext:
        target_c = d_c if o_c is None else o_c
        z_i = z_spatial_pred[:, 3, target_c[0], target_c[1]] if target_c is not None else np.zeros(N_BLIND)
    else:
        z_i = np.zeros(N_BLIND)

    # 空間振幅比例 (嚴格隨自身活躍度衰減，防止低流量網格暴走)
    spatial_scale = min(sig_w, 2.0) * p_act if sig_w > 0.05 else (0.35 * p_act)

    # 中低規律路線 (S_reg < 0.35)：中軸按活躍度適度阻尼
    b_eff = b_90 * min(1.0, p_act * 1.5) if s_reg < 0.35 else b_90

    # 🌟 核心三層物理合成
    y_raw = b_eff + (g_val * psi_90 * sig_w) + (z_i * spatial_scale)
    y_pred = np.maximum(0.0, y_raw)

    # 🌟 低活躍/低規律度截斷：低於分位數的天數自然沉降至 0.0
    if p_act < 0.50:
        zero_quantile = (1.0 - p_act) * 100.0
        q_val = np.percentile(y_pred, zero_quantile)
        y_pred[y_pred <= q_val] = 0.0

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

print(f"✅ Prediction generated! Non-zero entries: {n_written:,}")
print(f"✅ Saved to: {OUT_TSV}")

# 5. 官方評測
print("\n" + "=" * 80)
print("📊 [Evaluation] Official April Evaluation Score on Real Ground Truth")
print("=" * 80)
scores = evaluate_predictions(str(PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'), str(OUT_TSV))
for k, v in scores.items():
    print(f"  • {k:<25}: {v:.5f}")
print("=" * 80)
