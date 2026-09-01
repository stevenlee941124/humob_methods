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

# 預計算 Recovery Gate（形狀：(N_BLIND,)）
# τ=0 → Feb 1（剛災後），τ=1 → Apr 30（接近恢復）
recovery_gate = np.array(
    [(j / (N_BLIND - 1)) ** GATE_ALPHA for j in range(N_BLIND)],
    dtype=np.float32
)

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

    # ── 🌟 改進 1：萃取歷史 ψ₇ 並計算規律性分數 ────────────────────────────
    pre_obs = [(dates_str[oi], raw[oi])
               for oi in train_days_idx
               if oi < len(raw) and not np.isnan(raw[oi])]

    psi_comp  = np.zeros(N_BLIND)
    psi_scale = 0.4          # 預設（低規律路線）
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

        # 規律性 = 週間 ψ 的標準差 / 路線整體波動
        # 越高 → 鋸齒波越明顯 → ψ 越可信
        wd_stdev   = np.std(psi_7)
        regularity = float(np.clip(wd_stdev / max(sig_i, 1e-5), 0.0, 1.0))

        # ── 🌟 改進 2：規律性自動調整 ψ vs Diffusion 比重 ──────────────────
        # regularity=1.0 → psi_scale=0.85（幾乎全靠 ψ）
        # regularity=0.0 → psi_scale=0.30（幾乎不用 ψ）
        psi_scale = 0.30 + 0.55 * regularity

        # ── 🌟 改進 3：加入 Recovery Gate（τ²，慢起快收）──────────────────
        for j in range(N_BLIND):
            gate = recovery_gate[j]           # τ²：0→0.01→0.25→1.0
            psi_comp[j] = psi_7[cal_dts[blind_idxs[j]].weekday()] * gate * psi_scale

    # Diffusion 比重：規律性越高 → diffusion 佔比越小（防暴走）
    diff_weight = 1.0 - regularity * 0.6     # regularity=1 → 0.4，regularity=0 → 1.0
    diff_scale  = sig_i * 0.4 * diff_weight

    # 路線獨立 Z-Score 標準化（防 U-Net 整體偏移）
    z_i_raw = z_pred_all[:, c_idx, ox, oy]
    z_std   = np.std(z_i_raw)
    if z_std > 1e-6:
        z_i = (z_i_raw - np.mean(z_i_raw)) / z_std
    else:
        z_i = np.zeros_like(z_i_raw)
    z_i = np.clip(z_i, -2.5, 2.5)

    # 最終還原
    if mean_v >= 0.30 and p_act >= 0.25:
        y_pred = np.maximum(0.0, base_90 + psi_comp + z_i * diff_scale)
    elif mean_v >= 0.15 and p_act >= 0.15:
        y_pred = np.maximum(0.0, base_90)
    else:
        y_pred = np.zeros(N_BLIND, dtype=np.float32)

    for j, d_str_cur in enumerate(blind_zone):
        val = float(y_pred[j])
        if val > 0.05:
            if o_str not in output_rows[d_str_cur]:
                output_rows[d_str_cur][o_str] = {}
            output_rows[d_str_cur][o_str][d_str] = round(val, 4)
            n_written += 1

# Non-active routes：純 Baseline
for pair_key, b_366 in baselines.items():
    if pair_key in active_keys: continue
    if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)): continue
    raw     = od_ts.get(pair_key)
    valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
    mean_v  = np.mean(valid_v) if valid_v else 0.0
    p_act   = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
    if mean_v < 0.25 or p_act < 0.20: continue
    parts = pair_key.split('-')
    o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
    d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
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
