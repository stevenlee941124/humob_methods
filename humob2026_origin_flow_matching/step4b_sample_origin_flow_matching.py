"""
===============================================================================
HuMob 2026: Step 4b - Sample Per-Origin Destination Maps via Flow Matching
===============================================================================
推論策略：
  1. 逐起點批次採樣（每批 batch_origins 個起點）
  2. 用 Euler ODE 20 步生成 destination map Z ∈ R^(1, 70, 100)
  3. 解碼：Y_pred = max(0, Baseline + Z * sigma)
  4. 輸出格式與現有 TSV 一致
===============================================================================
"""
import sys
import math
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from origin_flow_matching import OriginDestFlowUNet, OriginFlowMatching
from japan_calendar import JAPAN_HOLIDAYS
from evaluator import evaluate_predictions

CHECKPOINT = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint.pt'
META_PKL   = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_meta.pkl'
OUT_TSV    = PACKAGE_ROOT / 'data' / 'outputs'  / 'origin_fm_predictions.tsv'

# ── 共用資料（與 destination_diffusion 共享，不需重跑 step1）──
SHARED_DATA = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
OD_PKL      = SHARED_DATA / 'processed' / 'od_time_series.pkl'
RAW_TSV     = SHARED_DATA / 'raw' / 'humob2026-dataset.tsv'

ODE_STEPS     = 20    # Euler ODE 步數（夠用，比 DDIM 50 步快）
BATCH_ORIGINS = 4     # 每批處理 4 個起點（4*90=360張地圖，顯存佔用由 20GB 降至 ~2.5GB，徹底杜絕 OOM）
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 盲區日期 ─────────────────────────────────────────────────────────────────
start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
N_BLIND    = len(blind_zone)
print(f"盲區預測日數: {N_BLIND} 天 ({blind_zone[0]} ~ {blind_zone[-1]})")

# ── 8 維條件向量 ──────────────────────────────────────────────────────────────
def build_cond_8d(d_str: str, ox_norm: float, oy_norm: float) -> np.ndarray:
    dt    = datetime.strptime(d_str, '%Y%m%d')
    wd    = dt.weekday()
    month = dt.month - 1
    c_idx = cal_date_to_idx.get(d_str, 0)
    return np.array([
        math.sin(2 * math.pi * wd / 7),
        math.cos(2 * math.pi * wd / 7),
        1.0 if d_str in JAPAN_HOLIDAYS else 0.0,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        c_idx / 365.0,
        ox_norm,
        oy_norm,
    ], dtype=np.float32)

# ── 載入模型 & 元數據 ─────────────────────────────────────────────────────────
print("載入模型與元數據...", flush=True)
with open(META_PKL, 'rb') as f: meta = pickle.load(f)
with open(OD_PKL,   'rb') as f: od_ts = pickle.load(f)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model = OriginDestFlowUNet(
    cond_dim=ckpt.get('cond_dim', 8),
    base_ch=ckpt.get('base_ch', 32),
    time_dim=ckpt.get('time_dim', 128),
).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"✅ 模型載入完成 (Best Loss: {ckpt['best_loss']:.6f})", flush=True)

eval_origins      = meta['eval_origins']
eval_destinations = meta['eval_destinations']
origin_meta_list  = meta['origin_meta_list']
baselines         = meta['baselines']
origin_meta_dict  = {m['o_str']: m for m in origin_meta_list}

GRID_W, GRID_H = meta['grid_w'], meta['grid_h']

# 評測目的地的座標解析
def parse_coord(grid_str):
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
            return (x - 1, y - 1)
    except: pass
    return None

dest_coords = {d: parse_coord(d) for d in eval_destinations}

# ── 預先計算盲區條件向量（暫存，等等逐起點替換 origin 坐標）─────────────────
# blind_base_cond[j] = [sin_wd, cos_wd, is_hol, sin_m, cos_m, prog, 0, 0]（最後兩維待填）
blind_base_cond = np.array(
    [build_cond_8d(d, 0.0, 0.0) for d in blind_zone], dtype=np.float32
)  # (N_BLIND, 8)

# ── 主要推論迴圈 ──────────────────────────────────────────────────────────────
output_rows = {d: {} for d in blind_zone}
n_written   = 0

print(f"\n開始 per-origin Flow Matching 採樣 (ODE steps={ODE_STEPS})...", flush=True)

origin_batches = [eval_origins[i: i + BATCH_ORIGINS]
                  for i in range(0, len(eval_origins), BATCH_ORIGINS)]

for b_idx, origin_batch in enumerate(origin_batches):
    # 1. 構建此批的條件張量 (N_BLIND * len(origin_batch), 8)
    #    每個 origin 對應 N_BLIND 天
    cond_list = []
    for o_str in origin_batch:
        om = origin_meta_dict.get(o_str)
        if om is None: continue
        cond_o = blind_base_cond.copy()          # (N_BLIND, 8)
        cond_o[:, 6] = om['ox_norm']            # origin_x
        cond_o[:, 7] = om['oy_norm']            # origin_y
        cond_list.append(cond_o)

    n_valid_origins = len(cond_list)
    if n_valid_origins == 0: continue

    # cond_batch: (n_valid_origins * N_BLIND, 8)
    cond_batch = torch.tensor(
        np.concatenate(cond_list, axis=0), dtype=torch.float32, device=DEVICE
    )

    # 2. Euler ODE 採樣 → (n_valid_origins * N_BLIND, 1, 70, 100)
    shape = (n_valid_origins * N_BLIND, 1, GRID_W, GRID_H)
    with torch.inference_mode():
        z_pred_tensor = OriginFlowMatching.sample(model, cond_batch, shape, DEVICE, n_steps=ODE_STEPS)
        z_pred = z_pred_tensor.cpu().numpy()  # (n_valid_origins * N_BLIND, 1, 70, 100)
        del z_pred_tensor, cond_batch
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 3. 逐起點解碼
    for oi, o_str in enumerate(origin_batch):
        om = origin_meta_dict.get(o_str)
        if om is None: continue

        # 取出這個起點的預測切片: (N_BLIND, 1, 70, 100)
        z_o = z_pred[oi * N_BLIND: (oi + 1) * N_BLIND]
        sigma_o = om['sigma']   # (1, 70, 100)

        for d_str_k in eval_destinations:
            pair_key = f"{o_str}-{d_str_k}"
            d_coord  = dest_coords.get(d_str_k)
            if d_coord is None: continue
            dx, dy = d_coord

            b_366 = baselines.get(pair_key)
            raw_arr = od_ts.get(pair_key)

            # Baseline（盲區 90 天）
            base_90 = np.array([
                float(b_366[cal_date_to_idx[d]]) if b_366 is not None else 0.0
                for d in blind_zone
            ], dtype=np.float32)

            # 歷史平均（用於流量過濾）
            if raw_arr is not None:
                valid_v = [x for x in raw_arr if not np.isnan(x)]
                mean_v  = np.mean(valid_v) if valid_v else 0.0
                p_act   = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
            else:
                mean_v, p_act = 0.0, 0.0

            # 幾乎沒有流量的路線直接跳過
            if mean_v < 0.15 and p_act < 0.15:
                continue

            # 取出該目的地網格的 Z 預測值 (N_BLIND,)
            z_raw   = z_o[:, 0, dx, dy]          # (N_BLIND,)
            sig_val = float(sigma_o[0, dx, dy])

            # 路線獨立 Z-Score 標準化（防止 U-Net 輸出偏移）
            z_std = np.std(z_raw)
            if z_std > 1e-6:
                z_norm = (z_raw - np.mean(z_raw)) / z_std
            else:
                z_norm = np.zeros_like(z_raw)
            z_norm = np.clip(z_norm, -2.5, 2.5)

            # 還原預測值
            if mean_v >= 0.30 and p_act >= 0.25:
                y_pred = np.maximum(0.0, base_90 + z_norm * sig_val)
            elif mean_v >= 0.15:
                y_pred = np.maximum(0.0, base_90)
            else:
                y_pred = np.maximum(0.0, base_90)

            # 寫入輸出
            for j, d_date in enumerate(blind_zone):
                val = float(y_pred[j])
                if val > 0.05:
                    if o_str not in output_rows[d_date]:
                        output_rows[d_date][o_str] = {}
                    output_rows[d_date][o_str][d_str_k] = round(val, 4)
                    n_written += 1

    if (b_idx + 1) % 5 == 0 or (b_idx + 1) == len(origin_batches):
        print(f"  批次 {b_idx+1}/{len(origin_batches)} | 已寫入: {n_written:,}", flush=True)

# ── 儲存結果 ──────────────────────────────────────────────────────────────────
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str_cur in blind_zone:
        f.write(f"{d_str_cur}\t{output_rows[d_str_cur]}\n")

print(f"\n✅ 預測完成！有效寫入非零點數: {n_written:,}")
print(f"✅ 輸出 TSV → {OUT_TSV}")

# ── 評估 ─────────────────────────────────────────────────────────────────────
if RAW_TSV.exists():
    scores = evaluate_predictions(str(RAW_TSV), str(OUT_TSV))
    print("\n📊 評估結果:")
    for k, v in scores.items():
        print(f"  • {k:<25}: {v:.5f}")

# ── 🌟 自動導出官方提交格式並驗證 ──────────────────────────────────────────────
OUT_SUBMISSION = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_submission_official.tsv'
VALIDATOR      = PACKAGE_ROOT / 'humob2026_validator.py'

print("\n" + "=" * 75)
print("📦 正在導出官方提交檔案 (20240201 ~ 20240331)...")
with open(OUT_TSV, 'r', encoding='utf-8') as fin, open(OUT_SUBMISSION, 'w', encoding='utf-8') as fout:
    for line in fin:
        pts = line.strip().split('\t')
        if len(pts) == 2 and '20240201' <= pts[0] <= '20240331':
            fout.write(f"{pts[0]}\t{pts[1]}\n")

if VALIDATOR.exists():
    import subprocess
    print("🔍 正在執行官方 validator 驗證...")
    res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUBMISSION)], capture_output=True, text=True)
    print("Validator 輸出:")
    print(res.stdout)
    if res.returncode == 0:
        print(f"🎉 恭喜！官方提交檔案 100% 通過驗證！檔案位置 → {OUT_SUBMISSION}")
    else:
        print("❌ 驗證失敗:", res.stderr)
print("=" * 75)
