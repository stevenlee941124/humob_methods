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

# 支援指令列直接指定 Epoch (例如 python step4b_sample_origin_flow_matching.py 3)
target_ep = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
if target_ep:
    CHECKPOINT = PACKAGE_ROOT / 'data' / 'outputs' / f'origin_fm_checkpoint_ep{target_ep}.pt'
else:
    # 預設自動尋找早停候選權重 (優先 ep4 -> ep3 -> ep5 -> default)
    candidates = [
        PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint_ep4.pt',
        PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint_ep3.pt',
        PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint_ep5.pt',
        PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint.pt',
    ]
    CHECKPOINT = next((c for c in candidates if c.exists()), candidates[-1])
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

DATES_PKL   = SHARED_DATA / 'processed' / 'dates.pkl'
if DATES_PKL.exists():
    with open(DATES_PKL, 'rb') as f: obs_dates = pickle.load(f)
else:
    obs_dates = cal_dates
obs_date_to_idx = {d: i for i, d in enumerate(obs_dates)}

pre_dates  = [d for d in obs_dates if '20231101' <= d <= '20231225']
late_dates = [d for d in obs_dates if '20240120' <= d <= '20240131']

# ── 8 維條件向量 ──────────────────────────────────────────────────────────────
def build_cond_8d(d_str: str, ox_norm: float, oy_norm: float) -> np.ndarray:
    dt    = datetime.strptime(d_str, '%Y%m%d')
    wd    = dt.weekday()
    c_idx = cal_date_to_idx.get(d_str, 0)
    tau   = c_idx / 365.0
    return np.array([
        math.sin(2 * math.pi * wd / 7),          #週週期sin
        math.cos(2 * math.pi * wd / 7),          #週週期cos
        1.0 if d_str in JAPAN_HOLIDAYS else 0.0, #是否假日
        math.sin(2 * math.pi * tau),             #年週期sin
        math.cos(2 * math.pi * tau),             #年週期cos
        tau,                                     #年進度
        ox_norm,                                 #起點 x 座標正規化
        oy_norm,                                 #起點 y 座標正規化
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
print(f"✅ 模型載入完成: {CHECKPOINT.name} (Epoch: {ckpt.get('epoch')}, Best Loss: {ckpt.get('best_loss', 0.0):.6f})", flush=True)

eval_origins      = meta['eval_origins']
eval_destinations = meta['eval_destinations']
origin_meta_list  = meta['origin_meta_list']
baselines         = meta['baselines']

# 僅對 1,476 條對角線與活躍跨區主幹更新為 9-Class 物理雙錨點 Baseline (確保 Feb 1 銜接 1 月底而非 1 月最低點，且荒野死角不灌入浮點底噪)
FULL_BASE_PKL = SHARED_DATA / 'outputs' / 'full_year_baseline.pkl'
if FULL_BASE_PKL.exists():
    with open(FULL_BASE_PKL, 'rb') as f:
        full_baselines = pickle.load(f)
        n_upd = 0
        for pk in list(baselines.keys()):
            is_pk_diag = (pk.split('-')[0] == pk.split('-')[1])
            raw_arr = od_ts.get(pk)
            mean_v = float(np.nanmean(raw_arr)) if (raw_arr is not None and len(raw_arr) > 0) else 0.0
            if (is_pk_diag or mean_v >= 1.0) and pk in full_baselines:
                baselines[pk] = full_baselines[pk]
                n_upd += 1
    print(f"✅ 已將 {n_upd:,} 條活躍大動脈之 Baseline 精確校準至 1 月底錨點 (跨區死角維持純淨 0 底噪)", flush=True)

# 載入 9-Class 物理分類字典 (確保 Class 1: Persistent Zero 100% 嚴格歸 0)
ROUTE_META_PKL = SHARED_DATA / 'outputs' / 'meta_1476.pkl'
cls_map = {}
if ROUTE_META_PKL.exists():
    with open(ROUTE_META_PKL, 'rb') as f:
        r_meta = pickle.load(f)
        if isinstance(r_meta, dict) and 'active_routes' in r_meta:
            cls_map = {r['pair_key']: r.get('class_id', 6) for r in r_meta['active_routes']}
print(f"✅ 已載入 9-Class 分類字典 ({len(cls_map):,} 條路線，Class 1 Persistent Zero 共 {sum(1 for v in cls_map.values() if v == 1):,} 條)", flush=True)

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

# 9-Class 大圖展示專用路線名單 (Class 2~9 保持生動強烈波動；Class 1 100% 歸零)
PLOTTED_ROUTES = {
    '39_46-39_46', '58_43-58_43', '58_44-58_44', '41_47-41_47',
    '30_69-30_69', '38_43-38_43', '36_37-36_37', '53_37-53_37',
    '34_70-33_70', '43_45-43_44', '58_44-58_43', '41_46-41_47',
    '30_69-31_69', '34_38-34_37', '61_63-61_62', '31_47-31_48'
}

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

            # 1. 核心物理先驗：Class 1 (Persistent Zero) 全島 100% 絕對歸 0！(決無任何例外，徹底消除 702 條路線的虛假噪聲)
            cls_id = cls_map.get(pair_key, 6)
            if cls_id == 1:
                continue

            # 2. 幾乎沒有流量的路線直接跳過 (但確保 9-Class 展示路線始終保留並計算生成波動)
            if pair_key not in PLOTTED_ROUTES and mean_v < 0.15 and p_act < 0.15:
                continue

            # 取出該目的地網格的 Z 預測值 (N_BLIND,)
            z_raw   = z_o[:, 0, dx, dy]          # (N_BLIND,)
            sig_val = float(sigma_o[0, dx, dy])

            # 零中心 RMS 能量正規化 (絕不減均值！保持二月對齊！同時還原真實物理振幅)
            rms_val = float(np.sqrt(np.mean(z_raw ** 2)))
            if rms_val > 1e-4:
                z_norm = z_raw / rms_val
            else:
                z_norm = z_raw

            z_clip = np.clip(z_norm, -2.5, 2.5)

            is_diag = (o_str == d_str_k)
            # 全島所有路線（對角線 + 跨區）全面釋放 100% Flow Matching 原生波動振幅 (alpha=1.0)，一視同仁！
            alpha = 1.0

            # 計算 1 月底真實恢復起點 G_start (直接從 1 月底實際人流波動延續過去，決不歸 0)
            if raw_arr is not None:
                pre_v  = [raw_arr[obs_date_to_idx[d]] for d in pre_dates  if obs_date_to_idx.get(d, 9999) < len(raw_arr) and not np.isnan(raw_arr[obs_date_to_idx[d]])]
                late_v = [raw_arr[obs_date_to_idx[d]] for d in late_dates if obs_date_to_idx.get(d, 9999) < len(raw_arr) and not np.isnan(raw_arr[obs_date_to_idx[d]])]
                std_pre  = np.std(pre_v) if len(pre_v) >= 7 else 0.0
                std_late = np.std(late_v) if len(late_v) >= 5 else 0.0
                mean_pre = np.mean(pre_v) if pre_v else 0.0
                mean_late = np.mean(late_v) if late_v else 0.0
                if std_pre > 0.5:
                    g_start = float(np.clip(std_late / std_pre, 0.25, 1.15))
                elif mean_pre > 1.0:
                    g_start = float(np.clip(mean_late / mean_pre, 0.25, 1.15))
                else:
                    g_start = 1.0
            else:
                g_start = 1.0

            # 60 天真實延續門控：自 1 月底實際恢復水準 G_start (平均 84%~88%) 平滑延續至 4/1 滿載
            gate_curve = np.ones(N_BLIND, dtype=np.float32)
            for t_idx in range(N_BLIND):
                if t_idx < 60:
                    tau_60 = float(t_idx / 60.0)
                    gate_curve[t_idx] = g_start + (1.0 - g_start) * (tau_60 ** 1.5)
                else:
                    gate_curve[t_idx] = 1.0

            # 還原預測值 (採用自 1 月底延續的真實 Gate 門控)
            y_pred = np.maximum(0.0, base_90 + z_clip * sig_val * alpha * gate_curve)

            # 非對角線微弱路線截斷：平均人流小於 0.2 人者整條歸零 (清除跨區荒野底噪)
            if not is_diag and np.mean(y_pred) < 0.20 and pair_key not in PLOTTED_ROUTES:
                continue

            # 寫入輸出：0.2 以下全面硬截斷歸 0！(徹底杜絕稀疏死角微弱底噪帶來的超高 NRMSE 懲罰)
            for j, d_date in enumerate(blind_zone):
                val = float(y_pred[j])
                if val < 0.20:
                    continue  # 0.2 以下全島一律直接歸零，決不輸出
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
