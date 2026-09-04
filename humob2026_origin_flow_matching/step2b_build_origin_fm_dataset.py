"""
===============================================================================
HuMob 2026: Step 2b - Build Per-Origin Flow Matching Dataset  [Vectorized]
===============================================================================
每個訓練樣本 = (起點 O, 日期 d) 的標準化目的地殘差場 Z ∈ R^(1, 70, 100)
條件向量 c (8維):
    [sin(2π·wd/7), cos(2π·wd/7),          ← 週期性星期編碼（學一週起伏的關鍵）
     is_holiday,
     sin(2π·month/12), cos(2π·month/12),   ← 月份週期
     progression,
     origin_x/70, origin_y/100]            ← 起點空間座標

速度優化：
    - 消滅最內層 292 天 for-loop → numpy 整條時間軸一次賦值
    - Baseline 計算：obs_to_cal_idx 陣列預建，np 向量化 fill
    - 條件向量：一次性批次計算所有日期，不在 origin 迴圈內重複
    - 訓練切片：np indexing 取代逐天 append
===============================================================================
"""
import sys
import math
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from japan_calendar import JAPAN_HOLIDAYS
from baseline_module import compute_full_baseline

# ── 共用資料（從 destination_diffusion 的 processed 資料夾讀取，不需重跑 step1）──
SHARED_DATA = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
OD_PKL    = SHARED_DATA / 'processed' / 'od_time_series.pkl'
DATES_PKL = SHARED_DATA / 'processed' / 'dates.pkl'
DESTS_PKL = SHARED_DATA / 'processed' / 'eval_destinations.pkl'

# ── 本專案輸出 ────────────────────────────────────────────────────────────────
OUT_NPZ   = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_dataset.npz'
OUT_META  = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_meta.pkl'

GRID_W, GRID_H = 70, 100

print("=" * 75)
print("[Step 2b] Building Per-Origin Flow Matching Dataset  [Vectorized]")
print("=" * 75)
t_global = time.time()

with open(OD_PKL,    'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)
with open(DESTS_PKL, 'rb') as f: eval_destinations = pickle.load(f)

N_OBS = len(dates_str)

# ── 全域日曆 (366天) ──────────────────────────────────────────────────────────
start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

# 🚀 關鍵：預建 obs_to_cal_idx 陣列 (N_OBS,)，以後可以 numpy 整條賦值
obs_to_cal_idx = np.array(
    [cal_date_to_idx.get(d, -1) for d in dates_str], dtype=np.int32
)  # -1 表示該觀測日不在 cal 範圍內（不應發生）

# ── 訓練日索引 (排除 2024/02/01~04/30 盲區) ──────────────────────────────────
train_obs_indices = np.array(
    [i for i, d in enumerate(dates_str) if d < '20240201' or d >= '20240501'],
    dtype=np.int32
)
N_TRAIN = len(train_obs_indices)
print(f"✅ 訓練日數: {N_TRAIN} / {N_OBS}")

# ── 工具函數 ──────────────────────────────────────────────────────────────────
def parse_coord(grid_str: str):
    if grid_str == '-1_-1': return None
    try:
        x, y = map(int, grid_str.split('_'))
        if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
            return (x - 1, y - 1)
    except:
        pass
    return None

# ── 辨識所有 eval 起點 ─────────────────────────────────────────────────────────
print("正在識別所有 eval 起點...")
eval_dest_set = set(eval_destinations)
origin_set = set()
for pair_key in od_ts.keys():
    parts = pair_key.split('-')
    if len(parts) < 2: continue
    if parts[1] in eval_dest_set and parse_coord(parts[0]) is not None:
        origin_set.add(parts[0])

eval_origins = sorted(origin_set, key=lambda g: (int(g.split('_')[0]), int(g.split('_')[1])))
N_ORIGINS = len(eval_origins)
print(f"✅ 發現 eval 起點數: {N_ORIGINS}")

# ── 預解析所有目的地座標 ──────────────────────────────────────────────────────
dest_coord_map = {}  # d_str -> (dx, dy) or None
for d_str_k in eval_destinations:
    dest_coord_map[d_str_k] = parse_coord(d_str_k)
valid_dests = [(d, dest_coord_map[d]) for d in eval_destinations if dest_coord_map[d] is not None]
print(f"✅ 有效目的地數: {len(valid_dests)} / {len(eval_destinations)}")

# ── 🚀 批次計算所有 OD 路線的 Baseline (向量化 fill) ─────────────────────────
print(f"正在計算 OD Baseline（只計算實際存在於 od_ts 的路線）...")
t0 = time.time()

# 只需要計算存在於 od_ts 且 origin 在 eval_origins 的路線
eval_origin_set = set(eval_origins)
needed_keys = [k for k in od_ts.keys()
               if '-' in k and k.split('-')[0] in eval_origin_set
               and k.split('-')[1] in eval_dest_set]

# 預建 obs→cal 向量化 fill 工具
valid_obs_mask = obs_to_cal_idx >= 0   # 應該全部有效

baselines = {}
for i, pair_key in enumerate(needed_keys):
    raw_arr = od_ts[pair_key]           # (N_OBS,) 已存在
    # 🚀 向量化：一次填入 366 格
    y_366 = np.zeros(366, dtype=np.float64)
    cal_idxs = obs_to_cal_idx[valid_obs_mask]          # valid obs 的 cal 位置
    vals = raw_arr[valid_obs_mask].astype(np.float64)  # 對應的觀測值
    vals = np.where(np.isnan(vals), 0.0, vals)
    y_366[cal_idxs] = vals                             # 一次賦值，取代 for loop
    b_366, _, _ = compute_full_baseline(y_366, cal_dates, cal_date_to_idx)
    baselines[pair_key] = b_366

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        print(f"  Baseline 進度: {i+1:,}/{len(needed_keys):,} ({elapsed:.1f}s)", flush=True)

print(f"✅ Baseline 計算完成，共 {len(baselines):,} 條路線，耗時 {time.time()-t0:.1f}s")

# ── 🚀 預計算所有訓練日的條件向量矩陣（一次建好，不在 origin 迴圈內重複）────
# all_train_cond_base: (N_TRAIN, 6)，不含 origin_x/y
print("預計算訓練日條件向量...")
all_train_cond_base = np.zeros((N_TRAIN, 6), dtype=np.float32)
for i, ti in enumerate(train_obs_indices):
    d_str = dates_str[ti]
    dt    = datetime.strptime(d_str, '%Y%m%d')
    wd    = dt.weekday()
    c_idx = cal_date_to_idx.get(d_str, 0)
    tau   = c_idx / 365.0
    all_train_cond_base[i, 0] = math.sin(2 * math.pi * wd / 7)
    all_train_cond_base[i, 1] = math.cos(2 * math.pi * wd / 7)
    all_train_cond_base[i, 2] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
    all_train_cond_base[i, 3] = math.sin(2 * math.pi * tau)  # 連續平滑年季節週期 (取代階梯 month)
    all_train_cond_base[i, 4] = math.cos(2 * math.pi * tau)  # 連續平滑年季節週期 (取代階梯 month)
    all_train_cond_base[i, 5] = tau                          # 連續線性時間進程
print(f"✅ 條件矩陣 shape: {all_train_cond_base.shape}")

# ── 🚀 預建 cal_idx → b_366 的 obs 切片索引 ──────────────────────────────────
# b_366 是 366 長，obs_to_cal_idx 告訴我們每個觀測日對應哪個 cal 位置
# 所以 b_obs[t] = b_366[obs_to_cal_idx[t]]
# 用 numpy 一次取：b_obs = b_366[obs_to_cal_idx]

# ── 主迴圈：逐起點構建訓練樣本 ───────────────────────────────────────────────
# 預分配輸出陣列（避免 list.append 的記憶體碎片）
total_samples = N_ORIGINS * N_TRAIN
print(f"\n預分配輸出陣列: {N_ORIGINS} 起點 × {N_TRAIN} 天 = {total_samples:,} 樣本")
print(f"記憶體需求估算: sample_z ≈ {total_samples * 7000 * 4 / 1e9:.2f} GB")

sample_z_arr    = np.zeros((total_samples, 1, GRID_W, GRID_H), dtype=np.float32)
sample_cond_arr = np.zeros((total_samples, 8),                  dtype=np.float32)
origin_meta_list = []
write_ptr = 0   # 寫入指標

print("\n開始逐起點構建 destination map 訓練張量 [向量化]...")
t_loop = time.time()

for o_idx, o_str in enumerate(eval_origins):
    o_coord = parse_coord(o_str)
    if o_coord is None: continue
    ox, oy = o_coord
    ox_norm = ox / float(GRID_W)
    oy_norm = oy / float(GRID_H)

    # 🚀 構建 (N_OBS, 1, 70, 100) 不用任何 date for-loop
    y_o = np.zeros((N_OBS, GRID_W, GRID_H), dtype=np.float32)   # (T, W, H)
    b_o = np.zeros((N_OBS, GRID_W, GRID_H), dtype=np.float32)

    for d_str_k, d_coord in valid_dests:
        pair_key = f"{o_str}-{d_str_k}"
        raw_arr = od_ts.get(pair_key)
        b_366   = baselines.get(pair_key)
        dx, dy  = d_coord

        if raw_arr is not None:
            vals = raw_arr.copy()
            vals = np.where(np.isnan(vals), 0.0, vals)
            y_o[:, dx, dy] += vals                 # 🚀 整條時間軸一次加

        if b_366 is not None:
            b_obs = np.nan_to_num(b_366[obs_to_cal_idx], nan=0.0)  # 🚀 一次 fancy indexing + 防 NaN
            b_o[:, dx, dy] += b_obs.astype(np.float32)

    # 擴充 channel 維度: (N_OBS, 1, W, H)
    y_o = y_o[:, None, :, :]
    b_o = b_o[:, None, :, :]

    # sigma & 標準化
    resids  = y_o[train_obs_indices] - b_o[train_obs_indices]   # (N_TRAIN, 1, W, H)
    resids  = np.nan_to_num(resids, nan=0.0)
    sigma_o = np.std(resids, axis=0)                             # (1, W, H)
    sigma_o = np.nan_to_num(sigma_o, nan=0.1)
    sigma_o = np.maximum(sigma_o, 0.1)
    z_o     = (y_o - b_o) / sigma_o                             # (N_OBS, 1, W, H)
    z_o     = np.nan_to_num(z_o, nan=0.0, posinf=0.0, neginf=0.0)

    # 🚀 訓練切片：一次 numpy indexing，不用 for ti
    z_train = z_o[train_obs_indices]                             # (N_TRAIN, 1, W, H)
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)

    # 寫入預分配陣列
    sample_z_arr[write_ptr: write_ptr + N_TRAIN] = z_train

    # 條件向量：base (N_TRAIN, 6) + 後兩維 origin 座標
    cond = np.empty((N_TRAIN, 8), dtype=np.float32)
    cond[:, :6] = all_train_cond_base
    cond[:, 6]  = ox_norm
    cond[:, 7]  = oy_norm
    sample_cond_arr[write_ptr: write_ptr + N_TRAIN] = cond

    write_ptr += N_TRAIN

    # 儲存解碼元數據
    origin_meta_list.append({
        'o_str':   o_str,
        'ox': ox, 'oy': oy,
        'ox_norm': ox_norm,
        'oy_norm': oy_norm,
        'sigma':   sigma_o,   # (1, 70, 100)
    })

    if (o_idx + 1) % 50 == 0 or (o_idx + 1) == N_ORIGINS:
        elapsed = time.time() - t_loop
        eta = elapsed / (o_idx + 1) * (N_ORIGINS - o_idx - 1)
        print(f"  {o_idx+1:4d}/{N_ORIGINS} 起點 | 已寫入 {write_ptr:,} 樣本 | "
              f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

# 修正實際寫入數量（若有 None coord 被跳過）
sample_z_arr    = sample_z_arr[:write_ptr]
sample_cond_arr = sample_cond_arr[:write_ptr]

print(f"\n✅ 訓練樣本構建完成!  總耗時: {time.time()-t_global:.1f}s")
print(f"   sample_z shape   : {sample_z_arr.shape}")
print(f"   sample_cond shape: {sample_cond_arr.shape}")
print(f"   起點數           : {len(origin_meta_list)}")
print(f"   樣本數 / 起點    : {N_TRAIN} 天")

# ── 儲存 ──────────────────────────────────────────────────────────────────────
OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
print("正在壓縮存檔 origin_fm_dataset.npz...", flush=True)
np.savez_compressed(str(OUT_NPZ),
                    sample_z=sample_z_arr,
                    sample_cond=sample_cond_arr)

origin_fm_meta = {
    'grid_w': GRID_W,
    'grid_h': GRID_H,
    'cond_dim': 8,
    'eval_origins': eval_origins,
    'eval_destinations': eval_destinations,
    'origin_meta_list': origin_meta_list,
    'baselines': baselines,
    'dates_str': dates_str,
    'train_obs_indices': train_obs_indices.tolist(),
    'cal_date_to_idx': cal_date_to_idx,
}
with open(OUT_META, 'wb') as f:
    pickle.dump(origin_fm_meta, f)

print(f"✅  origin_fm_dataset.npz → {OUT_NPZ}")
print(f"✅  origin_fm_meta.pkl    → {OUT_META}")
print("=" * 75)
