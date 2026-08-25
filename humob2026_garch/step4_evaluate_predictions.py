"""
===============================================================================
HuMob 2026 流水線 - 步驟 4：最新官方規範嚴格評估指標計算 (GARCH vs. Baseline)
===============================================================================
詳細數學模型與評估指標推導，請參閱：GARCH_MATHEMATICAL_MODEL.md

資料結構範例 (Data Structure Examples)：
  - 輸入的預測 TSV 檔案格式 (例如 wave_garch_fullyear_holiday_garch.tsv):
    20240401 \t {'31_38': {'31_38': 13.5, '32_38': 0.8}, ...}
  - 讀取的真實數據集 (humob2026-dataset.tsv):
    20240401 \t {'31_38': {'31_38': 12.0, '32_38': 1.0}, ...}

輸入檔案：
  - data/raw/humob2026-dataset.tsv (包含 4 月份真實 Ground Truth)
  - data/outputs/wave_garch_fullyear_holiday_garch.tsv (GARCH 模型預測檔)
  - data/outputs/gap90_midpoint_centerline_baseline.tsv (宏觀平滑 Baseline 預測檔)
===============================================================================
"""

import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

# 設定標準輸出為 UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8')

# 定義專案路徑
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_RAW = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
GARCH_TSV = PACKAGE_ROOT / 'data' / 'outputs' / 'wave_garch_fullyear_holiday_garch.tsv'
BASE_TSV = PACKAGE_ROOT / 'data' / 'outputs' / 'gap90_midpoint_centerline_baseline.tsv'

# -----------------------------------------------------------------------------
# 官方最新公佈之標準化分母常數 (Official Baseline Constants)
# -----------------------------------------------------------------------------
MEAN_ACTUAL_DIAG_NEW = 26.57       # 對角線人流真實均值
MEAN_ACTUAL_OFFDIAG_NEW = 0.0176   # 非對角線流動真實均值

# 空間範圍定義 (X: 30~70, Y: 35~70 -> 共 41 * 36 = 1,476 個網格)
MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70

# 官方排除異常日期 (16 天)
EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

# 構建全域 1,476 個網格代碼清單
all_bbox_grids = [f"{x}_{y}" for x in range(MIN_X, MAX_X + 1) for y in range(MIN_Y, MAX_Y + 1)]
n_grids = len(all_bbox_grids)              # 1,476
n_diag_total = n_grids                      # 1,476 個對角線項目
n_offdiag_total = n_grids * (n_grids - 1)  # 2,177,100 個非對角線流動項目

def parse_tsv(filepath):
    """安全解析 TSV 檔案中的逐日 OD 字典"""
    data = {}
    filepath = Path(filepath)
    if not filepath.exists():
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            try:
                raw = parts[1].replace(': NA', ': None').replace(':NA', ':None')
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od is not None:
                    data[parts[0]] = od
            except Exception:
                pass
    return data

def evaluate_predictions(pred_dict, true_dict, eval_dates_list):
    """計算給定預測字典的 Diag RMSE、Off-Diag RMSE 與 Combined NRMSE"""
    daily_rmse_diag = []
    daily_rmse_offdiag = []

    for date_str in eval_dates_list:
        t_od = true_dict.get(date_str, {})
        p_od = pred_dict.get(date_str, {})
        
        # 1. 對角線評估 (Diagonal Stay Flow)
        sq_diag_sum = 0.0
        for g in all_bbox_grids:
            t_val = t_od.get(g, {}).get(g, 0.0) if isinstance(t_od.get(g, {}), dict) else 0.0
            p_val = p_od.get(g, {}).get(g, 0.0) if isinstance(p_od.get(g, {}), dict) else 0.0
            if t_val is None: t_val = 0.0
            if p_val is None: p_val = 0.0
            sq_diag_sum += (p_val - t_val) ** 2
            
        rmse_diag_d = math.sqrt(sq_diag_sum / n_diag_total)
        daily_rmse_diag.append(rmse_diag_d)
        
        # 2. 非對角線評估 (Off-Diagonal Flow)
        non_zero_off_sq_sum = 0.0
        srcs = set(t_od.keys()) | set(p_od.keys())
        for src in srcs:
            if src not in all_bbox_grids:
                continue
            t_dsts = t_od.get(src, {}) if isinstance(t_od.get(src, {}), dict) else {}
            p_dsts = p_od.get(src, {}) if isinstance(p_od.get(src, {}), dict) else {}
            all_dsts = (set(t_dsts.keys()) | set(p_dsts.keys())) - {src}
            
            for dst in all_dsts:
                if dst not in all_bbox_grids:
                    continue
                t_val = t_dsts.get(dst, 0.0)
                p_val = p_dsts.get(dst, 0.0)
                if t_val is None: t_val = 0.0
                if p_val is None: p_val = 0.0
                non_zero_off_sq_sum += (p_val - t_val) ** 2
                
        rmse_off_d = math.sqrt(non_zero_off_sq_sum / n_offdiag_total)
        daily_rmse_offdiag.append(rmse_off_d)

    mean_rmse_diag = float(np.mean(daily_rmse_diag))
    mean_rmse_off = float(np.mean(daily_rmse_offdiag))

    nrmse_diag = mean_rmse_diag / MEAN_ACTUAL_DIAG_NEW
    nrmse_off = mean_rmse_off / MEAN_ACTUAL_OFFDIAG_NEW
    combined_nrmse = 0.5 * (nrmse_diag + nrmse_off)

    return {
        'rmse_diag': mean_rmse_diag,
        'nrmse_diag': nrmse_diag,
        'rmse_off': mean_rmse_off,
        'nrmse_off': nrmse_off,
        'combined_nrmse': combined_nrmse
    }

print("=" * 100)
print("📊 [Step 4] Official Metric Evaluation (mean_actual_diag=26.57, mean_actual_offdiag=0.0176)...")
print("=" * 100)

true_data = parse_tsv(DATA_RAW)
garch_data = parse_tsv(GARCH_TSV)
base_data = parse_tsv(BASE_TSV)

# 篩選 4 月份有效評估天數
eval_dates = sorted([d for d in true_data.keys() if d.startswith('202404') and d not in EXCLUDED_DATES])
print(f"• 4 月份有效評估天數：共 {len(eval_dates)} 天\n")

# 1. 評估 GARCH 模型
res_garch = evaluate_predictions(garch_data, true_data, eval_dates)

# 2. 評估 Baseline 模型
res_base = evaluate_predictions(base_data, true_data, eval_dates) if base_data else None

# =============================================================================
# 輸出分類評估成績單 (按用戶指定格式清楚印出)
# =============================================================================

# ── 1. GARCH 模型成績單 ──
print("=" * 100)
print("👑 【GARCH 模型 評估成績單】")
print("=" * 100)
print(f"• Combined NRMSE        : {res_garch['combined_nrmse']:.5f}  (約 {res_garch['combined_nrmse']*100:.2f}%)")
print(f"• Diag RMSE             : {res_garch['rmse_diag']:.2f} 人  (NRMSE_diag = {res_garch['nrmse_diag']:.5f})")
print(f"• Off-Diag RMSE         : {res_garch['rmse_off']:.4f} 人  (NRMSE_off = {res_garch['nrmse_off']:.5f})")
print("=" * 100)

# ── 2. Baseline 模型成績單 ──
if res_base:
    print("\n" + "=" * 100)
    print("📏 【Baseline 基準評估成績單】")
    print("=" * 100)
    print(f"• Combined NRMSE        : {res_base['combined_nrmse']:.5f}  (約 {res_base['combined_nrmse']*100:.2f}%)")
    print(f"• Diag RMSE             : {res_base['rmse_diag']:.2f} 人  (NRMSE_diag = {res_base['nrmse_diag']:.5f})")
    print(f"• Off-Diag RMSE         : {res_base['rmse_off']:.4f} 人  (NRMSE_off = {res_base['nrmse_off']:.5f})")
    print("=" * 100)

    # ── 3. 模型改善效益對比 ──
    diff_comb = res_garch['combined_nrmse'] - res_base['combined_nrmse']
    diff_diag = res_garch['rmse_diag'] - res_base['rmse_diag']
    diff_off = res_garch['rmse_off'] - res_base['rmse_off']
    pct_comb = (diff_comb / res_base['combined_nrmse']) * 100
    pct_diag = (diff_diag / res_base['rmse_diag']) * 100
    pct_off = (diff_off / res_base['rmse_off']) * 100

    print("\n" + "=" * 100)
    print("🔥 【模型改善效益對比 (GARCH vs. Baseline)】")
    print("=" * 100)
    print(f"• Combined NRMSE 降幅   : {diff_comb:.5f}  (相對改善 {abs(pct_comb):.2f}% 🏆)")
    print(f"• Diag 停留誤差減少     : {diff_diag:.2f} 人 / 網格 (下降 {abs(pct_diag):.2f}%)")
    print(f"• Off-Diag 流動誤差減少 : {diff_off:.4f} 人 / 路線 (下降 {abs(pct_off):.2f}%)")
    print("=" * 100)
