"""
===============================================================================
HuMob 2026 流水線 - 步驟 3：全年度去污染 GARCH 分層波形合成預測模型
===============================================================================
詳細數學模型與推導，請參閱：GARCH_MATHEMATICAL_MODEL.md

資料結構範例 (Data Structure Examples)：
  - 讀取的 OD 時序矩陣 (od_ts):
    {
      '31_38-31_38': array([12.0, 15.0, np.nan, ...]),
      '31_38-32_38': array([ 1.0,  0.0, np.nan, ...]),
      ...
    }
  - 輸出的 GARCH 預測 TSV 格式 (out_gap90_garch):
    20240201 \t {'31_38': {'31_38': 13.5678, '32_38': 0.8}, ...}
    20240202 \t {'31_38': {'31_38': 14.1234, ...}, ...}

輸入檔案：
  - data/processed/od_time_series.pkl
  - data/processed/dates.pkl
  - data/processed/grid_final_classification.csv
  - data/outputs/gap90_midpoint_centerline_baseline.tsv (非對角線平滑參考)
輸出檔案：
  - data/outputs/wave_garch_fullyear_holiday_garch.tsv (最終 90 天預測 TSV)
===============================================================================
"""

import sys
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import CubicHermiteSpline, interp1d

# 設定標準輸出為 UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8')

# 動態注入 src 路徑以載入日本國定假日模組
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from japan_calendar import JAPAN_HOLIDAYS, get_holiday_features
from data_loader import parse_tsv

# 定義檔案路徑
OD_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
CLASS_CSV = PACKAGE_ROOT / 'data' / 'processed' / 'grid_final_classification.csv'
BASE_TSV = PACKAGE_ROOT / 'data' / 'outputs' / 'gap90_midpoint_centerline_baseline.tsv'
OUT_TSV = PACKAGE_ROOT / 'data' / 'outputs' / 'wave_garch_fullyear_holiday_garch.tsv'

# 能登半島官方空間邊界 (Bounding Box: X∈[30,70], Y∈[35,70])
MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70

# 官方競賽規定之異常或無效觀測排除日期 (16 天)
EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

print("=" * 100)
print("👑 [Step 3] Running GARCH: Full-Year Decontaminated GARCH Framework...")
print("=" * 100)

# 載入前置處理數據
with open(OD_PKL, 'rb') as f:
    od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f:
    dates_str = pickle.load(f)

df_class = pd.read_csv(CLASS_CSV)
class_map = dict(zip(df_class['grid_id'], df_class['final_class']))
base_data = parse_tsv(BASE_TSV) if BASE_TSV.exists() else {}

# -----------------------------------------------------------------------------
# 建立全年 366 天連續日曆索引 (2023/11/01 ~ 2024/10/31)
# -----------------------------------------------------------------------------
start_dt = datetime(2023, 11, 1)
end_dt = datetime(2024, 10, 31)
total_days = (end_dt - start_dt).days + 1
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(total_days)]
cal_dts = [start_dt + timedelta(days=i) for i in range(total_days)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

# 預測目標：90 天盲區序列 (2024/02/01 ~ 2024/04/30)
gap90_dates = [(datetime(2024, 2, 1) + timedelta(days=i)).strftime('%Y%m%d') for i in range(90)]
gap90_dows = [datetime.strptime(d, '%Y%m%d').weekday() for d in gap90_dates]

# 關鍵基準錨點索引
idx_jan15 = cal_date_to_idx['20240115']
idx_feb01 = cal_date_to_idx['20240201']
idx_apr01 = cal_date_to_idx['20240401']
idx_apr30 = cal_date_to_idx['20240430']

routes_list = sorted(list(od_ts.keys()))
out_gap90_garch = {d: {} for d in gap90_dates}

# =============================================================================
# 逐條 OD 路線執行 GARCH 分層合成
# =============================================================================
for od in routes_list:
    parts = od.replace('-1_-1-', '-1_-1_').split('-') if od.startswith('-1_-1-') else od.split('-')
    orig = '-1_-1' if od.startswith('-1_-1-') else parts[0]
    dest = parts[1].replace('_', '-') if od.startswith('-1_-1-') else parts[1]
    is_diag = (orig == dest)

    raw_arr = od_ts[od]
    y_366 = np.full(total_days, np.nan, dtype=float)
    
    # 將歷史觀測數據對齊至 366 天日曆（排除異常日期）
    for i, d in enumerate(dates_str):
        if d in cal_date_to_idx and d not in EXCLUDED_DATES:
            v = raw_arr[i]
            if not np.isnan(v) and v >= 0:
                y_366[cal_date_to_idx[d]] = v

    # 1. 計算多時期基準錨點 (Anchors) 並擬合 Hermite 樣條中軸線 B(t)
    jan_vals = y_366[idx_jan15:idx_feb01]
    jan_valid = jan_vals[~np.isnan(jan_vals)]
    jan_anchor = float(np.mean(jan_valid)) if len(jan_valid) > 0 else 0.0

    post_vals = y_366[idx_apr01:cal_date_to_idx['20240630']+1]
    post_valid = post_vals[~np.isnan(post_vals)]
    post_anchor = float(np.mean(post_valid)) if len(post_valid) > 0 else jan_anchor

    if jan_anchor < 0.01 and post_anchor < 0.01:
        continue

    pre_vals = y_366[:cal_date_to_idx['20231231']]
    pre_valid = pre_vals[~np.isnan(pre_vals)]
    pre_anchor = float(np.mean(pre_valid)) if len(pre_valid) > 0 else jan_anchor

    end_vals = y_366[cal_date_to_idx['20240801']:]
    end_valid = end_vals[~np.isnan(end_vals)]
    end_anchor = float(np.mean(end_valid)) if len(end_valid) > 0 else post_anchor

    t_anchors = [0, cal_date_to_idx['20231231'], idx_jan15, idx_feb01, idx_apr01, cal_date_to_idx['20240501'], total_days-1]
    y_anchors = [pre_anchor, pre_anchor, jan_anchor, jan_anchor, post_anchor, post_anchor, end_anchor]
    spline = CubicHermiteSpline(t_anchors, y_anchors, [0]*len(t_anchors))
    b_366 = np.clip(spline(np.arange(total_days)), 0.01, None)

    # 2. 對角線停留流動：執行 GARCH 去污染 7 天週載波合成
    if is_diag:
        dow_vals = {d: [] for d in range(7)}
        for c_i, d_str in enumerate(cal_dates):
            if np.isnan(y_366[c_i]): 
                continue
            if d_str in JAPAN_HOLIDAYS or (c_i + 1 < total_days and cal_dates[c_i+1] in JAPAN_HOLIDAYS):
                continue
            dow = cal_dts[c_i].weekday()
            dow_vals[dow].append(y_366[c_i] - b_366[c_i])

        carrier_raw = np.array([np.mean(dow_vals[d]) if len(dow_vals[d]) >= 2 else 0.0 for d in range(7)])
        c_std = np.std(carrier_raw)
        carrier_norm = carrier_raw / (c_std + 1e-6) if c_std > 1e-4 else np.zeros(7)
        carrier_func = interp1d(np.arange(35), np.tile(carrier_norm, 5), kind='cubic', bounds_error=False, fill_value="extrapolate")

        amp_scale = (float(np.percentile(post_valid, 90)) - float(np.percentile(post_valid, 10))) / 2.0 if len(post_valid) >= 10 else 0.35 * post_anchor
        omega = ((0.5 * amp_scale) ** 2) * (1.0 - 0.25 - 0.65)
        beta = 0.65
        sigma2 = (0.8 * amp_scale) ** 2

        for t_step, (d_str, dow) in enumerate(zip(gap90_dates, gap90_dows)):
            c_idx = cal_date_to_idx[d_str]
            b_val = b_366[c_idx]

            sigma2 = omega + beta * sigma2
            curr_sigma = math.sqrt(max(1e-4, sigma2))
            phi_t = 0.35 * math.exp(-0.02 * t_step)
            hol_feat = get_holiday_features(d_str)

            effective_dow = 6 if hol_feat['is_holiday'] else (dow + phi_t)
            c_val = float(carrier_func(14.0 + effective_dow))
            hol_modifier = 0.15 * amp_scale if hol_feat['is_holiday_eve'] else 0.0

            pred_flow = max(0.0, b_val + curr_sigma * c_val + hol_modifier)
            if pred_flow > 0.1:
                if orig not in out_gap90_garch[d_str]:
                    out_gap90_garch[d_str][orig] = {}
                out_gap90_garch[d_str][orig][dest] = round(float(pred_flow), 4)
    else:
        # 3. 非對角線跨區流動：僅對實體活躍且平滑值 > 0.1 的路線輸出，0.1 以下均視為 0
        for t_step, d_str in enumerate(gap90_dates):
            base_val = base_data.get(d_str, {}).get(orig, {}).get(dest, None)
            if base_val is not None and float(base_val) > 0.1:
                if orig not in out_gap90_garch[d_str]:
                    out_gap90_garch[d_str][orig] = {}
                out_gap90_garch[d_str][orig][dest] = round(float(base_val), 4)

# =============================================================================
# 匯出為官方規範格式 TSV 檔案 (Date \t {Origin: {Destination: Flow}})
# =============================================================================
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TSV, 'w', encoding='utf-8') as f:
    for d_str in gap90_dates:
        od_dict = out_gap90_garch[d_str]
        f.write(f"{d_str}\t{od_dict}\n")

print(f"✅ Generated and saved GARCH predictions to: {OUT_TSV}")
