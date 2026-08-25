"""
===============================================================================
HuMob 2026: Unified C^1 Smooth Physics-Consistent Baseline Module
===============================================================================
核心數學原則：
  1. 全域 366 天 C^1 連續平滑：
     - 徹底杜絕任何「分段拼接」產生的折點、斷崖下墜 (如 5/1 垂直下墜) 與突然暴走！
  2. 零值真實感知 (Zero-Awareness):
     - 包含無人流天數 (0.0)，讓稀疏網格 Baseline 自然平滑貼近 0 軸規律 (0.2 ~ 0.8 人)。
  3. 盲區 (2~4月) 對數飽和單調過渡 (Log-Saturation Smooth Transition):
     - 1 月末水平 -> 5 月初水平 以對數飽和規律平滑過渡。
  4. 全年一體化 1D 高斯濾波 (sigma=6.0):
     - 保證整條 Baseline 毫無折角，在 1/1, 1/31, 5/1 全流程極度平滑！
===============================================================================
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

FLOW_THRESHOLD = 0.05


def compute_full_baseline(y_366_input, cal_dates, cal_date_to_idx):
    total = len(cal_dates)
    b_366 = np.full(total, np.nan, dtype=np.float64)

    if np.sum(y_366_input > 0.05) < 5:
        return b_366, 0.0, "Dead Zero"

    i_jan01 = cal_date_to_idx['20240101']
    i_feb01 = cal_date_to_idx['20240201']
    i_apr01 = cal_date_to_idx['20240401']
    i_may01 = cal_date_to_idx['20240501']

    # 1. 計算真實觀測日的 7 天滾動均值 (排除官方排除日與 2~3月預測盲區)
    rolling = np.full(total, np.nan, dtype=np.float64)
    for i in range(total):
        d_str = cal_dates[i]
        if d_str in EXCLUDED_DATES or (i_feb01 <= i < i_apr01):
            continue
        win = [y_366_input[j] for j in range(max(0, i - 3), min(total, i + 4))
               if cal_dates[j] not in EXCLUDED_DATES and not (i_feb01 <= j < i_apr01)]
        if win:
            rolling[i] = np.mean(win)

    # 2. 1月個體受災極值動態偵測 (左側起點錨點)
    pre_slice = [rolling[j] for j in range(max(0, i_jan01 - 28), i_jan01) if not np.isnan(rolling[j])]
    y_pre = float(np.mean(pre_slice)) if pre_slice else 0.0

    jan_vals = [(j, rolling[j]) for j in range(i_jan01, i_feb01) if not np.isnan(rolling[j])]
    
    if jan_vals and y_pre > 1.0:
        j_min, v_min = min(jan_vals, key=lambda x: x[1])
        j_max, v_max = max(jan_vals, key=lambda x: x[1])
        
        if v_max > 1.25 * y_pre and j_max <= i_jan01 + 14:
            shock_type = 'surge'
            y_shock = v_max
        elif v_min < 0.85 * y_pre:
            shock_type = 'depression'
            y_shock = v_min
        else:
            shock_type = 'normal'
            jan_end_slice = [rolling[j] for j in range(i_feb01 - 7, i_feb01) if not np.isnan(rolling[j])]
            y_shock = float(np.mean(jan_end_slice)) if jan_end_slice else y_pre
    else:
        shock_type = 'normal'
        jan_end_slice = [rolling[j] for j in range(i_feb01 - 7, i_feb01) if not np.isnan(rolling[j])]
        y_shock = float(np.mean(jan_end_slice)) if jan_end_slice else y_pre

    # 3. 4月真實基準 (右側終點錨點，緊鄰 3/31 盲區結束)
    apr_clean_slice = [rolling[j] for j in range(i_apr01, min(total, i_apr01 + 14)) if not np.isnan(rolling[j])]
    y_apr = float(np.mean(apr_clean_slice)) if apr_clean_slice else y_shock

    # 4. 2~3月 (60天盲區) 指數復甦 / 避難消退插值
    total_blind = float(i_apr01 - i_feb01) # 60 days
    for ci in range(i_feb01, i_apr01):
        t = float(ci - i_feb01)
        if shock_type == 'surge':
            w = math.exp(-3.0 * (t / total_blind))
            rolling[ci] = y_apr + (y_shock - y_apr) * w
        else:
            w = math.log(1.0 + 9.0 * (t / total_blind)) / math.log(10.0)
            rolling[ci] = y_shock + (y_apr - y_shock) * w

    # 5. 全年一體化高斯濾波 (消除所有拼接折角與斷崖)
    valid_mask = ~np.isnan(rolling)
    if valid_mask.sum() < 5:
        return b_366, 0.0, "Dead Zero"

    x_all = np.arange(total)
    rolling_filled = np.interp(x_all, x_all[valid_mask], rolling[valid_mask])
    b_366 = gaussian_filter1d(rolling_filled, sigma=4.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    return b_366, 0.05, f"April_Anchored_Exp_{shock_type}"
