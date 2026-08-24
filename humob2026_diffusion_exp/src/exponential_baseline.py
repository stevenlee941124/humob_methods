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
    i_may01 = cal_date_to_idx['20240501']

    # 1. 計算真實觀測日的 7 天滾動均值 (排除官方排除日與 2~4月盲區)
    rolling = np.full(total, np.nan, dtype=np.float64)
    for i in range(total):
        d_str = cal_dates[i]
        if d_str in EXCLUDED_DATES or (i_feb01 <= i < i_may01):
            continue
        win = [y_366_input[j] for j in range(max(0, i - 3), min(total, i + 4))
               if cal_dates[j] not in EXCLUDED_DATES and not (i_feb01 <= j < i_may01)]
        if win:
            rolling[i] = np.mean(win)

    # 2. 盲區 (2/1 ~ 4/30) 對數飽和單調過渡
    jan_end_slice = [rolling[j] for j in range(i_feb01 - 7, i_feb01) if not np.isnan(rolling[j])]
    may_beg_slice = [rolling[j] for j in range(i_may01, i_may01 + 7) if not np.isnan(rolling[j])]

    y_jan_end = float(np.mean(jan_end_slice)) if jan_end_slice else 0.0
    y_may_beg = float(np.mean(may_beg_slice)) if may_beg_slice else y_jan_end

    total_blind = float(i_may01 - i_feb01)
    for ci in range(i_feb01, i_may01):
        t = float(ci - i_feb01)
        # 對數飽和過渡權重 w(t) ∈ [0, 1]
        w = math.log(1.0 + 9.0 * (t / total_blind)) / math.log(10.0)
        rolling[ci] = y_jan_end + (y_may_beg - y_jan_end) * w

    # 3. 全年一體化高斯濾波 (消除所有拼接折角與斷崖)
    valid_mask = ~np.isnan(rolling)
    if valid_mask.sum() < 5:
        return b_366, 0.0, "Dead Zero"

    x_all = np.arange(total)
    rolling_filled = np.interp(x_all, x_all[valid_mask], rolling[valid_mask])
    b_366 = gaussian_filter1d(rolling_filled, sigma=6.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    return b_366, 0.05, "Unified_Smooth_Baseline"
