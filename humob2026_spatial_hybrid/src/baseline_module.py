"""
===============================================================================
HuMob 2026: Layer 1 - Zero-Aware C^1 Smooth Physics Baseline Module
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
    # 🌟 關鍵物理修復：5/1~5/6 為日本黃金週連假，流量會出現顯著假日深谷。
    # 必須取 5/7~5/21 排除黃金週後的常態通勤工作日作為 5 月中軸錨點！
    jan_end_slice = [rolling[j] for j in range(i_feb01 - 14, i_feb01) if not np.isnan(rolling[j])]
    may_clean_slice = [rolling[j] for j in range(i_may01 + 6, min(total, i_may01 + 21)) if not np.isnan(rolling[j])]

    y_jan_end = float(np.mean(jan_end_slice)) if jan_end_slice else 0.0
    y_may_beg = float(np.mean(may_clean_slice)) if may_clean_slice else y_jan_end

    total_blind = float(i_may01 - i_feb01)
    for ci in range(i_feb01, i_may01):
        t = float(ci - i_feb01)
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

    return b_366, 0.05, "Unified_C1_Smooth_Baseline"
