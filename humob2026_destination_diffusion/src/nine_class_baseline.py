"""
===============================================================================
HuMob 2026: 9-Class Disaster Dynamics & Dual-Anchor OD Transition Baseline
===============================================================================
Stage 1: 9-Class Classification + IQR Outlier Cleaning
Stage 2: Dynamic Dual-Anchor (Jan 20-31 -> Apr 01-14) Cubic S-Curve Transition
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def clean_iqr_outliers(series):
    """階段一：四分位距 (IQR) 剔除極端離群值"""
    valid = series[~np.isnan(series) & (series > 0.0)]
    if len(valid) < 8:
        return series
    q25, q75 = np.percentile(valid, [25, 75])
    iqr = q75 - q25
    lower = max(0.0, q25 - 2.5 * iqr)
    upper = q75 + 2.5 * iqr
    cleaned = np.copy(series)
    cleaned[(cleaned < lower) | (cleaned > upper)] = np.nan
    return cleaned

def compute_9class_baseline(y_366_raw, cal_dates, cal_date_to_idx):
    """
    計算 9 大災害類別物理動態 Baseline
    """
    total = len(cal_dates)
    b_366 = np.zeros(total, dtype=np.float64)

    if np.sum(y_366_raw > 0.05) < 5:
        return b_366, "Class 1: Persistent Zero", 1

    i_jan01 = cal_date_to_idx['20240101']
    i_jan20 = cal_date_to_idx['20240120']
    i_feb01 = cal_date_to_idx['20240201']
    i_apr01 = cal_date_to_idx['20240401']
    i_apr14 = min(total, cal_date_to_idx['20240414'])

    # 1. IQR 離群清洗
    y_clean = clean_iqr_outliers(y_366_raw)

    # 2. 計算真實觀測日 7 天滾動均值 (排除官方異常日與 2~3 月盲區)
    rolling = np.full(total, np.nan, dtype=np.float64)
    for i in range(total):
        d_str = cal_dates[i]
        if d_str in EXCLUDED_DATES or (i_feb01 <= i < i_apr01):
            continue
        win = [y_clean[j] for j in range(max(0, i - 3), min(total, i + 4))
               if cal_dates[j] not in EXCLUDED_DATES and not (i_feb01 <= j < i_apr01) and not np.isnan(y_clean[j])]
        if win:
            rolling[i] = np.mean(win)

    # 3. 提取雙錨點與關鍵時段特徵
    # P_pre (震前常態: 2024-01-01 前 28 天)
    pre_slice = [rolling[j] for j in range(max(0, i_jan01 - 28), i_jan01) if not np.isnan(rolling[j])]
    y_pre = float(np.mean(pre_slice)) if pre_slice else 0.0

    # 1月震後極值
    jan_vals = [(j, rolling[j]) for j in range(i_jan01, i_feb01) if not np.isnan(rolling[j])]
    j_max, v_max = max(jan_vals, key=lambda x: x[1]) if jan_vals else (i_jan01, y_pre)

    # P_jan (災後應急左錨點: 2024-01-20 ~ 2024-01-31)
    jan_anchor_slice = [rolling[j] for j in range(i_jan20, i_feb01) if not np.isnan(rolling[j])]
    y_jan_anchor = float(np.mean(jan_anchor_slice)) if jan_anchor_slice else y_pre

    # P_apr (初期待復原右錨點: 2024-04-01 ~ 2024-04-14)
    apr_anchor_slice = [rolling[j] for j in range(i_apr01, i_apr14) if not np.isnan(rolling[j])]
    y_apr_anchor = float(np.mean(apr_anchor_slice)) if apr_anchor_slice else y_jan_anchor

    # 4. 階段一：嚴格區分 1 到 9 大類別 (Strict 1-to-9 Disaster Dynamics Classification)
    if y_pre < 0.05 and y_apr_anchor < 0.05:
        cls_id = 1
        cls_name = "Class 1: (人流量低下區) Persistent Zero"
    elif y_apr_anchor > 1.25 * max(y_pre, 1.0) and y_apr_anchor >= y_jan_anchor:
        cls_id = 9
        cls_name = "Class 9: (長期增加區) Persistent Increase"
    elif y_jan_anchor > 1.15 * max(y_apr_anchor, 1.0) and y_apr_anchor > 1.10 * max(y_pre, 1.0):
        cls_id = 8
        cls_name = "Class 8: (部分消退區) Partial Dissipation"
    elif v_max > 1.25 * max(y_pre, 1.0) and abs(y_apr_anchor - y_pre) <= 0.25 * max(y_pre, 1.0) and j_max <= i_jan01 + 14:
        cls_id = 7
        cls_name = "Class 7: (短期流入暴增後消退) Short-term Surge & Dissipation"
    elif v_max > 1.25 * max(y_pre, 1.0) and y_jan_anchor > 1.10 * max(y_pre, 1.0):
        cls_id = 2
        cls_name = "Class 2: (災後臨時避難區) Temporary Shelter & Assembly"
    elif y_apr_anchor < 0.70 * max(y_pre, 1.0) and y_apr_anchor <= y_jan_anchor:
        cls_id = 3
        cls_name = "Class 3: (地震重災與長期衰退區) Heavy Damage & Long-term Decline"
    elif y_jan_anchor < 0.80 * max(y_pre, 1.0) and y_apr_anchor < 0.90 * max(y_pre, 1.0) and y_apr_anchor > y_jan_anchor:
        cls_id = 4
        cls_name = "Class 4: (部分恢復區) Partial Recovery"
    elif y_jan_anchor < 0.85 * max(y_pre, 1.0) and y_apr_anchor >= 0.85 * max(y_pre, 1.0) and y_apr_anchor > y_jan_anchor:
        cls_id = 5
        cls_name = "Class 5: (大致恢復區) General Recovered"
    else:
        cls_id = 6
        cls_name = "Class 6: (常態平穩區) Normal Steady"

    # 5. 階段二：動態雙錨點 OD 矩陣轉移 (Dynamic Dual-Anchor Transition)
    total_blind = float(i_apr01 - i_feb01) # 60 days
    for ci in range(i_feb01, i_apr01):
        tau = float(ci - i_feb01) / total_blind # tau in [0, 1]
        
        if cls_id == 1:
            rolling[ci] = 0.0
        elif cls_id == 7: # 指數衰減 dissipation (1 - e^-3tau)
            f_tau = (1.0 - math.exp(-3.0 * tau)) / (1.0 - math.exp(-3.0))
            rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau
        elif cls_id == 8: # 指數耗散模型 (1 - e^-2tau)
            f_tau = (1.0 - math.exp(-2.0 * tau)) / (1.0 - math.exp(-2.0))
            rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau
        elif cls_id == 9: # 平方根型加速增長 sqrt(tau)
            f_tau = math.sqrt(tau)
            rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau
        elif cls_id == 4: # 二次曲線緩慢回升 tau^2
            f_tau = tau * tau
            rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau
        elif cls_id in (2, 3, 5, 6): # 三次 Hermite S 曲線平滑轉移
            f_tau = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            rolling[ci] = y_jan_anchor + (y_apr_anchor - y_jan_anchor) * f_tau

    # 6. 一體化平滑濾波 (消除接縫噪聲)
    valid_mask = ~np.isnan(rolling)
    if valid_mask.sum() < 5:
        return np.zeros(total, dtype=np.float64), cls_name, cls_id

    x_all = np.arange(total)
    rolling_filled = np.interp(x_all, x_all[valid_mask], rolling[valid_mask])
    b_366 = gaussian_filter1d(rolling_filled, sigma=3.0, mode='nearest')
    b_366 = np.maximum(0.0, b_366)

    return b_366, cls_name, cls_id
