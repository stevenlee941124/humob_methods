"""
===============================================================================
HuMob 2026: Layer 2 - Adaptive Regularity Quantification & Periodic Gating Module
===============================================================================
核心功能：
  1. 規律度量化 (Regularity Index S_reg):
     - 結合 7 天滯後自相關 r_7 與 7 天週期可解釋變異比 R^2_weekly。
  2. 自適應門控 G_i ∈ [0.0, 1.0]:
     - 規律網格 (S_reg ≥ 0.45) -> G_i ≈ 1.0，給予 GARCH 式純淨 7 天通勤齒輪。
     - 不規律網格 (S_reg < 0.20) -> G_i ≈ 0.0，完全關閉 ψ，防止偶發噪聲被放大成虛假鋸齒。
  3. 日曆去污染 7 天週期載波 ψ_i(DOW_t + φ_t) 提純與插值。
===============================================================================
"""
import math
import numpy as np
from scipy.interpolate import interp1d
from japan_calendar import JAPAN_HOLIDAYS, get_holiday_features

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def compute_regularity_and_carrier(y_366, b_366, cal_dates, cal_dts):
    """
    計算該 OD 路線的規律度評分 S_reg、自適應門控 G_i、7天週期載波函數 psi_func 與週波標準差 sigma_weekly
    """
    total = len(cal_dates)
    
    # 收集扣除 Baseline 後的有效殘差與星期分佈
    dow_resids = {d: [] for d in range(7)}
    all_resids = []
    
    for ci in range(total):
        d_str = cal_dates[ci]
        y_val = y_366[ci]
        b_val = b_366[ci]
        
        if np.isnan(y_val) or np.isnan(b_val) or d_str in EXCLUDED_DATES:
            continue
        if d_str in JAPAN_HOLIDAYS or (ci + 1 < total and cal_dates[ci + 1] in JAPAN_HOLIDAYS):
            continue
        # 排除 2~4 月盲區
        if '20240201' <= d_str <= '20240430':
            continue
            
        r = float(y_val - b_val)
        dow = cal_dts[ci].weekday()
        dow_resids[dow].append(r)
        all_resids.append(r)
        
    if len(all_resids) < 14:
        return 0.0, 0.0, None, 0.0
        
    all_resids = np.array(all_resids)
    var_total = float(np.var(all_resids))
    if var_total < 1e-6:
        return 0.0, 0.0, None, 0.0

    # 1. 計算每週 7 天平均載波均值
    dow_means = np.array([np.mean(dow_resids[d]) if len(dow_resids[d]) >= 2 else 0.0 for d in range(7)])
    
    # 計算各星期內部殘差方差，得到 R^2_weekly
    unexplained_resids = []
    for d in range(7):
        unexplained_resids.extend([r - dow_means[d] for r in dow_resids[d]])
    var_unexplained = float(np.var(unexplained_resids)) if unexplained_resids else var_total
    r2_weekly = max(0.0, min(1.0, 1.0 - var_unexplained / (var_total + 1e-8)))
    
    # 2. 計算 7 天自相關 r_7
    r7_pairs_x, r7_pairs_y = [], []
    for ci in range(total - 7):
        if not np.isnan(y_366[ci]) and not np.isnan(y_366[ci+7]) and not np.isnan(b_366[ci]) and not np.isnan(b_366[ci+7]):
            d1, d2 = cal_dates[ci], cal_dates[ci+7]
            if d1 not in EXCLUDED_DATES and d2 not in EXCLUDED_DATES:
                if not ('20240201' <= d1 <= '20240430') and not ('20240201' <= d2 <= '20240430'):
                    r7_pairs_x.append(y_366[ci] - b_366[ci])
                    r7_pairs_y.append(y_366[ci+7] - b_366[ci+7])
    
    if len(r7_pairs_x) >= 20:
        rx = np.array(r7_pairs_x)
        ry = np.array(r7_pairs_y)
        cov = np.mean((rx - np.mean(rx)) * (ry - np.mean(ry)))
        denom = (np.std(rx) * np.std(ry)) + 1e-8
        r7 = float(cov / denom)
    else:
        r7 = 0.0
        
    r7_score = max(0.0, min(1.0, r7))
    
    # 3. 綜合規律度評分 S_reg ∈ [0, 1]
    s_reg = 0.6 * r2_weekly + 0.4 * r7_score
    
    # 4. 自適應門控 G_i ∈ [0, 1] (採用平滑 Sigmoid 過渡)
    # 當 S_reg < 0.20 時，G_i ≈ 0.0；當 S_reg ≥ 0.45 時，G_i ≈ 1.0
    gate_g = 1.0 / (1.0 + math.exp(-12.0 * (s_reg - 0.30)))
    if s_reg < 0.15:
        gate_g = 0.0

    # 5. 構建三次樣條插值載波 ψ 函數
    carrier_std = np.std(dow_means)
    if carrier_std > 1e-4:
        carrier_norm = dow_means / carrier_std
    else:
        carrier_norm = np.zeros(7)
        
    carrier_func = interp1d(
        np.arange(35),
        np.tile(carrier_norm, 5),
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    sigma_weekly = float(carrier_std)
    
    return s_reg, gate_g, carrier_func, sigma_weekly
