"""
===============================================================================
HuMob 2026: Official Evaluator (26.57 / 0.0176 Constants)
===============================================================================
"""

import sys
import math
import numpy as np
from pathlib import Path

# 官方最新公佈之標準化分母常數
MEAN_ACTUAL_DIAG = 26.57       # 對角線人流真實均值
MEAN_ACTUAL_OFFDIAG = 0.0176   # 非對角線流動真實均值

# 空間範圍定義 (X: 30~70, Y: 35~70 -> 共 41 * 36 = 1,476 個網格)
MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

all_bbox_grids = [f"{x}_{y}" for x in range(MIN_X, MAX_X + 1) for y in range(MIN_Y, MAX_Y + 1)]
n_grids = len(all_bbox_grids)              # 1,476
n_diag_total = n_grids                      # 1,476
n_offdiag_total = n_grids * (n_grids - 1)  # 2,177,100

def parse_tsv(filepath):
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

def evaluate_predictions(gt_tsv_path, pred_tsv_path, eval_month_prefix='202404'):
    """
    計算預測結果與真實值在 4 月份的各項評估指標
    """
    gt_data = parse_tsv(gt_tsv_path)
    pred_data = parse_tsv(pred_tsv_path)
    
    # 篩選評估日期 (4 月且非排除日)
    eval_dates = [d for d in gt_data.keys() if d.startswith(eval_month_prefix) and d not in EXCLUDED_DATES]
    eval_dates.sort()
    
    if not eval_dates:
        return {"error": "No valid evaluation dates found"}
        
    diag_se_list = []
    offdiag_se_list = []
    
    for d in eval_dates:
        gt_day = gt_data.get(d, {})
        pred_day = pred_data.get(d, {})
        
        # 1. 對角線誤差
        day_diag_se = 0.0
        for g in all_bbox_grids:
            y_true = gt_day.get(g, {}).get(g, 0.0) or 0.0
            y_pred = pred_day.get(g, {}).get(g, 0.0) or 0.0
            diff = y_true - y_pred
            day_diag_se += diff * diff
        diag_se_list.append(day_diag_se / n_diag_total)
        
        # 2. 非對角線誤差
        # 遍歷活躍 OD 路線與 GT OD 路線
        day_offdiag_se = 0.0
        # 找出所有在 GT 或 Pred 中有值的 off-diag 對
        candidate_origins = set(gt_day.keys()) | set(pred_day.keys())
        nonzero_offdiag_count = 0
        
        for orig in candidate_origins:
            if orig not in all_bbox_grids:
                continue
            gt_dests = gt_day.get(orig, {})
            pred_dests = pred_day.get(orig, {})
            candidate_dests = set(gt_dests.keys()) | set(pred_dests.keys())
            
            for dest in candidate_dests:
                if dest not in all_bbox_grids or orig == dest:
                    continue
                y_true = gt_dests.get(dest, 0.0) or 0.0
                y_pred = pred_dests.get(dest, 0.0) or 0.0
                diff = y_true - y_pred
                day_offdiag_se += diff * diff
                nonzero_offdiag_count += 1
                
        offdiag_se_list.append(day_offdiag_se / n_offdiag_total)
        
    rmse_diag = math.sqrt(np.mean(diag_se_list))
    nrmse_diag = rmse_diag / MEAN_ACTUAL_DIAG
    
    rmse_off = math.sqrt(np.mean(offdiag_se_list))
    nrmse_off = rmse_off / MEAN_ACTUAL_OFFDIAG
    
    combined_nrmse = 0.5 * (nrmse_diag + nrmse_off)
    
    return {
        "eval_days": len(eval_dates),
        "rmse_diag": rmse_diag,
        "nrmse_diag": nrmse_diag,
        "rmse_off": rmse_off,
        "nrmse_off": nrmse_off,
        "combined_nrmse": combined_nrmse
    }
