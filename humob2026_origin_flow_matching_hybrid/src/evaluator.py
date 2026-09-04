import csv
import sys
import numpy as np

csv.field_size_limit(sys.maxsize if sys.platform != 'win32' else 2147483647)

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

TOTAL_EVAL_DESTINATIONS = 1476
TOTAL_OFFDIAG_PAIRS = TOTAL_EVAL_DESTINATIONS * (TOTAL_EVAL_DESTINATIONS - 1)


def in_eval_bbox(grid_str):
    if grid_str == '-1_-1':
        return False
    parts = grid_str.split('_')
    if len(parts) != 2:
        return False
    try:
        x, y = int(parts[0]), int(parts[1])
        return (30 <= x <= 70) and (35 <= y <= 70)
    except ValueError:
        return False


def parse_od_dict(raw_str):
    if not raw_str:
        return {}
    clean = raw_str.replace(': NA', ': None').replace(':NA', ':None')
    try:
        data = eval(clean, {"__builtins__": {}}, {"None": None})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_ground_truth(gt_path, eval_month_prefix='202404'):
    gt_by_date = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            if not date_str.startswith(eval_month_prefix):
                continue
            if date_str in EXCLUDED_DATES:
                continue
            od_data = parse_od_dict(row[1])
            if od_data:
                gt_by_date[date_str] = od_data
    return gt_by_date


def load_predictions(pred_path, target_dates):
    pred_by_date = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            if date_str in target_dates:
                pred_by_date[date_str] = parse_od_dict(row[1])
    return pred_by_date


def evaluate_predictions(gt_path, pred_path, eval_month_prefix='202404'):
    gt = load_ground_truth(gt_path, eval_month_prefix)
    eval_dates = sorted(list(gt.keys()))
    N_eval_days = len(eval_dates)

    if N_eval_days == 0:
        return {'error': 'No valid evaluation dates found in ground truth.'}

    target_dates = set(eval_dates)
    pred = load_predictions(pred_path, target_dates)

    eval_dest_grids = [f"{x}_{y}" for x in range(30, 71) for y in range(35, 71)]

    diag_sq_errors = []

    for d in eval_dates:
        gt_day = gt.get(d, {})
        pred_day = pred.get(d, {})

        for g in eval_dest_grids:
            y_true = gt_day.get(g, {}).get(g, 0.0) or 0.0
            y_pred = pred_day.get(g, {}).get(g, 0.0) or 0.0
            diag_sq_errors.append((y_true - y_pred) ** 2)

    total_diag_elements = N_eval_days * TOTAL_EVAL_DESTINATIONS
    rmse_diag = np.sqrt(np.sum(diag_sq_errors) / total_diag_elements)
    nrmse_diag = rmse_diag / MEAN_ACTUAL_DIAG

    offdiag_sq_error_sum = 0.0

    for d in eval_dates:
        gt_day = gt.get(d, {})
        pred_day = pred.get(d, {})

        active_origins = set(gt_day.keys()) | set(pred_day.keys())

        for o in active_origins:
            if not in_eval_bbox(o):
                continue

            gt_dests = gt_day.get(o, {})
            pred_dests = pred_day.get(o, {})

            all_dests = set(gt_dests.keys()) | set(pred_dests.keys())

            for dest in all_dests:
                if not in_eval_bbox(dest):
                    continue
                if o == dest:
                    continue

                y_true = gt_dests.get(dest, 0.0) or 0.0
                y_pred = pred_dests.get(dest, 0.0) or 0.0

                offdiag_sq_error_sum += (y_true - y_pred) ** 2

    total_offdiag_elements = N_eval_days * TOTAL_OFFDIAG_PAIRS
    rmse_off = np.sqrt(offdiag_sq_error_sum / total_offdiag_elements)
    nrmse_off = rmse_off / MEAN_ACTUAL_OFFDIAG

    combined_nrmse = 0.5 * (nrmse_diag + nrmse_off)

    return {
        'eval_days': float(N_eval_days),
        'rmse_diag': float(rmse_diag),
        'nrmse_diag': float(nrmse_diag),
        'rmse_off': float(rmse_off),
        'nrmse_off': float(nrmse_off),
        'combined_nrmse': float(combined_nrmse),
    }
