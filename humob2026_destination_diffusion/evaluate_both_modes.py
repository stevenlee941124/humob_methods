"""
===============================================================================
Comprehensive Evaluation for Pure & Non-Pure Diffusion in Both Modes:
1. Filtered (有刪除 0 流量，限定 1,836 條活躍路線)
2. Full Bounding Box (未刪除 0 流量，包含 2,178,576 條全部網格對)
===============================================================================
"""
import sys, math, pickle, numpy as np
from pathlib import Path
from tabulate import tabulate
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from nine_class_baseline import compute_9class_baseline

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176

MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

all_bbox_grids = [f"{x}_{y}" for x in range(MIN_X, MAX_X + 1) for y in range(MIN_Y, MAX_Y + 1)]
bbox_set = set(all_bbox_grids)

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

def run_evaluation(gt_data, pred_data, od_ts, dates_str, filter_zero_flow=True):
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    
    eval_month_prefix = '202404'
    eval_dates = [d for d in gt_data.keys() if d.startswith(eval_month_prefix) and d not in EXCLUDED_DATES]
    eval_dates.sort()
    n_days = len(eval_dates)
    
    class_names = {
        1: "Persistent Zero",
        2: "Temporary Increase",
        3: "Persistent Decrease",
        4: "Partial Recovery",
        5: "Fully Recovered",
        6: "Stable Inflow",
        7: "Emergent/Temporary activity",
        8: "Partial Dissipation",
        9: "Persistent Increase"
    }

    if filter_zero_flow:
        # Mode 1: Active routes within bounding box (1836 routes)
        od_pairs = []
        for pair in od_ts.keys():
            parts = pair.split('-')
            orig, dest = (f"{parts[0]}-{parts[1]}", parts[2]) if len(parts) == 3 else (parts[0], parts[1])
            if orig in bbox_set and dest in bbox_set:
                od_pairs.append(pair)
    else:
        # Mode 2: Full 1476x1476 = 2,178,576 pairs
        od_pairs = [f"{orig}-{dest}" for orig in all_bbox_grids for dest in all_bbox_grids]

    pair_classes = {}
    class_counts = {cid: {"diag": 0, "off": 0} for cid in range(1, 10)}
    total_counts = {"diag": 0, "off": 0}

    for pair in od_pairs:
        raw = od_ts.get(pair)
        if raw is None:
            cid = 1
        else:
            y_366 = np.zeros(366, dtype=np.float64)
            for oi, v in enumerate(raw):
                d_str = dates_str[oi]
                if d_str in cal_date_to_idx:
                    y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0
            _, _, cid = compute_9class_baseline(y_366, cal_dates, cal_date_to_idx)
            
        pair_classes[pair] = cid
        parts = pair.split('-')
        orig, dest = (f"{parts[0]}-{parts[1]}", parts[2]) if len(parts) == 3 else (parts[0], parts[1])
        if orig == dest:
            class_counts[cid]["diag"] += 1
            total_counts["diag"] += 1
        else:
            class_counts[cid]["off"] += 1
            total_counts["off"] += 1

    class_stats = {cid: {"diag_se": 0.0, "off_se": 0.0} for cid in range(1, 10)}
    total_diag_se = 0.0
    total_off_se = 0.0

    for d in eval_dates:
        gt_day = gt_data.get(d, {})
        p_day = pred_data.get(d, {})

        for pair in od_pairs:
            parts = pair.split('-')
            orig, dest = (f"{parts[0]}-{parts[1]}", parts[2]) if len(parts) == 3 else (parts[0], parts[1])
            y_true = gt_day.get(orig, {}).get(dest, 0.0) or 0.0
            y_pred = p_day.get(orig, {}).get(dest, 0.0) or 0.0
            diff = y_true - y_pred
            se = diff * diff
            cid = pair_classes[pair]

            if orig == dest:
                class_stats[cid]["diag_se"] += se
                total_diag_se += se
            else:
                class_stats[cid]["off_se"] += se
                total_off_se += se

    table = []
    for cid in range(1, 10):
        cname = class_names[cid]
        d_cnt = class_counts[cid]["diag"]
        o_cnt = class_counts[cid]["off"]
        tot_cnt = d_cnt + o_cnt

        d_den = d_cnt * n_days if d_cnt > 0 else 1
        o_den = o_cnt * n_days if o_cnt > 0 else 1

        rmse_d = math.sqrt(class_stats[cid]["diag_se"] / d_den) if d_cnt > 0 else 0.0
        rmse_o = math.sqrt(class_stats[cid]["off_se"] / o_den) if o_cnt > 0 else 0.0
        nrmse_d = rmse_d / MEAN_ACTUAL_DIAG
        nrmse_o = rmse_o / MEAN_ACTUAL_OFFDIAG
        comb = 0.5 * (nrmse_d + nrmse_o)

        table.append([
            cid, cname, f"{tot_cnt} 條",
            f"{rmse_d:.2f} 人", f"{nrmse_d:.5f}",
            f"{rmse_o:.2f} 人", f"{nrmse_o:.5f}",
            f"{comb:.5f}"
        ])

    # Total row
    t_d_cnt = total_counts["diag"]
    t_o_cnt = total_counts["off"]
    t_tot = t_d_cnt + t_o_cnt

    t_d_den = t_d_cnt * n_days if t_d_cnt > 0 else 1
    t_o_den = t_o_cnt * n_days if t_o_cnt > 0 else 1

    t_rmse_d = math.sqrt(total_diag_se / t_d_den) if t_d_cnt > 0 else 0.0
    t_rmse_o = math.sqrt(total_off_se / t_o_den) if t_o_cnt > 0 else 0.0
    t_nrmse_d = t_rmse_d / MEAN_ACTUAL_DIAG
    t_nrmse_o = t_rmse_o / MEAN_ACTUAL_OFFDIAG
    t_comb = 0.5 * (t_nrmse_d + t_nrmse_o)

    table.append([
        "Total", "All Routes", f"{t_tot} 條",
        f"{t_rmse_d:.2f} 人", f"{t_nrmse_d:.5f}",
        f"{t_rmse_o:.2f} 人", f"{t_nrmse_o:.5f}",
        f"{t_comb:.5f}"
    ])

    return table

def main():
    gt_tsv_path = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
    od_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
    dates_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
    
    pred_mix_path = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions.tsv'
    pred_pure_path = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions_pure.tsv'

    print("Loading data...")
    gt_data = parse_tsv(gt_tsv_path)
    with open(od_pkl_path, 'rb') as f: od_ts = pickle.load(f)
    with open(dates_pkl_path, 'rb') as f: dates_str = pickle.load(f)
    
    headers = ["id", "class", "數量", "RMSE_diag", "NRMSE_diag", "RMSE_off", "NRMSE_off", "Combined NRMSE"]

    # 1. Non-Pure (Mix ψ7 + Diff)
    print("[*] Evaluating Non-Pure (Mix ψ7 + Diffusion)...")
    pred_mix = parse_tsv(pred_mix_path)
    t1 = run_evaluation(gt_data, pred_mix, od_ts, dates_str, filter_zero_flow=True)
    t2 = run_evaluation(gt_data, pred_mix, od_ts, dates_str, filter_zero_flow=False)

    # 2. Pure (Pure Diffusion)
    print("[*] Evaluating Pure (Pure Diffusion)...")
    pred_pure = parse_tsv(pred_pure_path)
    t3 = run_evaluation(gt_data, pred_pure, od_ts, dates_str, filter_zero_flow=True)
    t4 = run_evaluation(gt_data, pred_pure, od_ts, dates_str, filter_zero_flow=False)

    print("\n" + "="*95)
    print("📋 [表格 1] 非 Pure 方法 (ψ7 + Diffusion) - 【有刪除 0 流量】(1,836 條活躍路線)")
    print("="*95)
    print(tabulate(t1, headers=headers, tablefmt="grid"))

    print("\n" + "="*95)
    print("📋 [表格 2] 非 Pure 方法 (ψ7 + Diffusion) - 【沒刪除 0 流量】(2,178,576 條全部網格)")
    print("="*95)
    print(tabulate(t2, headers=headers, tablefmt="grid"))

    print("\n" + "="*95)
    print("📋 [表格 3] Pure 方法 (純 Diffusion) - 【有刪除 0 流量】(1,836 條活躍路線)")
    print("="*95)
    print(tabulate(t3, headers=headers, tablefmt="grid"))

    print("\n" + "="*95)
    print("📋 [表格 4] Pure 方法 (純 Diffusion) - 【沒刪除 0 流量】(2,178,576 條全部網格)")
    print("="*95)
    print(tabulate(t4, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
