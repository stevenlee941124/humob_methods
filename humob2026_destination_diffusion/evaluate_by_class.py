import sys, math, pickle, numpy as np
from pathlib import Path
from tabulate import tabulate
from datetime import datetime, timedelta

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_by_class.py <prediction_tsv_path>")
        sys.exit(1)
        
    pred_tsv_path = sys.argv[1]
    gt_tsv_path = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
    od_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
    dates_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
    
    print("Loading data...")
    gt_data = parse_tsv(gt_tsv_path)
    pred_data = parse_tsv(pred_tsv_path)
    with open(od_pkl_path, 'rb') as f: od_ts = pickle.load(f)
    with open(dates_pkl_path, 'rb') as f: dates_str = pickle.load(f)
    
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    
    eval_month_prefix = '202404'
    eval_dates = [d for d in gt_data.keys() if d.startswith(eval_month_prefix) and d not in EXCLUDED_DATES]
    eval_dates.sort()
    
    # Evaluate ALL 1476x1476 = 2,178,576 pairs in the bounding box (Competition Range with zero flow included)
    print(f"Generating all {len(all_bbox_grids) * len(all_bbox_grids)} OD pairs in the bounding box...")
    
    od_pairs = []
    for orig in all_bbox_grids:
        for dest in all_bbox_grids:
            od_pairs.append(f"{orig}-{dest}")
                    
    print(f"Total OD pairs to evaluate: {len(od_pairs)}")
    
    # Class name mapping
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
    
    # Initialize accumulators
    class_stats = {cid: {"count": 0, "diag_se": 0.0, "diag_n": 0, "off_se": 0.0, "off_n": 0} for cid in range(1, 10)}
    total_stats = {"count": 0, "diag_se": 0.0, "diag_n": 0, "off_se": 0.0, "off_n": 0}
    
    # Map OD pair to class
    pair_classes = {}
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
        class_stats[cid]["count"] += 1
        total_stats["count"] += 1

    for d in eval_dates:
        gt_day = gt_data.get(d, {})
        pred_day = pred_data.get(d, {})
        
        for pair in od_pairs:
            parts = pair.split('-')
            if len(parts) == 3: # Handle '-1_-1-dest'
                orig = f"{parts[0]}-{parts[1]}"
                dest = parts[2]
            else:
                orig, dest = parts[0], parts[1]
                
            y_true = gt_day.get(orig, {}).get(dest, 0.0) or 0.0
            y_pred = pred_day.get(orig, {}).get(dest, 0.0) or 0.0
            
            diff = y_true - y_pred
            se = diff * diff
            
            cid = pair_classes[pair]
            
            if orig == dest:
                class_stats[cid]["diag_se"] += se
                class_stats[cid]["diag_n"] += 1
                total_stats["diag_se"] += se
                total_stats["diag_n"] += 1
            else:
                class_stats[cid]["off_se"] += se
                class_stats[cid]["off_n"] += 1
                total_stats["off_se"] += se
                total_stats["off_n"] += 1
                
    # Format Table
    table = []
    for cid in range(1, 10):
        st = class_stats[cid]
        cname = class_names[cid]
        count = f"{st['count']} 條"
        
        # We need to divide by total number of diag / offdiag grids to get MSE for the whole region
        # Wait, the official metric divides by ALL grids (1476) and ALL off-grids (1476*1475).
        # But if we break it down by class, we should divide by the TOTAL number of grids (so the sum of class MSEs = total MSE).
        # Wait, RMSE = sqrt( MSE ). If we calculate RMSE per class, the denominator must be the same as official (n_grids), 
        # but multiplied by eval_days.
        # Let's check: Official day MSE is sum(SE) / n_diag_total. Then mean over days.
        # So MSE = total_SE / (n_diag_total * n_days)
        
        n_days = len(eval_dates)
        n_diag_total = 1476
        n_off_total = 1476 * 1475
        
        # Calculate class specific RMSE using the official denominator, 
        # so that they represent their contribution to the total error?
        # OR using their own count? 
        # Let's look at the image: Class 6 has 6,368 routes. RMSE_diag = 5.07. 
        # If they divided by their own count, it would be much higher.
        # Actually, in the user's previous code, they probably divided the SE by the number of days, and then by their own route count?
        # Let's see: NRMSE = RMSE / MEAN_ACTUAL.
        # For class 6: RMSE_diag = 5.07. 5.07 / 26.57 = 0.19100. This perfectly matches the image!
        # What about RMSE_off? 0.93 / 0.0176 = 52.84... (Image says 52.96090, close enough, maybe mean is slightly different).
        
        # If RMSE_diag is 5.07, and it's the RMSE of the class, how is MSE calculated?
        # If they calculated MSE per route:
        # MSE_class = class_se / (class_diag_count * n_days) -- wait, a class only has a certain number of diag routes.
        
        # Let's just calculate it by taking the SE / (number of days * number of routes in this class for diag/off).
        # Wait, if a class has NO diag routes, what is RMSE_diag? 0.
        diag_den = st["diag_n"] if st["diag_n"] > 0 else 1
        off_den = st["off_n"] if st["off_n"] > 0 else 1
        
        rmse_diag = math.sqrt(st["diag_se"] / diag_den) if st["diag_n"] > 0 else 0.0
        rmse_off = math.sqrt(st["off_se"] / off_den) if st["off_n"] > 0 else 0.0
        
        nrmse_diag = rmse_diag / MEAN_ACTUAL_DIAG
        nrmse_off = rmse_off / MEAN_ACTUAL_OFFDIAG
        combined = 0.5 * (nrmse_diag + nrmse_off)
        
        table.append([
            cid, cname, count, 
            f"{rmse_diag:.2f} 人", f"{nrmse_diag:.5f}",
            f"{rmse_off:.2f} 人", f"{nrmse_off:.5f}",
            f"{combined:.5f}"
        ])
        
    # Total
    diag_den = total_stats["diag_n"] if total_stats["diag_n"] > 0 else 1
    off_den = total_stats["off_n"] if total_stats["off_n"] > 0 else 1
    
    rmse_diag = math.sqrt(total_stats["diag_se"] / diag_den) if total_stats["diag_n"] > 0 else 0.0
    rmse_off = math.sqrt(total_stats["off_se"] / off_den) if total_stats["off_n"] > 0 else 0.0
    
    nrmse_diag = rmse_diag / MEAN_ACTUAL_DIAG
    nrmse_off = rmse_off / MEAN_ACTUAL_OFFDIAG
    combined = 0.5 * (nrmse_diag + nrmse_off)
    
    table.append([
        "Total", "All Routes", f"{total_stats['count']} 條", 
        f"{rmse_diag:.2f} 人", f"{nrmse_diag:.5f}",
        f"{rmse_off:.2f} 人", f"{nrmse_off:.5f}",
        f"{combined:.5f}"
    ])
    
    print("\n" + "="*90)
    print("預測結果(去除0流量路線) :")
    headers = ["id", "class", "數量", "RMSE_diag", "NRMSE_diag", "RMSE_off", "NRMSE_off", "Combined NRMSE"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
