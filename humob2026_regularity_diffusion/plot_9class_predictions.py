import sys, pickle, matplotlib.pyplot as plt, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PACKAGE_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from nine_class_baseline import compute_9class_baseline

# Matplotlib dark theme styling to match the user's image
plt.style.use('dark_background')

def parse_tsv(filepath):
    data = {}
    filepath = Path(filepath)
    if not filepath.exists():
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            try:
                raw = parts[1].replace(': NA', ': None').replace(':NA', ':None')
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od is not None:
                    data[parts[0]] = od
            except:
                pass
    return data

def get_od_pred(pred_data, orig, dest):
    preds = []
    # extract for dates between 20240201 and 20240430
    eval_dates = [d for d in pred_data.keys() if '20240201' <= d <= '20240430']
    eval_dates.sort()
    for d in eval_dates:
        val = pred_data.get(d, {}).get(orig, {}).get(dest, np.nan)
        preds.append(val)
    return np.array(preds), eval_dates

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_9class_predictions.py <prediction_tsv_path>")
        sys.exit(1)
        
    pred_tsv_path = sys.argv[1]
    title_prefix = "HuMob 2026: Destination Diffusion | Top Route per Class"
    
    print("Loading data...")
    pred_data = parse_tsv(pred_tsv_path)
    od_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
    dates_pkl_path = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
    
    with open(od_pkl_path, 'rb') as f: od_ts = pickle.load(f)
    with open(dates_pkl_path, 'rb') as f: dates_str = pickle.load(f)
    
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    
    # Pre-select representative routes for each class (based on highest mean flow)
    print("Classifying routes to find representatives...")
    class_representatives = {}
    class_max_flow = {c: -1 for c in range(1, 10)}
    
    for pair, raw in od_ts.items():
        if np.all(np.isnan(raw)): continue
        
        valid = [v for v in raw if not np.isnan(v)]
        mean_v = np.mean(valid) if valid else 0
        if mean_v < 0.1: continue
        
        y_366 = np.zeros(366, dtype=np.float64)
        for oi, v in enumerate(raw):
            d_str = dates_str[oi]
            if d_str in cal_date_to_idx:
                y_366[cal_date_to_idx[d_str]] = float(v) if not np.isnan(v) else 0.0
        _, cname, cid = compute_9class_baseline(y_366, cal_dates, cal_date_to_idx)
        
        if mean_v > class_max_flow[cid]:
            # Verify if this pair is in the diffusion bounding box
            parts = pair.split('-')
            if len(parts) == 3: orig, dest = f"{parts[0]}-{parts[1]}", parts[2]
            else: orig, dest = parts[0], parts[1]
            
            try:
                dx, dy = map(int, dest.split('_'))
                if not (30 <= dx <= 70 and 35 <= dy <= 70):
                    continue # Skip routes outside the 1476 bounding box
            except:
                continue
                
            p_arr, _ = get_od_pred(pred_data, orig, dest)
            if not np.all(np.isnan(p_arr)):
                class_max_flow[cid] = mean_v
                class_representatives[cid] = (pair, cname, raw)
                
    # Create the 3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    
    # X-axis mapping
    x_idx = np.arange(366)
    x_ticks = [cal_date_to_idx[d] for d in cal_dates if d.endswith('01') and int(d[4:6]) % 2 != 0] # Nov, Jan, Mar, May, Jul, Sep
    x_labels = [datetime.strptime(d, '%Y%m%d').strftime('%m-%d') for d in cal_dates if d.endswith('01') and int(d[4:6]) % 2 != 0]
    
    blind_start = cal_date_to_idx['20240201']
    blind_end = cal_date_to_idx['20240430']
    
    for i, cid in enumerate(range(1, 10)):
        ax = axes[i // 3, i % 3]
        if cid not in class_representatives:
            ax.set_title(f"Class {cid:02d}: No Data")
            continue
            
        pair, cname, raw = class_representatives[cid]
        parts = pair.split('-')
        if len(parts) == 3: orig, dest = f"{parts[0]}-{parts[1]}", parts[2]
        else: orig, dest = parts[0], parts[1]
        
        # 1. Plot Ground Truth (from raw)
        # We need to map dates_str indices to cal_dates indices
        raw_full = np.full(366, np.nan)
        for d_str, val in zip(dates_str, raw):
            if d_str in cal_date_to_idx:
                raw_full[cal_date_to_idx[d_str]] = val
                
        ax.plot(x_idx, raw_full, color='#E6194B', linewidth=1.2, label='Ground Truth')
        
        # 2. Plot Baseline
        b_366, _, _ = compute_9class_baseline(raw_full, cal_dates, cal_date_to_idx)
        ax.plot(x_idx, b_366, color='#FFE119', linestyle='-.', linewidth=2, label='Extracted Baseline')
        
        # 3. Plot Prediction
        p_arr, eval_dates = get_od_pred(pred_data, orig, dest)
        pred_x = [cal_date_to_idx[d] for d in eval_dates]
        ax.plot(pred_x, p_arr, color='#42D4F4', linestyle='--', linewidth=1.5, label='Destination Diffusion Pred')
        
        # 4. Shade Evaluation Gap
        ax.axvspan(blind_start, blind_end, color='#FFFAC8', alpha=0.15, label='Evaluation Gap')
        
        # Formatting
        clean_name = cname.split(') ')[-1] if ')' in cname else cname
        ax.set_title(f"Class {cid:02d}: {clean_name} ({orig}_{dest})", fontsize=10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_xlim(0, 365)
        
    # Legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_img = Path(pred_tsv_path).parent / f"9class_plot_{Path(pred_tsv_path).stem}.png"
    plt.savefig(out_img, dpi=150)
    print(f"Plot saved to: {out_img}")

if __name__ == "__main__":
    main()
