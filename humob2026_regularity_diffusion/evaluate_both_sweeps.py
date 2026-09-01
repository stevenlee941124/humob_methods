"""
===============================================================================
Evaluate Weight Sweep for BOTH Modes:
Mode 1: 有刪除 0 流量 (1,836 條活躍路線)
Mode 2: 沒刪除 0 流量 (2,178,576 條全網格)
===============================================================================
"""
import sys, math, pickle, torch, numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM
from japan_calendar import JAPAN_HOLIDAYS
from nine_class_baseline import compute_9class_baseline

CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_1476_checkpoint.pt'
META_PKL     = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'
BASELINE_PKL = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
OD_PKL       = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
GT_TSV       = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
CACHE_Z_FILE = PACKAGE_ROOT / 'data' / 'outputs' / 'z_pred_cache.npy'

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

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

    blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
    blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
    N_BLIND = len(blind_zone)

    with open(META_PKL, 'rb')     as f: meta_1476 = pickle.load(f)
    with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
    with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
    with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)
    gt_data = parse_tsv(GT_TSV)

    eval_month_prefix = '202404'
    eval_dates = [d for d in gt_data.keys() if d.startswith(eval_month_prefix) and d not in EXCLUDED_DATES]
    eval_dates.sort()
    eval_day_indices = [blind_zone.index(d) for d in eval_dates]
    n_days = len(eval_dates)

    z_pred_all = np.load(CACHE_Z_FILE)
    train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101']

    active_routes_info = []
    active_keys = set()

    for r in meta_1476['active_routes']:
        pair_key = r['pair_key']
        active_keys.add(pair_key)
        parts = pair_key.split('-')
        o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
        d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
        is_diag = (o_str == d_str)
        
        c_idx = r['c_idx']
        ox, oy = r['ox'], r['oy']
        sig_i = r['sigma_i']
        b_366 = baselines.get(pair_key)
        if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)):
            continue
        base_90 = np.copy(b_366[blind_idxs])
        
        raw = od_ts.get(pair_key)
        valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
        mean_v = np.mean(valid_v) if valid_v else 0.0
        p_act = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
        
        cls_id = r.get('class_id', 6)
        if cls_id == 1 or (mean_v < 0.10 and p_act < 0.10):
            continue

        pre_obs = [(dates_str[oi], raw[oi]) for oi in train_days_idx if oi < len(raw) and not np.isnan(raw[oi])]
        psi_comp_raw = np.zeros(N_BLIND, dtype=np.float32)
        if len(pre_obs) >= 14 and mean_v >= 1.0:
            overall_m = np.mean([v for _, v in pre_obs])
            wd_map = {w: [] for w in range(7)}
            for d_str_k, v in pre_obs:
                wd_map[datetime.strptime(d_str_k, '%Y%m%d').weekday()].append(v)
            psi_7 = np.zeros(7, dtype=np.float32)
            for w in range(7):
                psi_7[w] = (np.mean(wd_map[w]) - overall_m) if wd_map[w] else 0.0
            
            for j, ci in enumerate(blind_idxs):
                psi_comp_raw[j] = psi_7[cal_dts[ci].weekday()]

        z_i_raw = z_pred_all[:, c_idx, ox, oy]
        z_std = np.std(z_i_raw)
        if z_std > 1e-6:
            z_i = (z_i_raw - np.mean(z_i_raw)) / z_std
        else:
            z_i = np.zeros_like(z_i_raw)
        z_i = np.clip(z_i, -2.5, 2.5)

        active_routes_info.append({
            'pair_key': pair_key,
            'orig': o_str,
            'dest': d_str,
            'is_diag': is_diag,
            'base_90': base_90,
            'psi_comp_raw': psi_comp_raw,
            'z_diff_raw': z_i * sig_i,
            'mean_v': mean_v,
            'p_act': p_act,
            'cls_id': cls_id
        })

    fallback_routes_info = []
    for pair_key, b_366 in baselines.items():
        if pair_key in active_keys: continue
        if b_366 is None or isinstance(b_366, str) or not isinstance(b_366, (list, np.ndarray)): continue
        raw = od_ts.get(pair_key)
        valid_v = [x for x in raw if not np.isnan(x)] if raw is not None else []
        mean_v = np.mean(valid_v) if valid_v else 0.0
        p_act = (sum(1 for x in valid_v if x > 0) / len(valid_v)) if valid_v else 0.0
        if mean_v < 0.25 or p_act < 0.20: continue
        parts = pair_key.split('-')
        o_str = '-1_-1' if pair_key.startswith('-1_-1-') else parts[0]
        d_str = parts[1].replace('_', '-') if pair_key.startswith('-1_-1-') else parts[1]
        base_90 = b_366[blind_idxs]
        fallback_routes_info.append({
            'orig': o_str,
            'dest': d_str,
            'is_diag': (o_str == d_str),
            'base_90': base_90
        })

    # Mode 1: 1836 filtered routes list
    filtered_pairs = []
    filtered_diag_count = 0
    filtered_off_count = 0
    for pair in od_ts.keys():
        parts = pair.split('-')
        orig, dest = (f"{parts[0]}-{parts[1]}", parts[2]) if len(parts) == 3 else (parts[0], parts[1])
        if orig in bbox_set and dest in bbox_set:
            filtered_pairs.append((orig, dest, orig == dest))
            if orig == dest:
                filtered_diag_count += 1
            else:
                filtered_off_count += 1

    # Mode 2 constants
    n_diag_full = len(all_bbox_grids) # 1476
    n_off_full = len(all_bbox_grids) * (len(all_bbox_grids) - 1) # 2177100

    w_psi_list = np.linspace(0.0, 1.0, 11)
    results_full = []
    results_filtered = []

    for w_psi in w_psi_list:
        w_psi = round(float(w_psi), 2)
        w_diff = round(1.0 - w_psi, 2)
        
        pred_dict = {d: {} for d in eval_dates}
        
        for r in active_routes_info:
            o_str, d_str = r['orig'], r['dest']
            mean_v, p_act = r['mean_v'], r['p_act']
            base_90 = r['base_90']
            psi_comp = r['psi_comp_raw'] * w_psi
            diff_comp = r['z_diff_raw'] * w_diff
            
            if mean_v >= 1.0 or (mean_v >= 0.30 and p_act >= 0.25):
                y_pred = np.maximum(0.0, base_90 + psi_comp + diff_comp)
            elif mean_v >= 0.15 and p_act >= 0.15:
                y_pred = np.maximum(0.0, base_90)
            else:
                y_pred = np.zeros(N_BLIND, dtype=np.float32)

            for d_str_cur, j in zip(eval_dates, eval_day_indices):
                val = float(y_pred[j])
                if val > 0.05:
                    if o_str not in pred_dict[d_str_cur]: pred_dict[d_str_cur][o_str] = {}
                    pred_dict[d_str_cur][o_str][d_str] = val

        for r in fallback_routes_info:
            o_str, d_str = r['orig'], r['dest']
            base_90 = r['base_90']
            for d_str_cur, j in zip(eval_dates, eval_day_indices):
                val = float(base_90[j])
                if val > 0.05:
                    if o_str not in pred_dict[d_str_cur]: pred_dict[d_str_cur][o_str] = {}
                    pred_dict[d_str_cur][o_str][d_str] = val

        # --- Evaluate Full Mode (Mode 2: 2.17M) ---
        tot_diag_se_full = 0.0
        tot_off_se_full = 0.0

        for d in eval_dates:
            gt_day = gt_data.get(d, {})
            p_day = pred_dict.get(d, {})

            for g in all_bbox_grids:
                y_true = gt_day.get(g, {}).get(g, 0.0) or 0.0
                y_pred = p_day.get(g, {}).get(g, 0.0) or 0.0
                diff = y_true - y_pred
                tot_diag_se_full += diff * diff

            active_origins = (set(gt_day.keys()) | set(p_day.keys())) & bbox_set
            for orig in active_origins:
                gt_dests = gt_day.get(orig, {})
                p_dests = p_day.get(orig, {})
                active_dests = (set(gt_dests.keys()) | set(p_dests.keys())) & bbox_set
                for dest in active_dests:
                    if orig == dest: continue
                    y_true = gt_dests.get(dest, 0.0) or 0.0
                    y_pred = p_dests.get(dest, 0.0) or 0.0
                    diff = y_true - y_pred
                    tot_off_se_full += diff * diff

        r_diag_full = math.sqrt(tot_diag_se_full / (n_diag_full * n_days))
        nr_diag_full = r_diag_full / MEAN_ACTUAL_DIAG
        r_off_full = math.sqrt(tot_off_se_full / (n_off_full * n_days))
        nr_off_full = r_off_full / MEAN_ACTUAL_OFFDIAG
        comb_full = 0.5 * (nr_diag_full + nr_off_full)

        results_full.append({
            "w_psi": w_psi, "w_diff": w_diff,
            "r_d": r_diag_full, "nr_d": nr_diag_full,
            "r_o": r_off_full, "nr_o": nr_off_full,
            "comb": comb_full
        })

        # --- Evaluate Filtered Mode (Mode 1: 1836 routes) ---
        tot_diag_se_filt = 0.0
        tot_off_se_filt = 0.0

        for d in eval_dates:
            gt_day = gt_data.get(d, {})
            p_day = pred_dict.get(d, {})

            for orig, dest, is_diag in filtered_pairs:
                y_true = gt_day.get(orig, {}).get(dest, 0.0) or 0.0
                y_pred = p_day.get(orig, {}).get(dest, 0.0) or 0.0
                diff = y_true - y_pred
                if is_diag:
                    tot_diag_se_filt += diff * diff
                else:
                    tot_off_se_filt += diff * diff

        r_diag_filt = math.sqrt(tot_diag_se_filt / (filtered_diag_count * n_days))
        nr_diag_filt = r_diag_filt / MEAN_ACTUAL_DIAG
        r_off_filt = math.sqrt(tot_off_se_filt / (filtered_off_count * n_days))
        nr_off_filt = r_off_filt / MEAN_ACTUAL_OFFDIAG
        comb_filt = 0.5 * (nr_diag_filt + nr_off_filt)

        results_filtered.append({
            "w_psi": w_psi, "w_diff": w_diff,
            "r_d": r_diag_filt, "nr_d": nr_diag_filt,
            "r_o": r_off_filt, "nr_o": nr_off_filt,
            "comb": comb_filt
        })

    headers = [
        "ψ7 權重", "Diff 權重", 
        "對角線 RMSE", "對角線 NRMSE", 
        "非對角線 RMSE", "非對角線 NRMSE", 
        "Combined NRMSE"
    ]

    print("\n" + "="*95)
    print("📋 【表格 A】沒刪除 0 流量 (2,178,576 條全網格 - 官方計分標準)")
    print("="*95)
    t_full = []
    best_f = int(np.argmin([r['comb'] for r in results_full]))
    for idx, r in enumerate(results_full):
        tag = " 🌟 [最佳]" if idx == best_f else ""
        t_full.append([
            f"{r['w_psi']:.1f}", f"{r['w_diff']:.1f}",
            f"{r['r_d']:.3f} 人", f"{r['nr_d']:.5f}",
            f"{r['r_o']:.3f} 人", f"{r['nr_o']:.5f}",
            f"{r['comb']:.5f}{tag}"
        ])
    print(tabulate(t_full, headers=headers, tablefmt="grid"))

    print("\n" + "="*95)
    print("📋 【表格 B】有刪除 0 流量 (1,836 條活躍路線 - 嚴苛真實評分)")
    print("="*95)
    t_filt = []
    best_flt = int(np.argmin([r['comb'] for r in results_filtered]))
    for idx, r in enumerate(results_filtered):
        tag = " 🌟 [最佳]" if idx == best_flt else ""
        t_filt.append([
            f"{r['w_psi']:.1f}", f"{r['w_diff']:.1f}",
            f"{r['r_d']:.3f} 人", f"{r['nr_d']:.5f}",
            f"{r['r_o']:.3f} 人", f"{r['nr_o']:.5f}",
            f"{r['comb']:.5f}{tag}"
        ])
    print(tabulate(t_filt, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
