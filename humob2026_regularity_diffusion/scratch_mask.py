import sys, pickle, numpy as np, math
from pathlib import Path

PACKAGE_ROOT = Path("humob2026_destination_diffusion")
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from evaluator import parse_tsv, MEAN_ACTUAL_DIAG, MEAN_ACTUAL_OFFDIAG, all_bbox_grids, n_diag_total, n_offdiag_total, EXCLUDED_DATES

gt_data = parse_tsv(PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv')
with open(PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl', 'rb') as f: od_ts = pickle.load(f)
with open(PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl', 'rb') as f: meta = pickle.load(f)
pred_data = parse_tsv(PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions.tsv')

eval_dates = [d for d in gt_data.keys() if d.startswith('202404') and d not in EXCLUDED_DATES]
eval_dates.sort()

def calc_score(p_map):
    diag_se_list, offdiag_se_list = [], []
    for d in eval_dates:
        gt_day = gt_data.get(d, {})
        pred_day = p_map.get(d, {})
        day_diag_se = sum((gt_day.get(g, {}).get(g, 0.0) - pred_day.get(g, {}).get(g, 0.0)) ** 2 for g in all_bbox_grids)
        diag_se_list.append(day_diag_se / n_diag_total)
        candidate_origins = set(gt_day.keys()) | set(pred_day.keys())
        day_offdiag_se = 0.0
        for orig in candidate_origins:
            if orig not in all_bbox_grids: continue
            gt_dests = gt_day.get(orig, {})
            pred_dests = pred_day.get(orig, {})
            for dest in (set(gt_dests.keys()) | set(pred_dests.keys())):
                if dest not in all_bbox_grids or orig == dest: continue
                diff = gt_dests.get(dest, 0.0) - pred_dests.get(dest, 0.0)
                day_offdiag_se += diff * diff
        offdiag_se_list.append(day_offdiag_se / n_offdiag_total)
    d_nrmse = math.sqrt(np.mean(diag_se_list)) / MEAN_ACTUAL_DIAG
    o_nrmse = math.sqrt(np.mean(offdiag_se_list)) / MEAN_ACTUAL_OFFDIAG
    return 0.5 * (d_nrmse + o_nrmse), d_nrmse, o_nrmse

score_raw, d_raw, o_raw = calc_score(pred_data)
print(f"1. Raw 1476 Diffusion Score: Combined={score_raw:.5f} (Diag={d_raw:.5f}, Off={o_raw:.5f})", flush=True)

for thresh in [0.05, 0.10, 0.20, 0.30]:
    masked_pred = {d: {} for d in eval_dates}
    for d in eval_dates:
        pred_day = pred_data.get(d, {})
        for orig, dests in pred_day.items():
            for dest, v in dests.items():
                pair_k = f"{orig}-{dest}"
                raw = od_ts.get(pair_k)
                if raw is None: continue
                valid_vals = [x for x in raw if not np.isnan(x)]
                if not valid_vals: continue
                p_act = sum(1 for x in valid_vals if x > 0) / len(valid_vals)
                mean_v = np.mean(valid_vals)
                if p_act >= thresh and mean_v >= thresh:
                    masked_pred[d].setdefault(orig, {})[dest] = v
    score_m, d_m, o_m = calc_score(masked_pred)
    print(f"Threshold >= {thresh:.2f}: Combined={score_m:.5f} (Diag={d_m:.5f}, Off={o_m:.5f})", flush=True)
