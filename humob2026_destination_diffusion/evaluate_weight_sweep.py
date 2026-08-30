"""
===============================================================================
HuMob 2026: Weight Sweep Evaluator (ψ7 Weekly Periodic vs Diffusion)
Evaluates linear blends: (ψ7 * w_psi + Diffusion * (1 - w_psi)) for w in [0.0, 0.1, ..., 1.0]
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

def get_z_predictions(DEVICE):
    if CACHE_Z_FILE.exists():
        print(f"[*] 載入現有 DDIM 快取: {CACHE_Z_FILE}")
        return np.load(CACHE_Z_FILE)
        
    print("[*] 正在執行 DDIM 採樣 (50 steps)...")
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
    N_BLIND = len(blind_zone)

    blind_cond = np.zeros((N_BLIND, 4), dtype=np.float32)
    for j, d_str in enumerate(blind_zone):
        dt = datetime.strptime(d_str, '%Y%m%d')
        blind_cond[j, 0] = dt.weekday() / 6.0
        blind_cond[j, 1] = 1.0 if d_str in JAPAN_HOLIDAYS else 0.0
        blind_cond[j, 2] = (dt.month - 1) / 11.0
        blind_cond[j, 3] = cal_date_to_idx[d_str] / 365.0

    model = MultiChannelSpatialUNet(in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

    cond_tensor = torch.tensor(blind_cond, device=DEVICE)
    z_pred_list = []
    batch_days = 15
    with torch.no_grad():
        for start_b in range(0, N_BLIND, batch_days):
            end_b = min(N_BLIND, start_b + batch_days)
            sub_shape = (end_b - start_b, 1476, 70, 100)
            sub_cond = cond_tensor[start_b:end_b]
            sub_z = ddpm.ddim_sample(model, sub_shape, c_cond=sub_cond, n_steps=50).cpu()
            z_pred_list.append(sub_z)

    z_pred_all = torch.cat(z_pred_list, dim=0).numpy()
    np.save(CACHE_Z_FILE, z_pred_all)
    print(f"[*] DDIM 採樣完成並快取至: {CACHE_Z_FILE}")
    return z_pred_all

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] 執行環境: {DEVICE}")

    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

    blind_zone = [d for d in cal_dates if '20240201' <= d <= '20240430']
    blind_idxs = [cal_date_to_idx[d] for d in blind_zone]
    N_BLIND = len(blind_zone)

    print("[*] 讀取資料集與元數據...")
    with open(META_PKL, 'rb')     as f: meta_1476 = pickle.load(f)
    with open(BASELINE_PKL, 'rb') as f: baselines = pickle.load(f)
    with open(OD_PKL, 'rb')       as f: od_ts     = pickle.load(f)
    with open(DATES_PKL, 'rb')    as f: dates_str = pickle.load(f)
    gt_data = parse_tsv(GT_TSV)

    eval_month_prefix = '202404'
    eval_dates = [d for d in gt_data.keys() if d.startswith(eval_month_prefix) and d not in EXCLUDED_DATES]
    eval_dates.sort()
    eval_day_indices = [blind_zone.index(d) for d in eval_dates]

    z_pred_all = get_z_predictions(DEVICE)
    train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101']

    # 1. 預先提取與分類每條路線的基礎組件 (Pre-extract route components)
    print("[*] 預先提取路線特徵與 9-Class 標籤...")
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

        # 歷史 7 天週期波動 (Raw 7-day pattern)
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

        # 擴散輸出 Z-Score 標準化
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

    # 其他未進入 1476 channel 的非活耀 baseline 路線
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

    # 建立 2.17M 網格評估結構
    # 預先計算各 Class 包含的路線數 (包含幽靈零流量 2.17M)
    all_pairs_set = [f"{orig}-{dest}" for orig in all_bbox_grids for dest in all_bbox_grids]
    pair_classes = {}
    for pair in all_pairs_set:
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

    class_counts = {cid: {"diag": 0, "off": 0} for cid in range(1, 10)}
    for pair in all_pairs_set:
        parts = pair.split('-')
        orig, dest = (f"{parts[0]}-{parts[1]}", parts[2]) if len(parts) == 3 else (parts[0], parts[1])
        cid = pair_classes[pair]
        if orig == dest:
            class_counts[cid]["diag"] += 1
        else:
            class_counts[cid]["off"] += 1

    # 2. 開始進行權重掃描 (0.0 to 1.0)
    print("\n" + "="*95)
    print(" 🚀 開始評估權重分配: (ψ7 * w_psi + Diffusion * (1 - w_psi))")
    print("="*95)

    w_psi_list = np.linspace(0.0, 1.0, 11)
    summary_results = []
    class_detailed_records = {}

    for w_psi in w_psi_list:
        w_psi = round(float(w_psi), 2)
        w_diff = round(1.0 - w_psi, 2)
        
        # 建立當前權重下的預測表
        pred_dict = {d: {} for d in eval_dates}
        
        # 活耀路線預測
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

        # Fallback 路線預測
        for r in fallback_routes_info:
            o_str, d_str = r['orig'], r['dest']
            base_90 = r['base_90']
            for d_str_cur, j in zip(eval_dates, eval_day_indices):
                val = float(base_90[j])
                if val > 0.05:
                    if o_str not in pred_dict[d_str_cur]: pred_dict[d_str_cur][o_str] = {}
                    pred_dict[d_str_cur][o_str][d_str] = val

        # 執行全網格 2.17M 官方精確評分
        class_stats = {cid: {"diag_se": 0.0, "off_se": 0.0} for cid in range(1, 10)}
        total_diag_se = 0.0
        total_off_se = 0.0

        for d in eval_dates:
            gt_day = gt_data.get(d, {})
            p_day = pred_dict.get(d, {})

            # 對角線
            for g in all_bbox_grids:
                y_true = gt_day.get(g, {}).get(g, 0.0) or 0.0
                y_pred = p_day.get(g, {}).get(g, 0.0) or 0.0
                diff = y_true - y_pred
                se = diff * diff
                pair = f"{g}-{g}"
                cid = pair_classes.get(pair, 1)
                class_stats[cid]["diag_se"] += se
                total_diag_se += se

            # 非對角線 (稀疏運算)
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
                    se = diff * diff
                    pair = f"{orig}-{dest}"
                    cid = pair_classes.get(pair, 1)
                    class_stats[cid]["off_se"] += se
                    total_off_se += se

        n_days = len(eval_dates)
        n_diag_total = len(all_bbox_grids)
        n_off_total = len(all_bbox_grids) * (len(all_bbox_grids) - 1)

        # 總體指標
        tot_rmse_diag = math.sqrt(total_diag_se / (n_diag_total * n_days))
        tot_nrmse_diag = tot_rmse_diag / MEAN_ACTUAL_DIAG
        tot_rmse_off = math.sqrt(total_off_se / (n_off_total * n_days))
        tot_nrmse_off = tot_rmse_off / MEAN_ACTUAL_OFFDIAG
        tot_combined = 0.5 * (tot_nrmse_diag + tot_nrmse_off)

        summary_results.append({
            "w_psi": w_psi,
            "w_diff": w_diff,
            "rmse_diag": tot_rmse_diag,
            "nrmse_diag": tot_nrmse_diag,
            "rmse_off": tot_rmse_off,
            "nrmse_off": tot_nrmse_off,
            "combined": tot_combined
        })

        # 分類詳細紀錄
        class_table_rows = []
        for cid in range(1, 10):
            d_count = class_counts[cid]["diag"]
            o_count = class_counts[cid]["off"]
            d_se = class_stats[cid]["diag_se"]
            o_se = class_stats[cid]["off_se"]

            c_rmse_d = math.sqrt(d_se / (d_count * n_days)) if d_count > 0 else 0.0
            c_rmse_o = math.sqrt(o_se / (o_count * n_days)) if o_count > 0 else 0.0
            c_nrmse_d = c_rmse_d / MEAN_ACTUAL_DIAG
            c_nrmse_o = c_rmse_o / MEAN_ACTUAL_OFFDIAG
            c_comb = 0.5 * (c_nrmse_d + c_nrmse_o)
            class_table_rows.append({
                "cid": cid,
                "rmse_diag": c_rmse_d,
                "nrmse_diag": c_nrmse_d,
                "rmse_off": c_rmse_o,
                "nrmse_off": c_nrmse_o,
                "combined": c_comb
            })
        class_detailed_records[w_psi] = class_table_rows

    # 3. 輸出權重掃描總覽表格 (Summary Table)
    headers = [
        "ψ7 權重 (週期)", "Diff 權重 (擴散)", 
        "RMSE_diag", "NRMSE_diag", 
        "RMSE_off", "NRMSE_off", 
        "Combined NRMSE (總分)"
    ]
    table_data = []
    best_idx = int(np.argmin([r['combined'] for r in summary_results]))

    for idx, r in enumerate(summary_results):
        tag = " 🌟 [最佳]" if idx == best_idx else ""
        table_data.append([
            f"{r['w_psi']:.1f}",
            f"{r['w_diff']:.1f}",
            f"{r['rmse_diag']:.3f} 人",
            f"{r['nrmse_diag']:.5f}",
            f"{r['rmse_off']:.3f} 人",
            f"{r['nrmse_off']:.5f}",
            f"{r['combined']:.5f}{tag}"
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    best_r = summary_results[best_idx]
    print(f"\n🏆 最佳配重組合: ψ7 = {best_r['w_psi']:.1f}  |  Diffusion = {best_r['w_diff']:.1f}")
    print(f"👉 最低 Combined NRMSE: {best_r['combined']:.5f} (RMSE_diag: {best_r['rmse_diag']:.2f}, RMSE_off: {best_r['rmse_off']:.2f})\n")

    # 4. 輸出三個代表性配重的 9-Class 比較表
    # (1) Pure Diffusion (0.0 / 1.0)
    # (2) Best Mix (w_psi / w_diff)
    # (3) Pure ψ7 (1.0 / 0.0)
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

    comp_headers = [
        "Class ID", "類別名稱", "路線數", 
        "Pure Diff (0.0/1.0)", f"最佳配重 ({best_r['w_psi']:.1f}/{best_r['w_diff']:.1f})", "Pure ψ7 (1.0/0.0)"
    ]
    comp_data = []
    for cid in range(1, 10):
        tot_c = class_counts[cid]["diag"] + class_counts[cid]["off"]
        pure_comb = class_detailed_records[0.0][cid-1]["combined"]
        best_comb = class_detailed_records[best_r['w_psi']][cid-1]["combined"]
        pure_psi_comb = class_detailed_records[1.0][cid-1]["combined"]
        comp_data.append([
            cid,
            class_names[cid],
            f"{tot_c:,} 條",
            f"{pure_comb:.4f}",
            f"{best_comb:.4f}",
            f"{pure_psi_comb:.4f}"
        ])

    print("="*95)
    print(f"📊 各類別 Combined NRMSE 深度對照表 (Pure Diff vs 最佳配重 vs Pure ψ7):")
    print("="*95)
    print(tabulate(comp_data, headers=comp_headers, tablefmt="grid"))
    print("="*95)

if __name__ == "__main__":
    main()
