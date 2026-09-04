"""
===============================================================================
HuMob 2026: Comprehensive Evaluation for Adaptive Regularity Diffusion (V2)
===============================================================================
Evaluates predictions against official Ground Truth for April 2024.
Reports:
1. Overall Active Routes RMSE & MAE
2. Diagonal (Self-grid) vs Off-diagonal (Cross-grid) RMSE
3. Class-by-Class Performance Breakdown
4. Top 10 Largest Residual Routes
===============================================================================
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_DATA_DIR = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'
DATA_DIR = PACKAGE_ROOT / 'data'

def get_data_path(rel_path):
    local_p = DATA_DIR / rel_path
    if local_p.exists(): return local_p
    shared_p = SHARED_DATA_DIR / rel_path
    if shared_p.exists(): return shared_p
    raise FileNotFoundError(f"Cannot find {rel_path}")

RAW_TSV  = get_data_path('raw/humob2026-dataset.tsv')
META_PKL = get_data_path('outputs/meta_1476.pkl')
PRED_TSV = DATA_DIR / 'outputs' / 'dest1476_predictions_adaptive_v2.tsv'

if not PRED_TSV.exists():
    print(f"❌ 找不到預測檔案: {PRED_TSV}，請先執行 step4_sample_adaptive_v2.py")
    sys.exit(1)

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

print("1/3 載入 2024 年 4 月 Ground Truth (排除異常日)...")
gt_by_date = {}
with open(RAW_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2 and pts[0].startswith('202404') and pts[0] not in EXCLUDED_DATES:
            d_str = pts[0]
            raw_s = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw_s, {'__builtins__': {}}, {'None': None})
                if od: gt_by_date[d_str] = od
            except: pass

eval_dates = sorted(list(gt_by_date.keys()))
print(f"✅ 評測天數: {len(eval_dates)} 天")

print("2/3 載入模型預測檔案...")
pred_by_date = {}
with open(PRED_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        pts = line.strip().split('\t')
        if len(pts) >= 2 and pts[0] in gt_by_date:
            d_str = pts[0]
            raw_s = pts[1].replace(': NA', ': None').replace(':NA', ':None')
            try:
                od = eval(raw_s, {'__builtins__': {}}, {'None': None})
                if od: pred_by_date[d_str] = od
            except: pass

with open(META_PKL, 'rb') as f: meta = pickle.load(f)
cls_map = {r['pair_key']: r.get('class_id', 6) for r in meta['active_routes']}
active_routes = set(r['pair_key'] for r in meta['active_routes'])

print("3/3 執行分組指標統計...")
all_diffs = []
diag_diffs = []
offdiag_diffs = []
class_diffs = {c: [] for c in range(1, 10)}
route_stats = {}

for d in eval_dates:
    gt_day = gt_by_date.get(d, {})
    pr_day = pred_by_date.get(d, {})
    for pk in active_routes:
        o, dst = pk.split('-')
        y_true = gt_day.get(o, {}).get(dst)
        if y_true is None: continue
        y_pred = pr_day.get(o, {}).get(dst, 0.0) or 0.0
        
        diff = float(y_pred) - float(y_true)
        sq_err = diff ** 2
        all_diffs.append(sq_err)
        
        if o == dst:
            diag_diffs.append(sq_err)
        else:
            offdiag_diffs.append(sq_err)
            
        c_id = cls_map.get(pk, 6)
        if c_id in class_diffs:
            class_diffs[c_id].append(sq_err)
            
        if pk not in route_stats:
            route_stats[pk] = {'diffs': [], 'gt': [], 'is_diag': (o == dst), 'cls': c_id}
        route_stats[pk]['diffs'].append(sq_err)
        route_stats[pk]['gt'].append(float(y_true))

overall_rmse = np.sqrt(np.mean(all_diffs))
diag_rmse = np.sqrt(np.mean(diag_diffs)) if diag_diffs else 0.0
offdiag_rmse = np.sqrt(np.mean(offdiag_diffs)) if offdiag_diffs else 0.0

print("\n" + "=" * 70)
print(f"🏆 【全自適應 Regularity Diffusion (V2 Dual-Anchor)】4 月評測總成果")
print("=" * 70)
print(f"  • 全島活躍路線評測點數: {len(all_diffs):,}")
print(f"  • 全島活躍路線整體 RMSE: {overall_rmse:.4f} 人")
print(f"  • 對角線 (自身停留) RMSE: {diag_rmse:.4f} 人")
print(f"  • 非對角線 (跨區流動) RMSE: {offdiag_rmse:.4f} 人")
print("-" * 70)

cls_table = []
for c in range(1, 10):
    errs = class_diffs[c]
    rmse = np.sqrt(np.mean(errs)) if errs else 0.0
    cls_table.append([f"Class {c}", len(errs), f"{rmse:.2f} 人"])

print(tabulate(cls_table, headers=["類別 (Class)", "評測點數", "4月評測 RMSE"], tablefmt="github"))

worst_routes = []
for pk, data in route_stats.items():
    if len(data['diffs']) > 0:
        rmse = np.sqrt(np.mean(data['diffs']))
        worst_routes.append({
            'pair_key': pk,
            'is_diag': data['is_diag'],
            'cls': data['cls'],
            'rmse': rmse,
            'mean_gt': np.mean(data['gt'])
        })

worst_routes.sort(key=lambda x: x['rmse'], reverse=True)

print("\n🔥 4月評測誤差最大 Top 10 路線:")
w_table = []
for r in worst_routes[:10]:
    w_table.append([
        r['pair_key'],
        f"Class {r['cls']}",
        "對角線" if r['is_diag'] else "跨區",
        f"{r['rmse']:.2f} 人",
        f"{r['mean_gt']:.1f} 人"
    ])
print(tabulate(w_table, headers=["路線 (OD)", "類別", "類型", "RMSE", "真實均值"], tablefmt="github"))
print("=" * 70)
