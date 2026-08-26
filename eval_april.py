import sys, os
from pathlib import Path
sys.path.insert(0, str(Path("humob2026_destination_diffusion/src").resolve()))
from evaluator import evaluate_predictions

def eval_april(tsv_path, name):
    print(f"\nEvaluating April for: {name}")
    in_path = Path(tsv_path)
    if not in_path.exists():
        print(f"File not found: {tsv_path}")
        return
        
    tmp_path = in_path.with_name("tmp_april_" + in_path.name)
    with open(in_path, 'r', encoding='utf-8') as fin, open(tmp_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            date_str = line.split('\t')[0]
            if date_str.startswith('202404'):
                fout.write(line)
                
    gt_path = Path("humob2026_destination_diffusion/data/raw/humob2026-dataset.tsv")
    scores = evaluate_predictions(str(gt_path), str(tmp_path))
    for k, v in scores.items():
        print(f"  • {k:<25}: {v:.5f}")
        
    tmp_path.unlink()

eval_april("humob2026_diffusion_exp/data/outputs/diffusion_predictions.tsv", "diffusion_exp (4-Channel Spatial)")
eval_april("humob2026_destination_diffusion/data/outputs/dest1476_predictions.tsv", "destination_diffusion (1476-Channel)")
