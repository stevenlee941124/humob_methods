"""
===============================================================================
HuMob 2026: One-Click Full Diffusion Pipeline Runner
===============================================================================
"""
import os
import sys
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent

def run_step(cmd, desc):
    print("\n" + "=" * 70)
    print(f"🚀 {desc}")
    print("=" * 70)
    start_t = time.time()
    res = subprocess.run(cmd, shell=True, cwd=PACKAGE_ROOT)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"❌ Error occurred in: {desc}")
        sys.exit(res.returncode)
    print(f"✅ 完成 (耗時: {elapsed:.1f} 秒)")

def main():
    print("=" * 70)
    print("🏆 HuMob 2026: 4級精細分流 + 中軸校準 Diffusion 全流程執行器")
    print("=" * 70)
    
    # 步驟 1: Step 2 構建 2D 空間張量資料集
    run_step("python step2_build_spatial_dataset.py", "[Step 2 Spatial] 構建 4 通道 2D 空間地理張量資料集")
    
    # 步驟 2: Step 4 執行 2D 空間地理 DDIM 採樣與 OD 解碼
    run_step("python step4_sample_spatial_diffusion.py", "[Step 4 Spatial] 執行 2D 空間地理 DDIM 擴散採樣與全域 OD 解碼")
    
    # 步驟 3: 自動評估 4 月成績
    print("\n" + "=" * 70)
    print("📊 [Evaluation] 評估 4 月份官方真實驗證集 NRMSE 指標")
    print("=" * 70)
    sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
    from evaluator import evaluate_predictions
    
    gt_path = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
    pred_path = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_predictions.tsv'
    
    if gt_path.exists() and pred_path.exists():
        scores = evaluate_predictions(str(gt_path), str(pred_path))
        print("\n📈 最新官方真實驗證集評測成績：")
        print(f"  • 留存人流 RMSE  (Diag)     : {scores['rmse_diag']:.4f} 人")
        print(f"  • 留存人流 NRMSE (Diag)     : {scores['nrmse_diag']:.5f}")
        print(f"  • 跨區流動 RMSE  (Off-Diag) : {scores['rmse_off']:.5f} 人")
        print(f"  • 跨區流動 NRMSE (Off-Diag) : {scores['nrmse_off']:.5f}")
        print(f"  • ⭐ 總合分數 (Combined NRMSE) : {scores['combined_nrmse']:.5f}")
    
    print("\n" + "=" * 70)
    print("🎉 一鍵執行全部完成！請在瀏覽器刷新 Dashboard 查看最新波型：")
    print("👉 http://localhost:8502")
    print("=" * 70)

if __name__ == "__main__":
    main()
