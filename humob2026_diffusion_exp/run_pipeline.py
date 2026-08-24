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
    print(f"▶ {desc}")
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
    print("🚀 HuMob 2026: 2D Spatial-Temporal Diffusion 訓練與推論流水線")
    print("=" * 70)
    
    # 步驟 1: 提取 OD 時序並構建 2D 空間張量資料集
    run_step("python step2_build_spatial_dataset.py", "[Step 1] 提取 OD 時序並構建 4 通道 2D 空間流動張量資料集")
    
    # 步驟 2: 訓練 2D Spatial U-Net (DDPM)
    run_step("python step3_train_spatial_diffusion.py", "[Step 2] 訓練 2D Spatial U-Net (DDPM) 模型")
    
    # 步驟 3: 執行 DDIM 50-Step 採樣
    run_step("python step4_sample_spatial_diffusion.py", "[Step 3] 執行 2D Spatial DDIM 採樣與解碼重建")
    
    print("\n" + "=" * 70)
    print("✅ 全部完成！請在瀏覽器刷新 Dashboard 以查看最新結果！")
    print("👉 http://localhost:8502")
    print("=" * 70)

if __name__ == "__main__":
    main()
