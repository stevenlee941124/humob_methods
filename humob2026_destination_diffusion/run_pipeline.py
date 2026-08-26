"""
===============================================================================
HuMob 2026: End-to-End Pipeline for (1476, 70, 100) Destination Diffusion Model
===============================================================================
"""
import os
import sys
import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

def run_cmd(cmd, desc):
    print("\n" + "=" * 80)
    print(f"🚀 [Pipeline] {desc}")
    print(f"   Command: {cmd}")
    print("=" * 80 + "\n")
    res = subprocess.run(cmd, shell=True, cwd=str(PACKAGE_ROOT.parent))
    if res.returncode != 0:
        print(f"❌ Error during: {desc} (Exit code: {res.returncode})")
        sys.exit(res.returncode)

def main():
    print("""
    ===========================================================================
    🌌 HuMob 2026: (1476, 70, 100) Multi-Channel Destination Diffusion Pipeline
    ===========================================================================
    """)
    run_cmd(f"python humob2026_destination_diffusion/step1_extract_dest_data.py", "Step 1: Extract OD Data & Map 1,476 Target Destinations")
    run_cmd(f"python humob2026_destination_diffusion/step2_build_1476_dataset.py", "Step 2: Build 4-Month Anchored Dynamic Baseline & Tensor Meta")
    run_cmd(f"python humob2026_destination_diffusion/step3_train_1476_diffusion.py", "Step 3: Train Multi-Channel Spatial U-Net Diffusion Model")
    run_cmd(f"python humob2026_destination_diffusion/step4_sample_1476_diffusion.py", "Step 4: DDIM 50-Step Sampling & Official Metric Evaluation")
    print("\n🎉 All steps completed successfully!")

if __name__ == '__main__':
    main()
