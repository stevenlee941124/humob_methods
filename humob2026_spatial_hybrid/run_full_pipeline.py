"""
===============================================================================
HuMob 2026 Hybrid: One-Click Full Pipeline Runner
===============================================================================
"""
import sys, time, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent

steps = [
    ("Step 1: Extract OD Time Series", "step1_extract_data.py"),
    ("Step 2: Build Layer 1 Baseline & Layer 2 Gated psi", "step2_build_hybrid_base.py"),
    ("Step 3: Build 2D Spatial Residual Tensor Dataset", "step3_build_spatial_data.py"),
    ("Step 4: Train 2D Spatial-Temporal Diffusion Model", "step4_train_spatial_diff.py"),
    ("Step 5: DDIM Sampling & 3-Layer Synthesis & Evaluation", "step5_sample_and_synthesize.py")
]

print("=" * 80)
print("👑 HuMob 2026: Spatial-Temporal Hybrid Gated Diffusion Pipeline")
print("=" * 80)

total_start = time.time()

for name, script in steps:
    print(f"\n🚀 Running {name} ({script})...")
    st = time.time()
    res = subprocess.run([sys.executable, str(PACKAGE_ROOT / script)], cwd=str(PACKAGE_ROOT))
    if res.returncode != 0:
        print(f"❌ Error occurred in {script} (exit code {res.returncode})")
        sys.exit(res.returncode)
    print(f"✅ {name} finished in {time.time() - st:.1f}s")

print("\n" + "=" * 80)
print(f"🎉 Full Pipeline successfully completed in {time.time() - total_start:.1f}s!")
print("=" * 80)
