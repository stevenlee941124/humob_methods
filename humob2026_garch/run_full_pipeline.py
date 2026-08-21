import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

steps = [
    ("Step 1: Parse Raw Data -> OD Time Series", PACKAGE_ROOT / "step1_extract_od_series.py"),
    ("Step 2: Disaster Dynamic Grid Classification", PACKAGE_ROOT / "step2_classify_grids.py"),
    ("Step 3: Run GARCH Synthesis", PACKAGE_ROOT / "step3_run_garch.py"),
    ("Step 4: Official Metric Evaluation", PACKAGE_ROOT / "step4_evaluate_predictions.py"),
]

print("=" * 80)
print("🚀 Starting End-to-End GARCH Pipeline Execution")
print("=" * 80)

for desc, script_path in steps:
    print(f"\n▶️ Running {desc}...")
    res = subprocess.run([PYTHON_EXE, str(script_path)], cwd=str(PACKAGE_ROOT))
    if res.returncode != 0:
        print(f"❌ Execution failed at {script_path.name} with code {res.returncode}")
        sys.exit(res.returncode)

print("\n" + "=" * 80)
print("🎉 [SUCCESS] Entire GARCH Pipeline Completed Flawlessly!")
print("=" * 80)
