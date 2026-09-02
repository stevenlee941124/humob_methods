import sys
from pathlib import Path
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_TSV = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions_v2.tsv'
OUT_SUBMISSION = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_submission_official.tsv'
VALIDATOR = PACKAGE_ROOT / 'humob2026_validator.py'

print("=" * 75)
print("正在從 dest1476_predictions_v2.tsv 導出官方提交格式...")
print("=" * 75)

lines_written = 0
with open(SRC_TSV, 'r', encoding='utf-8') as fin, open(OUT_SUBMISSION, 'w', encoding='utf-8') as fout:
    for line in fin:
        pts = line.strip().split('\t')
        if len(pts) == 2:
            d_str = pts[0]
            # 官方提交日期範圍：僅限 20240201 至 20240331
            if '20240201' <= d_str <= '20240331':
                fout.write(f"{d_str}\t{pts[1]}\n")
                lines_written += 1

print(f"已成功寫入 {lines_written} 天的預測資料 → {OUT_SUBMISSION}")

# 執行官方 validator 驗證
print("\n正在執行官方 validator 驗證...")
res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_SUBMISSION)], capture_output=True, text=True)
print("Validator 輸出:")
print(res.stdout)
if res.stderr:
    print("Validator 錯誤:", res.stderr)
if res.returncode == 0:
    print("🎉 恭喜！官方提交檔案 100% 通過所有嚴格格式驗證！")
else:
    print("❌ 驗證失敗，請檢查錯誤訊息。")
