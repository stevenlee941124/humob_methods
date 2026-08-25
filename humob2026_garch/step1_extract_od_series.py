"""
===============================================================================
HuMob 2026 流水線 - 步驟 1：原始數據解析與 OD 時序矩陣提取
===============================================================================
功能說明：
  1. 讀取官方原始 TSV 數據集 (humob2026-dataset.tsv)。
  2. 解析每日的起訖點 (OD, Origin-Destination) 流量字典。
  3. 探索並收集全域所有出現過的活躍 OD 路線 (Active OD Pairs)。
  4. 構建每條 OD 路線在全觀測日期範圍內的連續時序矩陣 (od_time_series.pkl)，
     無觀測記錄的日期填入 NaN。
  5. 儲存提取出的 OD 時序矩陣與有效日期清單供後續模型步驟使用。

資料結構範例 (Data Structure Examples)：
  - 原始每日數據 (daily_od):
    {
      '20231101': {'31_38': {'31_38': 12.0, '32_38': 1.0, ...}, ...},
      '20231102': {'31_38': {'31_38': 15.0, '32_38': 0.0, ...}, ...},
      ...
    }
  - 輸出的連續時序矩陣 (od_time_series.pkl):
    {
      '31_38-31_38': array([12.0, 15.0, np.nan, ...]),
      '31_38-32_38': array([ 1.0,  0.0, np.nan, ...]),
      ...
    }

輸入檔案：
  - data/raw/humob2026-dataset.tsv
輸出檔案：
  - data/processed/od_time_series.pkl (OD 時序矩陣字典)
  - data/processed/dates.pkl (觀測日期列表)
===============================================================================
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# 設定標準輸出為 UTF-8 編碼，確保終端機顯示中文與 Emoji 正常
sys.stdout.reconfigure(encoding='utf-8')

# 定義專案根目錄與資料檔案路徑
PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_TSV = PACKAGE_ROOT / 'data' / 'raw' / 'humob2026-dataset.tsv'
OUT_OD_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
OUT_DATES_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'

print("=" * 80)
print("🚀 [Step 1] Parsing Raw Dataset -> Continuous OD Time Series...")
print("=" * 80)

# 檢查原始數據集是否存在
if not RAW_TSV.exists():
    print(f"❌ Error: 找不到原始數據集檔案：{RAW_TSV}")
    sys.exit(1)

dates = []
daily_od = {}

print(f"• 正在讀取原始 TSV 檔案：{RAW_TSV}...")

# 逐行讀取原始 TSV 數據並解析每一天的 OD 字典
with open(RAW_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        d_str = parts[0]
        try:
            # 將 TSV 中的 ': NA' 字串替換為 Python 的 None 以便安全 eval
            raw = parts[1].replace(': NA', ': None').replace(':NA', ':None')
            od = eval(raw, {'__builtins__': {}}, {'None': None})
            if od is not None:
                dates.append(d_str)
                daily_od[d_str] = od
        except Exception:
            pass

# 排序並去除重複的觀測日期
dates = sorted(list(set(dates)))
print(f"• 總觀測天數：{len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

# -----------------------------------------------------------------------------
# 收集全域所有出現過且流量 > 0 的活躍 OD 對 (Active OD Pairs)
# -----------------------------------------------------------------------------
all_pairs = set()
for d in dates:
    od = daily_od[d]
    for src, dsts in od.items():
        if isinstance(dsts, dict):
            for dst, val in dsts.items():
                if val is not None and val > 0:
                    od_key = f"{src}-{dst}"
                    all_pairs.add(od_key)

print(f"• 全域發現之活躍 OD 路線總數：{len(all_pairs):,} 條")

# -----------------------------------------------------------------------------
# 構建連續時間序列矩陣 (Time Series Matrix)
# 針對每條 OD 路線，建立長度等於 dates 的 float 陣列，缺測值補 NaN
# -----------------------------------------------------------------------------
od_time_series = {}
for pair in all_pairs:
    parts = pair.split('-')
    # 解析出發地 (src) 與目的地 (dst)，特別處理外部區域 '-1_-1'
    if len(parts) == 2:
        src, dst = parts[0], parts[1]
    elif len(parts) == 3 and parts[0] == '-1' and parts[1] == '-1':
        src, dst = '-1_-1', parts[2]
    else:
        continue
        
    ts = np.full(len(dates), np.nan, dtype=float)
    for i, d in enumerate(dates):
        val = daily_od[d].get(src, {}).get(dst, np.nan)
        if val is not None and not np.isnan(val):
            ts[i] = float(val)
    od_time_series[pair] = ts

# 確保輸出目錄存在
OUT_OD_PKL.parent.mkdir(parents=True, exist_ok=True)

# 序列化儲存為 Pickle 格式
with open(OUT_OD_PKL, 'wb') as f:
    pickle.dump(od_time_series, f)
with open(OUT_DATES_PKL, 'wb') as f:
    pickle.dump(dates, f)

print(f"✅ 已成功提取並儲存 {len(od_time_series):,} 條 OD 時序至：{OUT_OD_PKL}")
print(f"✅ 已儲存 {len(dates)} 天觀測日期序列至：{OUT_DATES_PKL}")
