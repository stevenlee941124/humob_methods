"""
===============================================================================
HuMob 2026 流水線 - 步驟 2：災後動力學 9 類別空間網格分類
===============================================================================
功能說明：
  1. 讀取步驟 1 生成的 OD 時序數據。
  2. 提取每個空間網格自身內部停留流量 (Diagonal Stay Flow, g-g)。
  3. 對比震前基準期 (< 2024/01/01) 與震後衝擊期 (2024/01/01 ~ 2024/01/31) 的流量變化。
  4. 依據災後動力學特徵，將網格分類為對應的動力學模式：
     - Persistent Zero: 持續零流量網格（非活躍/極偏遠區）
     - Persistent Low Volume: 持續極低流量網格
     - Partial Recovery: 災後人流顯著下降，處於逐步復原中
     - Temporary Increase: 災後人流異常激增（避難所/救災物資樞紐）
     - True Stable: 人流穩定受災情影響小
  5. 若已有前置分類表則直接驗證載入，否則自動執行啟發式分類並匯出 CSV。

輸入檔案：
  - data/processed/od_time_series.pkl
  - data/processed/dates.pkl
輸出檔案：
  - data/processed/grid_final_classification.csv (網格分類表)
===============================================================================
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# 設定標準輸出為 UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8')

# 定義檔案路徑
PACKAGE_ROOT = Path(__file__).resolve().parent
OD_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
OUT_CSV = PACKAGE_ROOT / 'data' / 'processed' / 'grid_final_classification.csv'

print("=" * 80)
print("🔍 [Step 2] Disaster Dynamic 9-Class Grid Classification Verification...")
print("=" * 80)

# -----------------------------------------------------------------------------
# 若已存在現成的精準分類表，直接載入並印出分佈統計
# -----------------------------------------------------------------------------
if OUT_CSV.exists():
    df_cls = pd.read_csv(OUT_CSV)
    print(f"✅ 成功載入現有網格分類表：共 {len(df_cls)} 個網格已分類。")
    print("• 類別分佈統計 (Class Distribution Summary):")
    print(df_cls['final_class'].value_counts().to_string())
else:
    # -------------------------------------------------------------------------
    # 若無現成分類表，啟動啟發式動力學分類演算法
    # -------------------------------------------------------------------------
    print(f"⚠️ 未找到預先生成的分類表，正在執行啟發式動力學分類演算法...")
    
    # 載入 OD 時序矩陣與日期
    with open(OD_PKL, 'rb') as f:
        od_ts = pickle.load(f)
    with open(DATES_PKL, 'rb') as f:
        dates = pickle.load(f)
        
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    # 定義時期：震前正常期 vs 震後 1 月衝擊期
    pre_dates = [d for d in dates if d < '20240101']
    jan_dates = [d for d in dates if '20240101' <= d <= '20240131']
    
    grid_classes = []
    
    # 收集全域所有出現過的網格 ID (Destination Grids)
    grids = set()
    for k in od_ts.keys():
        parts = k.split('-')
        dst = parts[-1]
        if dst != '-1_-1':
            grids.add(dst)
            
    # 逐一對每個網格的對角線 (Stay Flow) 進行動力學特徵識別
    for g in sorted(list(grids)):
        diag_key = f"{g}-{g}"
        ts = od_ts.get(diag_key, np.zeros(len(dates)))
        
        pre_vals = ts[[date_to_idx[d] for d in pre_dates if d in date_to_idx]]
        jan_vals = ts[[date_to_idx[d] for d in jan_dates if d in date_to_idx]]
        
        pre_m = np.nanmean(pre_vals) if len(pre_vals) > 0 else 0
        jan_m = np.nanmean(jan_vals) if len(jan_vals) > 0 else 0
        
        # 啟發式分類規則判定
        if pre_m < 0.5 and jan_m < 0.5:
            cls = "Persistent Zero"           # 持續零流量
        elif pre_m < 5.0 and jan_m < 5.0:
            cls = "Persistent Low Volume"      # 持續極低流量
        elif jan_m < pre_m * 0.6:
            cls = "Partial Recovery"          # 顯著受災衰減、緩步復原
        elif jan_m > pre_m * 1.3:
            cls = "Temporary Increase"         # 災後湧入激增
        else:
            cls = "True Stable"                # 常態穩定
            
        grid_classes.append({'grid_id': g, 'final_class': cls})
        
    df_cls = pd.DataFrame(grid_classes)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_cls.to_csv(OUT_CSV, index=False)
    print(f"✅ 已成功生成並儲存 {len(df_cls)} 個網格分類至：{OUT_CSV}")
