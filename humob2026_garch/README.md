# 👑 HuMob 2026: 全年去污染 GARCH(1,1) 動力學流動模型

本資料夾包含了 **HuMob 2026 能登半島災後人口流動預測挑戰賽** 中，基於 **GARCH(1,1) 動態條件異方差與日曆去污染確定性載波** 的時序動力學解法。

---

## 📖 1. 原始資料 (Raw Data) 詳解

### 1.1 競賽背景與時間跨度
- **資料檔案**：`data/raw/humob2026-dataset.tsv`
- **時間跨度**：2023 年 11 月 1 日 至 2024 年 10 月 31 日，共 **366 天**。
  - **重大歷史事件**：**2024 年 1 月 1 日 能登半島芮氏規模 7.6 強震**。
  - **90 天盲區 (Blind Gap)**：**2024 年 2 月 1 日 至 2024 年 4 月 30 日**。
  - **官方驗證集 (Ground Truth)**：**2024 年 4 月 1 日 至 2024 年 4 月 30 日**（共 28 天有效評測日，排除 4/8 與 4/25 異常維護日）。

### 1.2 網格地理系統與原始格式
- 地理空間劃分為能登半島 $70 \times 100$ 的 2D 網格矩陣（`X_Y`，外部區域代號 `-1_-1`）。
- 原始 TSV 採每行一個日期之全域 OD 巢狀字典：
```tsv
20231101	{'41_47': {'41_47': 620.5, '21_60': 15.2, '-1_-1': 45.0}, '10_23': {'10_23': 85.0}}
```
- 停留流動 (Diag) 與 跨區流動 (Off-Diag) 共有 **15,129 條活躍 OD 路線**。

### 1.3 官方評測指標 (Official NRMSE)
\[
\text{NRMSE}_{\text{diag}} = \frac{\text{RMSE}_{\text{diag}}}{26.57}, \quad \text{NRMSE}_{\text{off}} = \frac{\text{RMSE}_{\text{off}}}{0.0176}
\]
\[
\text{Combined NRMSE} = 0.5 \times \text{NRMSE}_{\text{diag}} + 0.5 \times \text{NRMSE}_{\text{off}}
\]

---

## 🏛️ 2. GARCH 核心演算法架構

1. **宏觀基準中軸線 $B(t)$**：
   採用三次 Hermite 樣條插值（Cubic Hermite Spline），鎖定 2023/11 至 2024/10 全年漸進復原趨勢。
2. **日曆去污染標準化載波 $\psi(\text{DOW}_t + \phi_t)$**：
   剔除歷史國定假日干擾，提純出純淨 7 天星期載波，並引入指數衰減相位位移 $\phi(t) = 0.35 \cdot e^{-0.02 t}$。
3. **GARCH(1,1) 動態條件異方差**：
   \[
   \sigma(t)^2 = \omega + 0.25 \cdot \text{shock}(t-1)^2 + 0.65 \cdot \sigma(t-1)^2
   \]
   動態捕捉震後波動聚集，並在 90 天盲區中平滑耗散回歸常態振幅。
4. **9 大災後動力學網格分類**：
   將全域 1,476 個網格依震後受損程度與復甦形態分類為 True Stable, Rebound Trend, Late Rebound, Low Flow Flat 等。

---

## 📁 3. 檔案與目錄結構

```
humob2026_garch/
├── data/
│   ├── raw/humob2026-dataset.tsv           # 官方原始 TSV 數據集
│   ├── processed/
│   │   ├── grid_final_classification.csv   # 9 大災後動力學網格分類表
│   │   ├── od_time_series.pkl              # 提取出的全域 OD 時序矩陣
│   │   └── dates.pkl                       # 有效觀測日期序列
│   └── outputs/
│       └── wave_garch_fullyear_holiday_garch.tsv  # GARCH 最終 90 天預測輸出
├── src/
│   ├── japan_calendar.py                  # 日本國定假日與日曆狀態判定引擎
│   └── data_loader.py                     # TSV 解析與空間 BBox 邊界過濾模組
├── step1_extract_od_series.py             # 步驟 1：原始數據 ➔ 整理為 OD 時序
├── step2_classify_grids.py                # 步驟 2：災後動力學 9 類別分類
├── step3_run_garch.py                     # 步驟 3：執行 GARCH 波形合成
├── step4_evaluate_predictions.py          # 步驟 4：官方標準評估
├── run_full_pipeline.py                   # 🚀 一鍵端到端完整執行腳本 (步驟 1~4)
├── app_dashboard.py                       # 🖥️ 獨立互動 Streamlit 視覺化儀表板 (Port 8501)
├── GARCH_MATHEMATICAL_MODEL.md            # 📐 Step 3 完整數學模型架構說明書
├── DATA_PROCESSING_PIPELINE.md            # 📚 從原始數據到產出結果的詳細技術指南
└── README.md                              # 本說明文件
```

---

## 🚀 4. 一鍵端到端執行指南

### 步驟 1：執行完整流水線
```powershell
python humob2026_garch/run_full_pipeline.py
```

### 步驟 2：啟動視覺化儀表板
```powershell
streamlit run humob2026_garch/app_dashboard.py --server.port 8501
```
