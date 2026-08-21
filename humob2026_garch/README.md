## 📁 檔案與目錄結構

```
humob2026_garch/
├── data/
│   ├── raw/
│   │   └── humob2026-dataset.tsv           # 官方原始 TSV 數據集
│   ├── processed/
│   │   ├── grid_final_classification.csv   # 9 大災後動力學網格分類表
│   │   ├── od_time_series.pkl              # 提取出的全域 OD 時序矩陣
│   │   └── dates.pkl                       # 有效觀測日期序列
│   └── outputs/
│       └── wave_garch_fullyear_holiday_garch.tsv  # GARCH 最終 90 天預測輸出
│
├── src/
│   ├── __init__.py
│   ├── japan_calendar.py                  # 日本國定假日與日曆狀態判定引擎
│   └── data_loader.py                     # TSV 解析與空間 BBox 邊界過濾模組
│
├── step1_extract_od_series.py             # 步驟 1：原始數據 ➔ 整理為 OD 時序
├── step2_classify_grids.py                # 步驟 2：災後動力學 9 類別驗證/分類
├── step3_run_garch.py                     # 步驟 3：執行 GARCH 波形合成
├── step4_evaluate_predictions.py          # 步驟 4：最新官方規範 (26.57 / 0.0176) 嚴格評估
│
├── run_full_pipeline.py                   # 🚀 一鍵端到端完整執行腳本 (步驟 1~4)
├── app_dashboard.py                       # 🖥️ 獨立互動 Streamlit 視覺化儀表板
├── requirements.txt                       # Python 相依套件清單
├── DATA_PROCESSING_PIPELINE.md            # 📚 從原始數據到產出結果的詳細技術指南
└── README.md                              # 本技術說明文檔
```

---

## 快速啟動指南 (Quickstart)

### 1. 一鍵執行完整流水線（從原始數據到預測生成與評估）：
```powershell
python run_full_pipeline.py
```

### 2. 啟動互動視覺化 Dashboard：
```powershell
cd humob2026_garch
streamlit run app_dashboard.py
```

---

## 📐 核心演算法架構 (GARCH Architecture)

1. **宏觀基準中軸線 $B(t)$**：
   採用三次 Hermite 樣條插值（Cubic Hermite Spline），鎖定 2023/11 至 2024/10 全年漸進復原趨勢。
2. **日曆去污染標準化載波 $\psi(\text{DOW}_t + \phi_t)$**：
   剔除歷史國定假日干擾，提純出純淨 7 天星期載波，並引入指數衰減相位位移 $\phi_t = 0.35 \cdot e^{-0.02t}$。
3. **GARCH(1,1) 動態條件異方差**：
   ```math
   \sigma_t^2 = \omega + 0.25 \cdot \text{shock}_{t-1}^2 + 0.65 \cdot \sigma_{t-1}^2
   ```
   動態捕捉震後波動聚集，並在 90 天盲區中平滑耗散回歸常態振幅。
4. **9 大日曆狀態調節**：
   日本昭和之日（4/29）通勤抑制與假日前夕出遊增益修正。
