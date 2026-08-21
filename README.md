# 📊 HuMob Methods Collection

本儲存庫（Repository）彙整了針對 **HuMob 2026（空間移動性預測挑戰賽）** 所開發之各項核心演算法與模型架構。

---

## 📁 專案目錄結構

* **[`humob2026_garch/`](./humob2026_garch/)**：全年度去污染 GARCH 分層波形合成模型
  * 核心演算法：三次 Hermite 樣條中軸線 + 7 天日曆去污染週載波 + GARCH(1,1) 動態條件異方差
  * 包含完整的資料前處理、網格動力學分類、預測合成與 Streamlit 互動視覺化儀表板。
  * 詳細資料處理流水線請參閱 [`DATA_PROCESSING_PIPELINE.md`](./humob2026_garch/DATA_PROCESSING_PIPELINE.md)。

---

## 🚀 快速啟動

切換至欲執行的模型目錄（例如 `humob2026_garch`）：

```powershell
cd humob2026_garch
python run_full_pipeline.py
```

啟動 Streamlit 視覺化儀表板：
```powershell
cd humob2026_garch
streamlit run app_dashboard.py
```
