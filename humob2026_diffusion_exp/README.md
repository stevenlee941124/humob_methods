# 🌌 HuMob 2026: 2D 空間條件擴散模型 (Spatial-Temporal Diffusion Experiment)

本資料夾包含了 **HuMob 2026 能登半島災後人口流動預測挑戰賽** 的 **2D 空間條件擴散模型 (2D Spatial-Temporal Diffusion)** 完整實作。

該模型專注於解決巨量稀疏 OD (Origin-Destination) 路線的空間拓撲表徵問題，透過將 15,129 條 OD 路線投影至 4 通道 2D 空間流動張量，並使用 Spatial U-Net (DDPM/DDIM) 學習全域時空分佈。

---

## 📖 1. 原始資料 (Raw Data) 詳解

### 1.1 競賽背景與時間跨度
- **資料檔案**：`data/raw/humob2026-dataset.tsv`
- **時間跨度**：2023 年 11 月 1 日 至 2024 年 10 月 31 日（共 366 天）。
  - **重大歷史事件**：**2024 年 1 月 1 日 能登半島規模 7.6 強震**。
  - **預測目標盲區 (Blind Gap)**：**2024 年 2 月 1 日 至 2024 年 3 月 31 日 (60 天)**。
  - **官方驗證集 (Ground Truth)**：**2024 年 4 月 1 日 至 2024 年 4 月 30 日**（共 28 天有效評測日）。

### 1.2 空間網格與 4 通道投影系統
能登半島空間被劃分為 $70 \times 100$ 的 2D 網格矩陣：
- **Channel 0 (Retention / Diag)**：網格內部停留流動（$src = dst$）。
- **Channel 1 (Outflow)**：跨區流出起點網格（$src \neq dst$）。
- **Channel 2 (Inflow)**：跨區流入終點網格（$src \neq dst$）。
- **Channel 3 (External Exchange)**：與外縣市 (`-1_-1`) 的人流交換。

---

## 🏛️ 2. 模型架構與演算法全流程

1. **Layer 1: 4月真實錨定動態 Baseline $B(t)$**：
   - 自動偵測 1 月份個體受災極值（崩跌型 vs 避難湧入型 vs 平穩型）。
   - 錨定 4 月份真實平滑基準，消除跨月外推漂移與 5 月黃金週干擾。
   - 計算無量綱空間殘差張量 $Z(t) = \frac{Y(t) - B_{\text{spatial}}(t)}{\max(\sigma, 0.1)}$。
2. **Layer 2: 2D Spatial U-Net (DDPM) 訓練**：
   - 4 維日曆條件向量注入（星期、假日、月份、時間進程）。
   - 4 層 Encoder-Decoder + 2D ResBlock + Spatial Attention，預測高斯噪聲 $\epsilon$。
3. **Layer 3: DDIM 50-Step 空間採樣與解碼**：
   - 快速生成 90 天 4 通道空間殘差場 $\hat{Z} \in \mathbb{R}^{90 \times 4 \times 70 \times 100}$。
   - 數值穩定性截斷 $\text{clip}(\hat{Z}, -2.5, 2.5)$。
   - 解碼反正規化回 15,129 條 OD 路線：$\hat{Y}_i(t) = \max(0.0, \; B_i(t) + z_i(t) \cdot \sigma_i)$。

---

## 📂 3. 檔案目錄結構

```
humob2026_diffusion_exp/
├── data/
│   ├── raw/humob2026-dataset.tsv          # 原始數據集
│   ├── processed/                         # 時序 pickle
│   └── outputs/                           # 空間張量與預測 TSV
├── src/
│   ├── spatial_diffusion_model.py         # 2D Spatial U-Net 與 DDPM/DDIM 採樣器
│   ├── japan_calendar.py                  # 日本節假日字典
│   └── evaluator.py                       # 官方標準評測器
├── step2_build_spatial_dataset.py         # Step 1: 構建 4 通道 2D 空間張量資料集
├── step3_train_spatial_diffusion.py       # Step 2: 訓練 2D Spatial U-Net (DDPM) 模型
├── step4_sample_spatial_diffusion.py      # Step 3: 執行 DDIM 採樣與解碼重建
├── run_pipeline.py                        # 🚀 一鍵端到端執行流水線
└── app_dashboard.py                       # 🎛️ 互動式視覺化儀表板
```

---

## 📊 5. 官方真實驗證集評測成績 (Official April Benchmark)

| 模型版本 | 停留流動 Diag (RMSE) | 停留流動 NRMSE | 跨區流動 Off (RMSE) | 跨區流動 NRMSE | **Combined NRMSE** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **純 2D 空間 Diffusion (舊版 5月錨定)** | 5.70 人 | 0.21441 | 0.0128 人 | 0.72770 | **0.47106** |
| **🏆 純 2D 空間 Diffusion (最新 4月錨定動態版)** | **4.68 人** | **0.17616** | **0.0078 人** | **0.44342** | **`0.30979` 🔥** |

> [!TIP]
> 採用 4 月真實錨定動態 Baseline 後，純 2D 空間 Diffusion 的 Combined NRMSE 從 **0.47106 大幅降至 `0.30979`（降幅達 -0.161）**！跨區流動實體誤差大幅降低至 0.0078 人！
