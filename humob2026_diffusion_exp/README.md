# 🌌 HuMob 2026: 2D 空間條件擴散模型 (Spatial-Temporal Diffusion Experiment)

本資料夾包含了 **HuMob 2026 能登半島災後人口流動預測挑戰賽** 中，基於 **2D 空間地理條件擴散模型 (Spatial-Temporal Diffusion Model)** 的完整實驗代碼、訓練流水線與互動式視覺化儀表板。

---

## 📖 1. 原始資料 (Raw Data) 詳解

### 1.1 競賽背景與時間跨度
- **資料檔案**：`data/raw/humob2026-dataset.tsv`
- **時間維度**：2023 年 11 月 1 日 至 2024 年 10 月 31 日，共 **366 天**。
  - **重大歷史事件**：**2024 年 1 月 1 日 能登半島芮氏規模 7.6 強震**，導致石川縣能登地區人流發生斷崖式崩跌。
  - **90 天盲區 (Blind Gap)**：**2024 年 2 月 1 日 至 2024 年 4 月 30 日**（參賽者需預測這 90 天所有 OD 路線人流）。
  - **官方驗證集 (Ground Truth)**：**2024 年 4 月 1 日 至 2024 年 4 月 30 日**（共 28 天有效評測日，排除 4/8 與 4/25 異常維護日）。

### 1.2 網格地理系統 (Spatial Grid Coordinate)
- 地理區域涵蓋日本石川縣能登半島，空間劃分為 $70 \times 100$ 的二維網格矩陣：
  - 網格代號格式：`X_Y`（其中 $X \in [1, 70]$ 代表緯度方向分塊，$Y \in [1, 100]$ 代表經度方向分塊）。
  - 外部區域代號：`-1_-1`（代表進出能登半島以外的全日本外部流動）。

### 1.3 原始資料格式 (TSV Nested Dictionary Format)
原始 TSV 每行記錄一個日期的全域 OD 流動巢狀字典：
```tsv
20231101	{'41_47': {'41_47': 620.5, '21_60': 15.2, '-1_-1': 45.0}, '10_23': {'10_23': 85.0}}
```
- **停留流動 (Diagonal Flow)**：起點等於終點（如 `41_47 -> 41_47`），代表留在該網格內的人口活動。
- **跨區流動 (Off-Diagonal Flow)**：起點不等於終點（如 `41_47 -> 21_60`），代表跨網格遷移/通勤。
- **稀疏性特徵**：全域共有 **15,129 條活躍 OD 路線**，其中超過 70% 的偏遠路線為高度稀疏的偶發人流。

### 1.4 官方評測指標 (Official NRMSE)
官方採用正規化均方根誤差 (Normalized RMSE)：
\[
\text{NRMSE}_{\text{diag}} = \frac{\text{RMSE}_{\text{diag}}}{26.57}, \quad \text{NRMSE}_{\text{off}} = \frac{\text{RMSE}_{\text{off}}}{0.0176}
\]
\[
\text{Combined NRMSE} = 0.5 \times \text{NRMSE}_{\text{diag}} + 0.5 \times \text{NRMSE}_{\text{off}}
\]

---

## 🏗️ 2. 2D 空間條件擴散模型架構

### 2.1 4 通道 2D 空間張量構建 ($N_{\text{days}} \times 4 \times 70 \times 100$)
為了將 15,129 條離散的 OD 路線轉化為具備地理拓撲關聯的連續空間場，我們設計了 4 通道空間張量投影：
- **Channel 0 (Stay / Diag)**：網格內部停留人流殘差 $\Delta Y(x, x, t)$。
- **Channel 1 (Outflow)**：從該網格出發前往其他網格的跨區流出總量。
- **Channel 2 (Inflow)**：從其他網格流入該網格的跨區流入總量。
- **Channel 3 (External)**：該網格與外部世界 (`-1_-1`) 的進出流動量。

### 2.2 條件引導特徵 (Conditioning Features)
構建 4 維全域日曆條件引導向量 $c(t)$：
1. **Day of Week (星期)**：$\text{DOW} / 6.0 \in [0, 1]$
2. **Japan Holiday Flag (日本國定假日)**：$1.0$（假日）或 $0.0$（平日本）
3. **Month of Year (月份進程)**：$(\text{Month} - 1) / 11.0 \in [0, 1]$
4. **Day of Year (震後時間進程)**：$\text{DayIdx} / 365.0 \in [0, 1]$

### 2.3 核心神經網路 (Spatial U-Net 2D)
- **結構**：4 層 Encoder-Decoder U-Net 架構，包含 2D Residual Blocks、Spatial Self-Attention 機制與 Skip Connections。
- **條件注入**：使用 FiLM (Feature-wise Linear Modulation) 與 Sinusoidal Time Embedding 將時間步 $t$ 與條件向量 $c$ 注入各層卷積特徵圖。
- **擴散過程**：
  - 前向擴散：$T=1000$ 步線性 Beta 調度加噪。
  - 反向推論：採用 **DDIM (Denoising Diffusion Implicit Models)** 進行 50-Step 快速採樣。

---

## 📂 3. 檔案目錄結構與腳本說明

```
humob2026_diffusion_exp/
├── data/
│   ├── raw/humob2026-dataset.tsv          # 官方原始數據集
│   ├── processed/                         # 提取之時序與日期 pickle
│   └── outputs/                           # 空間張量、模型權重與預測 TSV
├── src/
│   ├── spatial_diffusion.py               # SpatialUNet2D 與 SpatialDDPM 模型定義
│   ├── japan_calendar.py                  # 日本國定假日字典
│   └── evaluator.py                       # 官方標準評測器
├── step1_fit_exponential.py               # 宏觀趨勢擬合實驗
├── step2_build_spatial_dataset.py         # 構建 4 通道 2D 空間張量數據集
├── step3_train_spatial_diffusion.py       # 訓練 2D 空間擴散模型 (CUDA)
├── step4_sample_spatial_diffusion.py       # DDIM 採樣推論與合成解碼
├── run_pipeline.py                        # 一鍵端到端執行流水線
└── app_dashboard.py                       # 專屬 Streamlit 視覺化儀表板 (Port 8502)
```

---

## 🚀 4. 一鍵端到端執行指南

### 步驟 1：執行完整訓練與採樣流水線
在專案根目錄下執行：
```powershell
python humob2026_diffusion_exp/run_pipeline.py
```
該腳本會自動依序執行：
1. 提取 15,129 條 OD 路線時序。
2. 投影至 $(306, 4, 70, 100)$ 空間張量。
3. 在 GPU 上訓練 120 個 Epochs 的 2D Spatial U-Net。
4. 執行 50-Step DDIM 採樣並產出預測檔案 `diffusion_spatial_predictions.tsv`。
5. 輸出四月份官方評測成績。

### 步驟 2：啟動視覺化儀表板
```powershell
streamlit run humob2026_diffusion_exp/app_dashboard.py --server.port 8502
```
瀏覽器開啟：`http://localhost:8502`

---

## 📊 5. 實驗結果與經驗總結 (Lessons Learned)

### 5.1 官方評測成績
- **停留流動 (Diag) NRMSE**：`0.21441`（實體誤差約 5.70 人）
- **跨區流動 (Off-Diag) NRMSE**：`0.72770`（實體誤差約 0.0128 人）
- **⭐ Combined NRMSE**：**`0.47106`**

### 5.2 核心優勢與局限性剖析
1. **優勢**：
   - 2D 空間卷積成功捕捉到了相鄰網格間的地理空間牽引與外溢擴散效應。
   - 解決了傳統獨立時序模型無法感知地理鄰域關聯的問題。
2. **局限性（催生下一代 Hybrid 架構的原因）**：
   - **純擴散模型缺乏確定性通勤載波**：人類通勤具備極度乾淨、嚴格的 7 天週期律動（平日尖峰、週末回落），純 Diffusion 的隨機採樣會引入不可避免的高頻微幅毛刺。
   - **空間噪聲洩漏**：對於全年 99% 為 0 人流的偏遠死寂路線，相鄰網格的空間擴散殘差會洩漏至該路線，產生微小的假陽性尖峰。
   - **結論**：純 Diffusion 適合建模空間殘差場，但不適合單獨承擔宏觀中軸與確定性通勤載波。這促成了 **`humob2026_spatial_hybrid`** 三層物理融合架構的誕生。
