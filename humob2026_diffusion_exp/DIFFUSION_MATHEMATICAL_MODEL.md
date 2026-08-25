# 📐 HuMob 2026: 2D 空間條件擴散模型 (Spatial-Temporal Diffusion Model) 完整技術與數學推導手冊

本文件完整記錄 `humob2026_diffusion_exp` 中的 2D 空間條件去噪擴散機率模型 (Spatial DDPM/DDIM) 的全流程數學推導、神經網絡架構、特徵工程與還原解碼邏輯。

---

## 🧭 1. 模型整體架構總覽 (Spatial DDPM Architecture)

為了解決能登半島地震災後 15,129 條 OD (Origin-Destination) 路線極端稀疏、尺度跨度巨大 (0.1 人至 2,000 人) 且具備強烈空間地理關聯性的挑戰，我們提出 **「4 通道 2D 空間流動張量 + 條件 Spatial U-Net 去噪擴散模型」**：

```
                    【原始 OD 稀疏時序 (15,129 條)】
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │ 【步驟 1】 4月真實錨定動態物理 Baseline B_i(t)              │
      │  • 1月份個體受災極值動態偵測 (崩跌型 vs 避難湧入型 vs 平穩)  │
      │  • 4月份真實平滑基準錨定 (緊鄰 3/31，無外推誤差/無黃金週)  │
      │  • 2~3月 (60天) 負指數復甦/消退插值 + C^1 高斯濾波         │
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │ 【步驟 2】 4 通道 2D 空間流動張量投影 (70 × 100 網格)      │
      │  • Ch 0: Retention (留存) | Ch 1: Outflow (流出)          │
      │  • Ch 2: Inflow (流入)    | Ch 3: External (外域)          │
      │  • 標準化無量綱空間殘差張量 Z_0(t) ∈ R^(4, 70, 100)        │
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │ 【步驟 3】 條件 2D Spatial U-Net (DDPM) 訓練               │
      │  • 空間填充對齊: (70, 100) -> (72, 104)                   │
      │  • 條件注入: 正弦時間步 t + 4 維日曆特徵 (FiLM Scale/Shift)│
      │  • 瓶頸層: 4-Head Multi-Head Spatial Self-Attention        │
      │  • 前向 1000 步高斯加噪，訓練預測噪聲 ε_θ                  │
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │ 【步驟 4】 DDIM 50 步加速採樣與反向解碼                    │
      │  • 盲區 50-Step 確定性軌跡去噪生成 Z_pred (90, 4, 70, 100) │
      │  • 動態截斷 clip(-2.5, 2.5) 防止單點暴走                   │
      │  • 空間反向解碼: Y_pred = max(0, B_i(t) + z_i(t) · σ_i)   │
      └────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ 步驟一：空間網格特徵與 4 通道張量投影 (2D Spatial Tensor)

### 1.1 空間幾何對映
能登半島空間劃分為 $70 \times 100$ 的 2D 網格矩陣：$X \in [1, 70]$，以及 $Y \in [1, 100]$，轉換為 0-indexed 座標 $(x-1, y-1)$。外部區域標記為 `-1_-1`。

### 1.2 四通道物理特徵定義
每日 $t$ 的空間流量被編碼為張量 $\mathbf{Y}(t) \in \mathbb{R}^{4 \times 70 \times 100}$：
- **Channel 0 (Retention - 留存人流密度)**：$src = dst$ 時，網格自身的內部停留人數。
- **Channel 1 (Outflow - 內部流出總量)**：$src \neq dst$ 時，從該起點出發前往其他網格的總人數。
- **Channel 2 (Inflow - 內部流入總量)**：$src \neq dst$ 時，從其他網格抵達該終點的總人數。
- **Channel 3 (External Exchange - 外域交換)**：該網格與外縣市 (`-1_-1`) 的進出人流總和。

### 1.3 空間殘差標準化 (Spatial Residual Normalization)
將宏觀 Baseline 投影為 $\mathbf{B}(t) \in \mathbb{R}^{4 \times 70 \times 100}$，並計算空間標準差矩陣 $\boldsymbol{\sigma}_{\text{spatial}}$：
$$\mathbf{Z}_0(t) = \frac{\mathbf{Y}(t) - \mathbf{B}(t)}{\max(\boldsymbol{\sigma}_{\text{spatial}}, 0.1)}$$
標準化後，$\mathbf{Z}_0$ 全域分佈近似標準正態分佈 $\mathcal{N}(0, \mathbf{I})$。

---

## 2️⃣ 步驟二：條件引導 2D Spatial U-Net (DDPM) 架構

### 2.1 日曆條件向量 (Calendar Conditioning)
構造 4 維日曆特徵向量 $C(t)$：
$$C(t) = \left[\frac{\text{星期}}{6.0}, \; \text{是否國定假日}, \; \frac{\text{月份}-1}{11.0}, \; \frac{\text{年度進程}}{365.0}\right] \in \mathbb{R}^4$$

### 2.2 FiLM 條件調製 (Feature-wise Linear Modulation)
在每個卷積殘差塊 (ResBlock2D) 內部，時間步 $t$ 與條件向量 $C(t)$ 被投影為 Scale 與 Shift 參數：
$$\mathbf{h}_{\text{film}} = \mathbf{h} \cdot (1 + \text{Scale}(t, C)) + \text{Shift}(t, C)$$

### 2.3 網絡拓撲結構 (Network Topology)
- **邊界填充**：$(70, 100) \to (72, 104)$，以支援 2 次完整的下採樣。
- **Encoder**：
  - Block 1: 32 Channels, ResBlock2D ($72 \times 104$) $\to$ Downsample ($36 \times 52$)
  - Block 2: 64 Channels, ResBlock2D ($36 \times 52$) $\to$ Downsample ($18 \times 26$)
  - Block 3: 128 Channels, ResBlock2D ($18 \times 26$)
- **Bottleneck (空間自注意力層 Spatial Self-Attention)**：
  - 在 $18 \times 26$ 解析度下引入 4-Head Multi-Head Self-Attention：
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  - **物理意義**：直接捕捉相距數十公里的遠距交通樞紐（如金澤市與輪島市）之間的跨區域空間連動。
- **Decoder**：
  - ConvTranspose2d 逐層上採樣 + Skip Connections 拼接對應層 Encoder 特徵。
  - 最終還原為 4 通道並 Unpad 裁剪回 $(70, 100)$。

### 2.4 DDPM 訓練目標 (Loss Function)
- 總時間步 $T=1000$，前向加噪：
  $$q(\mathbf{Z}_t \mid \mathbf{Z}_0) = \sqrt{\bar{\alpha}_t} \mathbf{Z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
- 訓練損失函數 (MSE)：
  $$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{Z}_0, \boldsymbol{\epsilon}, t, C} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{Z}_t, t, C) \right\|_2^2 \right]$$

---

## 3️⃣ 步驟三：DDIM 加速採樣與 OD 路線解碼還原

### 3.1 DDIM 50-Step 快速採樣
推論盲區 (90 天) 時，從高斯白噪聲 $\mathbf{Z}_T \sim \mathcal{N}(0, \mathbf{I})$ 開始，使用 DDIM ($\eta=0$) 以 50 步反向去噪生成全域空間場 $\hat{\mathbf{Z}} \in \mathbb{R}^{90 \times 4 \times 70 \times 100}$：
$$\mathbf{Z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{Z}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta$$
施加數值截斷：$\hat{\mathbf{Z}}_{\text{pred}} = \text{clip}(\hat{\mathbf{Z}}, -2.5, 2.5)$。

### 3.2 空間場反向解碼至 15,129 條 OD 路線
對於每一條 OD 路線 $i = (src, dst)$：
- **停留流動 (Diag)**：$z_i(t) = \hat{\mathbf{Z}}_{0, x_o, y_o}(t)$
- **跨區流動 (Off-Diag)**：$z_i(t) = \frac{1}{2} \left(\hat{\mathbf{Z}}_{1, x_o, y_o}(t) + \hat{\mathbf{Z}}_{2, x_d, y_d}(t)\right)$
- **外域交換 (Ext)**：$z_i(t) = \hat{\mathbf{Z}}_{3, x, y}(t)$

### 3.3 最終反正規化與下界截斷
$$\hat{Y}_i(t) = \max\left(0.0, \; B_i(t) + z_i(t) \cdot \sigma_i\right)$$
嚴格圍繞 4 月真實錨定 Baseline 波動，格式化輸出為競賽標準 `diffusion_predictions.tsv`。

---

## 📊 4. 官方真實驗證集成績 (Official April Benchmark)

- **停留流動 (Diag) NRMSE**：`0.17616` (實體 RMSE: 4.68 人)
- **跨區流動 (Off-Diag) NRMSE**：`0.44342` (實體 RMSE: 0.0078 人)
- **綜合評測分數 (Combined NRMSE)**：**`0.30979`**（相較於舊版 0.47106 大幅降低 **-0.161**）
