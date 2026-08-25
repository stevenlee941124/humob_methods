# 📐 HuMob 2026: Spatial-Temporal Grid Diffusion Model (2D 空間擴散模型) 數學架構說明書

本文件詳細闡述 `humob2026_diffusion_exp` 中的擴散模型演算法架構、資料張量處理機制以及還原邏輯。

---

## 🧭 模型整體架構總覽 (Spatial DDPM Architecture)

為了處理極端稀疏且具備空間拓樸關係的 14,563 條 OD (Origin-Destination) 路線，我們將離散的路線映射回 2D 空間網格 (70×100)，並使用 4 個獨立的通道 (Channels) 捕捉不同維度的流動行為。接著以去噪擴散機率模型 (Denoising Diffusion Probabilistic Model, DDPM) 學習其時空分佈。

---

## 1️⃣ 步驟一：構建空間網格特徵 (2D Spatial Tensor Construction)

### 1.1 空間座標對應
能登半島空間座標系統定義：X ∈ [1, 70], Y ∈ [1, 100]，轉換為 0-indexed 的 70 × 100 張量。

### 1.2 四通道定義 (4-Channel Representation)
每一天 $t$ 的空間特徵被編碼為張量 $\mathbf{Y}(t) \in \mathbb{R}^{4 \times 70 \times 100}$：
- **Channel 0 (Retention - 留存人流密度)**：$src = dst$ 時，網格自身的內部停留人數。
- **Channel 1 (Total Outflow - 內部流出)**：$src \neq dst$ 時，離開該網格前往內部其他網格的總人數。
- **Channel 2 (Total Inflow - 內部流入)**：$src \neq dst$ 時，從內部其他網格抵達該網格的總人數。
- **Channel 3 (External Exchange - 外域交換)**：與外部節點 (例如 `-1_-1`) 的進出流動。

### 1.3 空間殘差標準化 (Spatial Residual Normalization)
我們並不直接讓模型預測原始流量 $\mathbf{Y}(t)$，而是預測與中軸線 Baseline $\mathbf{B}(t)$ 的標準化殘差 $\mathbf{Z}(t)$。
標準化公式為：
> $\mathbf{Z}_{c,x,y}(t) = \frac{\mathbf{Y}_{c,x,y}(t) - \mathbf{B}_{c,x,y}(t)}{\max(\sigma_{c,x,y}, 0.1)}$

---

## 2️⃣ 步驟二：2D Spatial U-Net (DDPM) 擴散模型訓練

### 2.1 日曆條件注入 (Calendar Conditioning)
擴散模型接受 4 維的日曆條件 $\mathbf{C}(t) \in \mathbb{R}^4$：
1. 星期 (Day of Week) $\in [0, 1]$
2. 國定假日 (Is Holiday) $\in \{0, 1\}$
3. 月份 (Month) $\in [0, 1]$
4. 年度時間推進 (Time Progression) $t/365 \in [0, 1]$

### 2.2 DDPM 訓練目標
在前向過程 (Forward Process) 中，真實分佈 $\mathbf{Z}_0$ 逐步加入高斯噪音：
> $q(\mathbf{Z}_t \mid \mathbf{Z}_0) = \mathcal{N}(\mathbf{Z}_t; \sqrt{\bar{\alpha}_t} \mathbf{Z}_0, (1 - \bar{\alpha}_t) \mathbf{I})$

U-Net 模型 $\epsilon_\theta(\mathbf{Z}_t, t, \mathbf{C})$ 被訓練以預測添加的噪音 $\epsilon$。
損失函數為：
> $L = \mathbb{E}_{\mathbf{Z}_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(\mathbf{Z}_t, t, \mathbf{C}) \|_2^2 \right]$

---

## 3️⃣ 步驟三：DDIM 加速採樣與還原 (Decoding)

### 3.1 空間場採樣 (Spatial Field Sampling)
在盲區 90 天 (2024/02/01 ~ 2024/04/30)，模型透過 DDIM 以 50 步加速生成各天的殘差張量 $\hat{\mathbf{Z}}(t)$。
我們進一步對生成結果進行數值穩定性截斷，確保其符合標準常態分佈：
> $\hat{\mathbf{Z}}_{clip}(t) = \text{clip}(\hat{\mathbf{Z}}(t), -2.5, 2.5)$

### 3.2 OD 路線解碼與反正規化 (OD Route Decoding)
對於每一條原始 OD 路線 $i$ (起點 $src$、終點 $dst$)，將其從空間場中萃取出對應的殘差預測 $z_i(t)$：
- **若為對角線 (Retention)**： $z_i(t) = \hat{\mathbf{Z}}_{0, x_{src}, y_{src}}(t)$
- **若為內部跨區流動 (Flow)**： $z_i(t) = \frac{1}{2} (\hat{\mathbf{Z}}_{1, x_{src}, y_{src}}(t) + \hat{\mathbf{Z}}_{2, x_{dst}, y_{dst}}(t))$
- **若為外部交換 (Ext)**： $z_i(t) = \hat{\mathbf{Z}}_{3, x_{target}, y_{target}}(t)$

**標準反正規化核心原則**：
所有非死寂路線的波動，100% 圍繞真實宏觀 Baseline 中心軸波動。
> $\hat{\mathbf{Y}}_i(t) = \max\left(0.0, \mathbf{B}_i(t) + z_i(t) \cdot \sigma_i\right)$
