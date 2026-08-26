# HuMob 2026: 4 通道 2D 空間條件擴散模型 (Diffusion Exp)

這套系統 (`humob2026_diffusion_exp`) 旨在解決災後人流預測中，**起點與終點相同（即自身網格停留，Stay/Diagonal Flow）**的高密度人流預測問題。

## 核心方法與設計哲學

我們捨棄了傳統的自迴歸 (Autoregressive) 時間序列模型（如 LSTM），轉而使用 **2D 空間條件擴散模型 (Spatial Conditional Diffusion Model)**。
我們將整個日本地區切割為 `70 x 100` 的空間網格，並將每一天的人流動態視為一張「4 通道的多光譜影像」。

透過擴散模型，我們能夠讓模型學習相鄰城市之間（例如災區與周圍避難區）在空間上的吸引力與排斥力連動關係。

### 資料結構轉換 (4-Channel Spatial Tensor)
對於每個網格 $(x, y)$，我們在每一天 $t$ 提取四個物理維度，構成一個 `(4, 70, 100)` 的 2D 張量：
1. **Channel 0 (Stay Flow)**: 該網格內部的停留人數（對角線流量）。
2. **Channel 1 (Outflow)**: 從該網格流出到其他區域的總人數。
3. **Channel 2 (Inflow)**: 從其他區域流入該網格的總人數。
4. **Channel 3 (Off-grid Flow)**: 來自評測範圍外（外部世界）的流入人數。

### 處理流程 (Pipeline)

1. **Step 1 & 2: 基礎建設與 9 大類別物理 Baseline**
   - 提取歷史資料，並計算每條路線的 90 天盲區平滑基準線（Baseline）。我們將受災路線分為 9 大物理類別（例如：地震重災長期衰退區、災後臨時避難區等），並利用雙錨點（1月災前、4月災後）擬合出平滑的 **三次 S 曲線 (Cubic S-curve)**。
2. **Step 3: 訓練 2D Spatial U-Net 擴散模型**
   - 模型不直接預測人流絕對數值，而是預測**標準化殘差 (Standardized Residuals)**。
   - 訓練時引入 4D 日曆條件 `[Weekday, Holiday, Month, Progression]` 作為條件引導 (Conditioning)，讓模型學會生成具有真實「週間/週末」頻率特徵的空間噪音。
3. **Step 4: DDIM 盲區採樣與解碼**
   - 透過 DDIM 進行 90 天盲區的快速生成採樣。
   - 將生成的 2D 空間殘差張量 $Z(t)$ 取出對應的網格位置，並透過每條路線專屬的歷史變異數 $\sigma_i$ 進行反標準化，疊加回 Baseline，形成最終預測值。

> ⚠️ 數學細節與公式推導，請參見 `DIFFUSION_MATHEMATICAL_MODEL.md`。
