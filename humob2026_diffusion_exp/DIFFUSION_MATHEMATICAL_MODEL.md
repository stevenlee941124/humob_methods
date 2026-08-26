# 4 通道 2D 空間條件擴散模型：數學與物理模型細節

本文件詳細推導 `humob2026_diffusion_exp` 中的數學機制，包含標準化殘差萃取、條件引導擴散、以及最終的還原邏輯。

## 1. 殘差分離與標準化 (Residual Normalization)

為了讓 U-Net 能專注於學習空間流動關係與短期的週間起伏，我們將原本的絕對人流觀測值 $Y_i(t)$ 分解為「長期趨勢」與「短期波動」：
$$Y_i(t) = B_i(t) + \epsilon_i(t)$$
其中：
*   $B_i(t)$ 為 **9 大類別雙錨點 S 曲線基準 (Baseline)**。
*   $\epsilon_i(t)$ 為真實殘差。

接著，我們計算該路線的歷史變異數 $\sigma_i$（取 2023 年觀測數據與基準線的標準差），將其轉換為標準化殘差 $Z_i(t)$：
$$Z_i(t) = \frac{Y_i(t) - B_i(t)}{\sigma_i}$$

這些標準化殘差將被填入 4 個空間通道 (Stay, Outflow, Inflow, Off-grid) 中，形成空間張量 $\mathbf{X}_0 \in \mathbb{R}^{4 \times 70 \times 100}$。

## 2. 條件擴散過程 (Conditional Diffusion Process)

我們採用標準的 Denoising Diffusion Probabilistic Models (DDPM) 架構。
在前向加噪過程中：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

逆向去噪過程由一個 Spatial U-Net 參數化：
$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{c}) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t, \mathbf{c}), \tilde{\beta}_t\mathbf{I})$$

其中條件向量 $\mathbf{c} \in \mathbb{R}^4$ 包含：
$$\mathbf{c} = \left[ \frac{\text{Weekday}}{6}, \text{Is\_Holiday}, \frac{\text{Month}}{11}, \text{Progression} \right]$$
此條件透過 FiLM (Feature-wise Linear Modulation) 層注入到 U-Net 的每一個 ResBlock 中。由於 4 通道模型的維度較小，U-Net 能非常成功地從 $\mathbf{c}$ 中學會 $Z_i(t)$ 的「週末低谷、平日高峰」等高頻頻率。

## 3. DDIM 採樣與空間反解碼 (Decoding)

在盲區 (2024/02/01 ~ 04/30)，我們透過 DDIM 進行 50 步的加速採樣，獲得預測的標準化殘差場 $\hat{\mathbf{X}}_0$。
對於每一條對角線路線 $i = (o_x, o_y)$，我們從通道 0 取出其對應網格位置的隨機波動 $\hat{Z}_i(t)$：
$$\hat{Z}_i(t) = \hat{\mathbf{X}}_0 [0, o_x, o_y, t]$$

為了避免極端單日白噪聲暴走，我們進行了輕微的截斷保護：
$$\hat{Z}_i(t) = \text{Clip}(\hat{Z}_i(t), -2.5, 2.5)$$

最終預測值 $\hat{Y}_i(t)$ 的還原完全信任 U-Net 所生成的隨機波動，將其放大回真實的物理尺度：
$$\hat{Y}_i(t) = \max\left(0, B_i(t) + \hat{Z}_i(t) \cdot \sigma_i \right)$$

這保證了：
1. 預測波形的長期走勢完全服從物理 Baseline $B_i(t)$。
2. 短期波動的振幅與歷史觀測到的變異數 $\sigma_i$ 完全一致，不會出現不自然的扁平化現象。

## 4. 跨區流動：引力聚合近似法 (Aggregated Gravity Approximation)

4 通道模型並沒有為每一個目的地保留專屬的維度。那麼，當我們要預測一條特定的跨區路線（例如從起點 $O$ 流向終點 $D$）時，是如何無中生有的？

### 物理假設
我們採用了一個巧妙的物理假設：**「如果今天起點 $O$ 的總流出意願變高了，且終點 $D$ 的總流入吸引力也變高了，那麼 $O \to D$ 這條特定路線的人流，理當也會成正比增加。」**

### 數學合成
在解碼階段，當遇到跨區路線 (Off-Diagonal) 時，我們是這樣合成它的標準化殘差 $Z_{O \to D}(t)$ 的：

$$Z_{O \to D}(t) = 0.5 \times \Big( \underbrace{Z_{\text{outflow}}(O, t)}_{\text{起點 O 的流出波動 (Ch 1)}} + \underbrace{Z_{\text{inflow}}(D, t)}_{\text{終點 D 的流入波動 (Ch 2)}} \Big)$$

也就是說，我們把「起點的流出殘差」與「終點的流入殘差」**相加平均**，作為這條特定路線的隨機動量。接著，再把這個合成出來的 $Z_{O \to D}(t)$ 乘上這條路線專屬的歷史變異數 $\sigma_{O \to D}$，最後疊加回這條路線專屬的 Baseline：

$$Y_{O \to D}(t) = \max\left(0, \ B_{O \to D}(t) + Z_{O \to D}(t) \cdot \sigma_{O \to D} \right)$$

### 正則化效應 (Regularization Effect)
這項「取巧」的技術帶來了驚人的優勢。在實際的 NRMSE 評估中，4 通道模型的非對角線誤差 (`nrmse_off` $\approx 0.44$) 竟然比擁有 1476 個專屬目的地通道的模型 (`nrmse_off` $\approx 0.74$) 還要低！

這是因為 1476 個通道過於巨大，模型容易學習到過度發散的雜訊；相反地，4 通道模型只專注預測「總流入」與「總流出」，預測得極度平穩且精準。利用這兩個精準的「總量殘差」去合成特定路線時，反而提供了非常強健、不易暴衝的基準動量，形成了一種類似深度學習中 **正則化 (Regularization)** 的強大效應。
