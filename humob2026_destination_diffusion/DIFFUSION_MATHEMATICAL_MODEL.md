# 1476 通道目的地擴散模型：數學與物理模型細節

本文件詳細推導 `humob2026_destination_diffusion` 中的數學機制。由於 1476 通道的空間生成維度過大，U-Net 難以直接完美掌握極高頻的「星期波動」，因此在解碼還原（Step 4）時引入了重大的物理防呆機制（重組解耦）。

## 1. 條件擴散過程 (Conditional Diffusion Process)

模型架構為 Multi-Channel Spatial U-Net，負責去噪預測：
$$\mathbf{X}_{t-1} = \text{DDIM}(\mathbf{X}_t, t, \mathbf{c}) \quad \text{where} \quad \mathbf{X} \in \mathbb{R}^{1476 \times 70 \times 100}$$

條件向量 $\mathbf{c} \in \mathbb{R}^4$ 包含 `[Weekday, Holiday, Month, Progression]`。雖然條件有被注入，但巨大參數空間使得輸出殘差 $Z(t)$ 通常呈現較寬的變異（甚至是雜訊暴衝），且缺乏細緻的 7 天週期性。

## 2. 路線獨立 Z-Score 標準化 (Route-wise Z-Score Normalization)

從 DDIM 採樣出預測殘差 $\hat{Z}_i(t)$ 後，為防止 U-Net 產生數值爆炸或極端平坦化（被 Clip 擠壓成方波），我們對**每一條路線獨立進行 90 天時間軸上的標準化**：
$$\hat{Z}_{i,\text{norm}}(t) = \frac{\hat{Z}_i(t) - \mu_{\hat{Z}_i}}{\sigma_{\hat{Z}_i} + 1\text{e-6}}$$
$$\hat{Z}_{i,\text{clip}}(t) = \text{Clip}(\hat{Z}_{i,\text{norm}}(t), -2.5, 2.5)$$
這保證了不論原始神經網路預測的數值尺度為何，最終用於還原的隨機波動項，必然完美服從標準常態分佈 $\mathcal{N}(0, 1)$。

## 3. 歷史 7 天通勤齒波萃取 ($\psi_7$)

為了保證波形具備極致的「物理寫實感」（完美的週末低谷、平日高峰），我們從該路線 2023 年 11~12 月的無缺測訓練數據中，嚴格萃取出零均值（Zero-mean）的 7 天週期波動：
$$\psi_7[w] = \mathbb{E}[Y_{i}(t) \mid t \text{ is weekday } w] - \mathbb{E}[Y_{i}]$$
由於 $\sum \psi_7 = 0$，這項特徵疊加後**絕對不會造成基準線(Baseline)的整體上下漂移**，保證能完美「穿透」Baseline！

## 4. 人流數量分類 (Volume Classification) 與最終還原

為了避免低流量稀疏路線受到極端離群值的干擾，我們依據歷史日均流量 (Mean) 將路線分為 A~E 級，並依此決定隨機動量 ($Z$) 與週期齒波 ($\psi$) 的比例。

最終預測公式如下：
$$\hat{Y}_i(t) = \max\left(0, \ B_i(t) + \psi_7[t] \cdot G_c(t) \cdot S_\psi + \hat{Z}_{i,\text{clip}}(t) \cdot S_Z \right)$$

其中：
*   $B_i(t)$: **9-Class 物理平滑基準線**（三次 S 曲線轉移）。
*   $G_c(t)$: 災後恢復阻尼閥門（例如 Class 4 路線在 2 月份可能尚未恢復正常的 7 天規律，因此 $G_c(t)$ 從 0.5 線性遞增至 1.0）。
*   $S_\psi$: **週期特徵保留率**（高流量 $S_\psi = 1.0$，稀疏流量 $S_\psi = 0.5$）。
*   $S_Z$: **隨機變異保留率**（一般設為 $\sigma_i \times 0.7$）。這確保了最終波形除了確定的週期外，每天都會有真實幅度的高低亂數跳動！

這套「解耦重組」方法，讓 1476 通道模型既能維持宏觀的 Baseline 物理意義，又能擁有銳利、寫實、不漂移的微觀 7 天跳動波形！
