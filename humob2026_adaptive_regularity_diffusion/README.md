# HuMob 2026: Adaptive Regularity Diffusion (V2 Dual-Anchor)

本專案實現了針對能登半島地震時空人流重構的 **「全自適應規律度條件擴散模型（Adaptive Regularity Diffusion V2）」**。

---

## 核心數學與物理模型架構

### 1. 34 週常態期理論無偏規律度 ($R_i$)
利用全島 34 週常態觀測期（排除地震影響期），計算各週正規化後的每日變異數平均：
$$R_i = \text{clip}\left(1.0 - \text{mean\_var}, \; 0.0, \; 1.0\right)$$

### 2. 雙軌四象限連續加權 (Dual-Track Continuous Weights)
- **對角線（自身核心停留）**：
  $$W_\psi = 0.35 + 0.85 \times R_i, \quad W_{\text{diff}} = \max(0.05, 1.0 - 0.90 \times R_i)$$
- **非對角線（跨區人流移動）**：
  $$W_\psi = 0.50 + 0.60 \times R_i, \quad W_{\text{diff}} = \max(0.0, 1.0 - R_i / 0.25)$$

### 3. 每路線專屬「雙錨點春季振幅自適應門控」（Dual-Anchor Spring Expansion Gate）
為解決春天通勤振幅比冬天擴大 30%~50% 的問題，不再人為設定死板門檻，而是依據盲區兩端的真實觀測資料：
- **左錨點（1/15 ~ 1/31）**：震災初期受抑振幅 $g_{\text{start}}^{(i)} = \text{clip}\left(\frac{\sigma_{\text{Jan}}}{\sigma_{\text{pre}}}, 0.25, 1.15\right)$
- **右錨點（5/01 ~ 5/20）**：春季復甦擴張振幅 $g_{\text{end}}^{(i)} = \text{clip}\left(\frac{\sigma_{\text{May}}}{\sigma_{\text{pre}}}, 0.35, 1.80\right)$
- **90 天平滑過渡**：
  $$\text{Gate}_i(t) = g_{\text{start}}^{(i)} + \left(g_{\text{end}}^{(i)} - g_{\text{start}}^{(i)}\right) \cdot \left(\frac{t}{89}\right)^{1.5}$$

### 4. 最終預測合成公式
$$y_{\text{pred}}(t) = \max\Big(0.0, \; B_i(t) + \big(\psi_7(t) \cdot W_\psi + Z_i(t) \cdot \sigma_i \cdot \alpha \cdot W_{\text{diff}}\big) \cdot \text{Gate}_i(t)\Big)$$

---

## 執行指令

```bash
# 1. 執行採樣推論 (生成 90 天盲區預測 TSV)
python step4_sample_adaptive_v2.py

# 2. 評估 4 月官方真實評測成果
python evaluate_adaptive_v2.py

# 3. 繪製核心大動脈波形診斷圖
python plot_adaptive_v2.py
```
