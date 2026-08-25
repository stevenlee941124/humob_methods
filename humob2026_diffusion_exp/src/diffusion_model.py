"""
===============================================================================
HuMob 2026: 1D Conditional DDPM Model Architecture
===============================================================================
詳細數學模型與推導，請參閱：DIFFUSION_MATHEMATICAL_MODEL.md

設計：
  - 輸入: (B, L) 帶噪聲的標準化殘差序列 (L=14天)
  - 條件: (B, L, 4) 每天的日曆特徵 [DoW, is_holiday, month, year_pos]
  - 輸出: (B, L) 預測的噪聲 ε_θ

  架構: 1D Conditional U-Net
    Encoder: L=14 → (downsample) → L=7
    Bottleneck: L=7
    Decoder: L=7 → (upsample) → L=14
    條件注入: 直接 concat 到輸入 channel 軸
    時間步注入: sinusoidal embedding → 每個 ResBlock 加法注入
===============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 時間步正弦嵌入
# ─────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) LongTensor → (B, dim) float"""
        device = t.device
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -freq)   # (half,)
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)             # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)              # (B, dim)


# ─────────────────────────────────────────────────────────────────
# 1D 殘差塊 (帶時間步注入)
# ─────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, out_ch)
        groups2 = min(8, out_ch)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.act   = nn.SiLU()
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.act(self.t_proj(t_emb)).unsqueeze(-1)   # 時間步加法注入
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────
# 1D Conditional U-Net (核心模型)
# ─────────────────────────────────────────────────────────────────
class ConditionalUNet1D(nn.Module):
    """
    輸入 / 輸出均為 (B, L=14) 序列。
    條件特徵 condition: (B, L, cond_dim=4) 直接 concat 到 input channel。
    """
    def __init__(self,
                 seq_len:    int = 14,
                 cond_dim:   int = 4,
                 base_ch:    int = 64,
                 time_dim:   int = 128):
        super().__init__()
        self.seq_len  = seq_len
        self.base_ch  = base_ch

        # 時間步嵌入網路
        self.time_net = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        in_ch = 1 + cond_dim   # 5 channels (signal + 4 conditions)

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = ResBlock1D(in_ch,       base_ch,     time_dim)   # L=14
        self.enc2 = ResBlock1D(base_ch,     base_ch,     time_dim)   # L=14 ← skip
        # Downsample: L=14 → L=7
        self.down = nn.Conv1d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        self.enc3 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim)  # L=7
        self.enc4 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim)  # L=7

        # ── Bottleneck ───────────────────────────────────────────
        self.mid1 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim)  # L=7
        self.mid2 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim)  # L=7

        # ── Decoder ──────────────────────────────────────────────
        self.dec1 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim)  # L=7
        # Upsample: L=7 → L=14
        self.up   = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        # Skip concat: base_ch (up) + base_ch (enc2 skip) = base_ch*2
        self.dec2 = ResBlock1D(base_ch * 2, base_ch,     time_dim)  # L=14
        self.dec3 = ResBlock1D(base_ch,     base_ch,     time_dim)  # L=14

        # ── Output ───────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv1d(base_ch, 1, 1)

    def forward(self,
                x_noisy:   torch.Tensor,   # (B, L)
                t:         torch.Tensor,   # (B,) LongTensor
                condition: torch.Tensor,   # (B, L, 4)
               ) -> torch.Tensor:          # (B, L)

        # 拼接 signal 與條件
        x = x_noisy.unsqueeze(1)               # (B, 1, L)
        c = condition.permute(0, 2, 1)         # (B, 4, L)
        x = torch.cat([x, c], dim=1)           # (B, 5, L)

        t_emb = self.time_net(t)               # (B, time_dim)

        # Encoder
        h1 = self.enc1(x,  t_emb)             # (B, 64, 14)
        h2 = self.enc2(h1, t_emb)             # (B, 64, 14) ← skip
        h  = F.silu(self.down(h2))            # (B,128,  7)
        h  = self.enc3(h,  t_emb)             # (B,128,  7)
        h  = self.enc4(h,  t_emb)             # (B,128,  7)

        # Bottleneck
        h  = self.mid1(h, t_emb)              # (B,128,  7)
        h  = self.mid2(h, t_emb)              # (B,128,  7)

        # Decoder
        h  = self.dec1(h, t_emb)              # (B,128,  7)
        h  = F.silu(self.up(h))               # (B, 64, 14)
        h  = torch.cat([h, h2], dim=1)        # (B,128, 14)  skip
        h  = self.dec2(h, t_emb)              # (B, 64, 14)
        h  = self.dec3(h, t_emb)              # (B, 64, 14)

        # Output projection
        h  = F.silu(self.out_norm(h))
        return self.out_conv(h).squeeze(1)    # (B, L)


# ─────────────────────────────────────────────────────────────────
# DDPM 工具類：噪聲調度 + 損失 + 採樣
# ─────────────────────────────────────────────────────────────────
class DDPM:
    """
    管理 DDPM 的噪聲調度表，提供：
      - q_sample : 前向加噪 (訓練用)
      - loss     : epsilon 預測 MSE 損失
      - ddim_sample : 快速 DDIM 採樣 (推論用, 50步)
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cpu'):
        self.T = T
        betas       = torch.linspace(beta_start, beta_end, T, device=device)
        alphas      = 1.0 - betas
        alpha_bars  = torch.cumprod(alphas, dim=0)

        self.betas              = betas
        self.alphas             = alphas
        self.alpha_bars         = alpha_bars
        self.sqrt_ab            = alpha_bars.sqrt()
        self.sqrt_one_minus_ab  = (1.0 - alpha_bars).sqrt()

    def to(self, device):
        self.betas             = self.betas.to(device)
        self.alphas            = self.alphas.to(device)
        self.alpha_bars        = self.alpha_bars.to(device)
        self.sqrt_ab           = self.sqrt_ab.to(device)
        self.sqrt_one_minus_ab = self.sqrt_one_minus_ab.to(device)
        return self

    # ── 前向加噪 (訓練) ──────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None):
        """x0: (B,L), t: (B,) → x_t: (B,L), noise: (B,L)"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.sqrt_ab[t].unsqueeze(-1)
        mab = self.sqrt_one_minus_ab[t].unsqueeze(-1)
        return ab * x0 + mab * noise, noise

    # ── MSE 損失 (訓練) ──────────────────────────────────────────
    def loss(self, model: nn.Module,
             x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        noise_pred = model(x_t, t, condition)
        return F.mse_loss(noise_pred, noise)

    # ── DDIM 快速採樣 (推論) ─────────────────────────────────────
    @torch.no_grad()
    def ddim_sample(self, model: nn.Module,
                    shape: tuple, condition: torch.Tensor,
                    n_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        shape     : (B, L) — 輸出形狀
        condition : (B, L, 4)
        n_steps   : DDIM 步數 (50 步比 1000 步快 20x)
        eta       : 0.0 = deterministic DDIM; 1.0 = stochastic DDPM
        回傳 x0_pred: (B, L)
        """
        device = condition.device
        x = torch.randn(shape, device=device)   # 純隨機雜訊起點

        # 均勻取樣 n_steps 個時間步
        timesteps = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(shape[0])
            eps     = model(x, t_batch, condition)   # 預測噪聲

            ab_t  = self.alpha_bars[t_val]
            ab_t1 = self.alpha_bars[timesteps[i + 1]] if i + 1 < n_steps else torch.tensor(1.0, device=device)

            # DDIM update
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)   # 防止極端值

            sigma = eta * ((1 - ab_t1) / (1 - ab_t) * (1 - ab_t / ab_t1)).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = ab_t1.sqrt() * x0_pred + (1 - ab_t1 - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise

        return x   # (B, L) — 去噪完畢的殘差序列
