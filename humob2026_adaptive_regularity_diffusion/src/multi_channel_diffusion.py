"""
===============================================================================
HuMob 2026: (1476, 70, 100) Multi-Channel Spatial U-Net & DDPM / DDIM Model
===============================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim=4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.gn1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch * 2) # FiLM scale and shift
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2) if cond_dim > 0 else None

        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.conv1(F.silu(self.gn1(x)))

        film = self.time_proj(t_emb)
        if c_emb is not None and self.cond_proj is not None:
            film = film + self.cond_proj(c_emb)

        scale, shift = film.chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(F.silu(self.gn2(h)))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.gn = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.gn(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (1.0 / math.sqrt(C // self.n_heads))
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class MultiChannelSpatialUNet(nn.Module):
    def __init__(self, in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        ) if cond_dim > 0 else None

        # Pad (70, 100) -> (72, 104) for clean 2x downsampling
        self.pad_h = (1, 1) # 70 -> 72
        self.pad_w = (2, 2) # 100 -> 104

        # 🌟 1x1 Conv 通道壓縮器 (1476 -> 64)
        self.compressor = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        ch1 = latent_channels     # 64
        ch2 = latent_channels * 2 # 128
        ch3 = latent_channels * 4 # 256

        # Encoder
        self.enc1 = ResBlock2D(ch1, ch1, time_dim, cond_dim=time_dim)
        self.down1 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1) # 72x104 -> 36x52

        self.enc2 = ResBlock2D(ch2, ch2, time_dim, cond_dim=time_dim)
        self.down2 = nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1) # 36x52 -> 18x26

        # Bottleneck with Spatial Self-Attention
        self.mid1 = ResBlock2D(ch3, ch3, time_dim, cond_dim=time_dim)
        self.mid_attn = SpatialSelfAttention(ch3, n_heads=4)
        self.mid2 = ResBlock2D(ch3, ch3, time_dim, cond_dim=time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(ch3, ch2, kernel_size=4, stride=2, padding=1) # 18x26 -> 36x52
        self.dec2 = ResBlock2D(ch2 + ch2, ch2, time_dim, cond_dim=time_dim)

        self.up1 = nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1) # 36x52 -> 72x104
        self.dec1 = ResBlock2D(ch1 + ch1, ch1, time_dim, cond_dim=time_dim)

        # 🌟 1x1 Conv 通道擴展器 (64 -> 1476)
        self.out_gn = nn.GroupNorm(min(8, ch1), ch1)
        self.expander = nn.Conv2d(ch1, in_channels, kernel_size=1)

    def forward(self, x, t, c=None):
        # x: (B, 1476, 70, 100)
        orig_h, orig_w = x.shape[2], x.shape[3]
        x = F.pad(x, (self.pad_w[0], self.pad_w[1], self.pad_h[0], self.pad_h[1])) # (B, 1476, 72, 104)

        t_e = self.time_emb(t)
        c_e = self.cond_emb(c) if (c is not None and self.cond_emb is not None) else None

        # 1x1 壓縮
        h0 = F.silu(self.compressor(x)) # (B, 64, 72, 104)

        # Enc
        h1 = self.enc1(h0, t_e, c_e)
        d1 = self.down1(h1)

        h2 = self.enc2(d1, t_e, c_e)
        d2 = self.down2(h2)

        # Bottleneck
        m1 = self.mid1(d2, t_e, c_e)
        ma = self.mid_attn(m1)
        m2 = self.mid2(ma, t_e, c_e)

        # Dec
        u2 = self.up2(m2)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), t_e, c_e)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), t_e, c_e)

        # 1x1 擴展還原
        out = self.expander(F.silu(self.out_gn(u1)))

        # Unpad back to (70, 100)
        out = out[:, :, self.pad_h[0]: self.pad_h[0] + orig_h, self.pad_w[0]: self.pad_w[0] + orig_w]
        return out


class MultiChannelDDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = torch.sqrt(self.alphas_bar[t])[:, None, None, None]
        sqrt_1m_ab = torch.sqrt(1.0 - self.alphas_bar[t])[:, None, None, None]
        return sqrt_ab * x_0 + sqrt_1m_ab * noise, noise

    def ddim_sample(self, model, shape, c_cond=None, n_steps=50, eta=0.0):
        step_size = self.T // n_steps
        time_steps = list(range(0, self.T, step_size))
        x = torch.randn(shape, device=self.device)

        for i in reversed(range(len(time_steps))):
            t_cur = time_steps[i]
            t_tensor = torch.full((shape[0],), t_cur, device=self.device, dtype=torch.long)
            with torch.no_grad():
                eps = model(x, t_tensor, c_cond)

            ab_cur = self.alphas_bar[t_cur]
            ab_prev = self.alphas_bar[time_steps[i - 1]] if i > 0 else torch.tensor(1.0, device=self.device)

            x_0_pred = (x - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur)
            dir_xt = torch.sqrt(1.0 - ab_prev) * eps
            x = torch.sqrt(ab_prev) * x_0_pred + dir_xt

        return x
