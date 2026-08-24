"""
===============================================================================
HuMob 2026: Layer 3 - 2D Spatial-Temporal Diffusion Model (ResBlock2D + DDIM)
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
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SpatialAttention2D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.norm = nn.GroupNorm(8, in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)

class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch)
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb, c_cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        time_bias = self.time_mlp(t_emb)[:, :, None, None]
        cond_bias = self.cond_mlp(c_cond)[:, :, None, None]
        h = h + time_bias + cond_bias

        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class SpatialUNet2D(nn.Module):
    def __init__(self, in_ch=4, cond_dim=4, base_ch=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.enc1 = ResBlock2D(base_ch, base_ch, time_dim, cond_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.enc2 = ResBlock2D(base_ch * 2, base_ch * 2, time_dim, cond_dim)
        self.attn2 = SpatialAttention2D(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        self.mid1 = ResBlock2D(base_ch * 4, base_ch * 4, time_dim, cond_dim)
        self.mid_attn = SpatialAttention2D(base_ch * 4)
        self.mid2 = ResBlock2D(base_ch * 4, base_ch * 4, time_dim, cond_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock2D(base_ch * 4, base_ch * 2, time_dim, cond_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock2D(base_ch * 2, base_ch, time_dim, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, c_cond):
        t_emb = self.time_mlp(t)
        orig_shape = x.shape[-2:]

        h0 = self.in_conv(x)
        h1 = self.enc1(h0, t_emb, c_cond)
        d1 = self.down1(h1)

        h2 = self.enc2(d1, t_emb, c_cond)
        h2 = self.attn2(h2)
        d2 = self.down2(h2)

        m = self.mid1(d2, t_emb, c_cond)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb, c_cond)

        u2 = self.up2(m)
        if u2.shape[-2:] != h2.shape[-2:]:
            u2 = F.interpolate(u2, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat([u2, h2], dim=1)
        dec2 = self.dec2(cat2, t_emb, c_cond)

        u1 = self.up1(dec2)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat([u1, h1], dim=1)
        dec1 = self.dec1(cat1, t_emb, c_cond)

        out = F.silu(self.out_norm(dec1))
        out = self.out_conv(out)
        if out.shape[-2:] != orig_shape:
            out = F.interpolate(out, size=orig_shape, mode='bilinear', align_corners=False)
        return out

class SpatialDDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, c_cond, n_steps=50, eta=0.0):
        step_indices = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long, device=self.device)
        x_t = torch.randn(shape, device=self.device)

        for i in reversed(range(n_steps)):
            t_val = step_indices[i]
            t_batch = torch.full((shape[0],), t_val, device=self.device, dtype=torch.long)

            eps_theta = model(x_t, t_batch, c_cond)
            alpha_bar_t = self.alphas_cumprod[t_val]

            if i > 0:
                t_prev = step_indices[i - 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)

            sigma_t = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma_t ** 2) * eps_theta
            noise = torch.randn_like(x_t) if sigma_t > 0 else 0.0

            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x_t
