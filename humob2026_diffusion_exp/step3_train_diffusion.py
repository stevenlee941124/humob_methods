"""
===============================================================================
HuMob 2026: Step 3 - Train 1D Conditional Diffusion Model on Clean Dataset
===============================================================================
"""
import sys
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from diffusion_model import ConditionalUNet1D, DDPM

# ── 超參數 ─────────────────────────────────────────────────────
EPOCHS       = 120
BATCH_SIZE   = 256
LR           = 1e-3
T_DIFFUSION  = 1000
CLIP_Z       = 3.5     # 截斷異常極端值
CHECKPOINT   = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_checkpoint.pt'
NPZ_PATH     = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_train_dataset.npz'
HIST_CSV     = PACKAGE_ROOT / 'data' / 'outputs' / 'training_history.csv'
LOSS_PNG     = PACKAGE_ROOT / 'data' / 'outputs' / 'loss_curve.png'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device.upper()}")

# ── Dataset ────────────────────────────────────────────────────
class ResidualDataset(Dataset):
    def __init__(self, npz_path, clip_val=3.5):
        data = np.load(npz_path)
        windows    = data['windows'].astype(np.float32)
        conditions = data['conditions'].astype(np.float32)
        windows    = np.clip(windows, -clip_val, clip_val)

        self.windows    = torch.from_numpy(windows)
        self.conditions = torch.from_numpy(conditions)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.conditions[idx]

dataset    = ResidualDataset(NPZ_PATH, clip_val=CLIP_Z)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(device == 'cuda'))

print(f"Clean Dataset: {len(dataset):,} windows (Profile 1 only)")
print(f"Batches per epoch: {len(dataloader)}")

# ── 模型 & 優化器 ──────────────────────────────────────────────
model  = ConditionalUNet1D(seq_len=14, cond_dim=4, base_ch=64, time_dim=128).to(device)
ddpm   = DDPM(T=T_DIFFUSION, device=device)
optim  = torch.optim.Adam(model.parameters(), lr=LR)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR * 0.05)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

history = {'epoch': [], 'train_loss': [], 'lr': []}

def save_and_plot_history(hist, csv_path, png_path):
    df = pd.DataFrame(hist)
    df.to_csv(csv_path, index=False)
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(df['epoch'], df['train_loss'], color='#00ADB5', linewidth=2.2, label='Clean Diffusion Train Loss (MSE)')
        best_idx = df['train_loss'].idxmin()
        best_ep = df.loc[best_idx, 'epoch']
        best_l = df.loc[best_idx, 'train_loss']
        ax1.scatter([best_ep], [best_l], color='#FF2E93', s=60, zorder=5, label=f'Best Loss: {best_l:.5f} (Epoch {best_ep})')
        
        ax1.set_ylabel('MSE Loss (ε - ε_θ)²', fontsize=12, fontweight='bold')
        ax1.set_title('Clean 1D Conditional DDPM Training Loss vs. Epoch', fontsize=14, fontweight='bold', pad=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=11)
        
        ax2.plot(df['epoch'], df['lr'], color='#F39C12', linewidth=1.8, label='Learning Rate (Cosine Annealing)')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Plot error ({e})")

print(f"Training clean model for {EPOCHS} epochs on {device.upper()}...")
print()

best_loss = float('inf')

# ── 訓練迴圈 ───────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for x0, cond in dataloader:
        x0   = x0.to(device)
        cond = cond.to(device)

        optim.zero_grad()
        loss = ddpm.loss(model, x0, cond)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        epoch_loss += loss.item()
        n_batches  += 1

    lr_now = sched.get_last_lr()[0]
    sched.step()
    avg_loss = epoch_loss / n_batches

    history['epoch'].append(epoch)
    history['train_loss'].append(round(avg_loss, 5))
    history['lr'].append(lr_now)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'loss':  best_loss}, CHECKPOINT)

    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        save_and_plot_history(history, HIST_CSV, LOSS_PNG)
        print(f"Epoch {epoch:>4}/{EPOCHS}  loss={avg_loss:.5f}  best={best_loss:.5f}  lr={lr_now:.2e}")

save_and_plot_history(history, HIST_CSV, LOSS_PNG)
print()
print(f"✅  Training complete. Best loss: {best_loss:.5f}")
print(f"✅  Checkpoint saved to: {CHECKPOINT}")
print(f"✅  Loss curve image saved to: {LOSS_PNG}")
