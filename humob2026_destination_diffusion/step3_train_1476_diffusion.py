"""
===============================================================================
HuMob 2026: Step 3 - Masked Training (1476, 70, 100) Spatial Diffusion Model
===============================================================================
"""
import sys
import time
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM

META_PKL   = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'
CHECKPOINT = PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_1476_checkpoint.pt'

BATCH_SIZE = 4
EPOCHS     = 60
LR         = 3e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 75, flush=True)
print(f"[Step 3 Masked] Training (1476, 70, 100) Spatial Diffusion with Active Mask on {DEVICE.upper()}", flush=True)
print("=" * 75, flush=True)

with open(META_PKL, 'rb') as f:
    meta = pickle.load(f)

N_TRAIN = meta['n_train']
N_CH = meta['n_channels']
GW, GH = meta['grid_w'], meta['grid_h']

print("1/3 正在預載入 264 天靜態記憶體張量與空間遮罩...", flush=True)
t0 = time.time()
train_z = torch.zeros((N_TRAIN, N_CH, GW, GH), dtype=torch.float32)
active_mask = torch.zeros((N_CH, GW, GH), dtype=torch.float32, device=DEVICE)

for r in meta['active_routes']:
    c = r['c_idx']
    ox, oy = r['ox'], r['oy']
    train_z[:, c, ox, oy] = torch.tensor(r['z_train'], dtype=torch.float32)
    active_mask[c, ox, oy] = 1.0

train_cond = torch.tensor(meta['train_cond'], dtype=torch.float32)
print(f"✅ 靜態張量與活躍遮罩預載完成 (活躍點數: {int(active_mask.sum()):,})！", flush=True)

dataset = TensorDataset(train_z, train_cond)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=(DEVICE=='cuda'))

model = MultiChannelSpatialUNet(in_channels=N_CH, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Multi-Channel U-Net 參數量: {n_params:,}", flush=True)
print(f"✅ 訓練樣本數: {len(dataset)} 天 | 批次大小: {BATCH_SIZE} | 總輪數: {EPOCHS}", flush=True)

print("\n2/3 開始 Active-Masked 擴散模型聚焦訓練...", flush=True)
start_time = time.time()
best_loss = 1e9

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    
    for x_0, cond in loader:
        x_0 = x_0.to(DEVICE, non_blocking=True)
        cond = cond.to(DEVICE, non_blocking=True)
        B = x_0.shape[0]
        
        t = torch.randint(0, ddpm.T, (B,), device=DEVICE).long()
        x_noisy, noise = ddpm.q_sample(x_0, t)
        
        noise_pred = model(x_noisy, t, cond)
        
        # 🌟 核心創新: Active Masked MSE Loss (100% 梯度聚焦於真實人流路線)
        diff_sq = (noise_pred - noise) ** 2
        loss = (diff_sq * active_mask[None, ...]).sum() / (active_mask.sum() * B + 1e-8)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * B
        
    scheduler.step()
    avg_loss = total_loss / len(dataset)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_loss': best_loss,
            'n_params': n_params
        }, str(CHECKPOINT))
        
    if epoch % 10 == 0 or epoch == EPOCHS:
        elapsed = time.time() - start_time
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:3d}/{EPOCHS:3d}] | Masked Loss: {avg_loss:.6f} (Best: {best_loss:.6f}) | LR: {lr_cur:.2e} | Elapsed: {elapsed:.1f}s", flush=True)

print("=" * 75, flush=True)
print(f"✅ Masked Training completed! Best Loss: {best_loss:.6f}", flush=True)
print(f"✅ Checkpoint saved to: {CHECKPOINT}", flush=True)
print("=" * 75, flush=True)
