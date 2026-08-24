"""
===============================================================================
HuMob 2026: Step 3 Spatial - Train 2D Spatial-Temporal Grid Diffusion Model
===============================================================================
"""
import sys, time, pickle, numpy as np, torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from spatial_diffusion_model import SpatialUNet2D, SpatialDDPM

DATASET_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_diffusion_dataset.npz'
CHECKPOINT  = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_ddpm_checkpoint.pt'

BATCH_SIZE = 8
EPOCHS     = 120
LR         = 3e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 75)
print(f"[Step 3 Spatial] Training 2D Spatial-Temporal Grid Diffusion Model on {DEVICE.upper()}")
print("=" * 75)

data = np.load(str(DATASET_NPZ))
spatial_z = torch.tensor(data['spatial_z'], dtype=torch.float32) # (N, 4, 70, 100)
cal_cond  = torch.tensor(data['cal_cond'],  dtype=torch.float32) # (N, 4)

dataset = TensorDataset(spatial_z, cal_cond)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

model = SpatialUNet2D(in_ch=4, cond_dim=4, base_ch=32, time_dim=128).to(DEVICE)
ddpm = SpatialDDPM(T=1000, device=DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"2D Spatial U-Net Model Parameters: {n_params:,}")
print(f"Total training samples: {len(spatial_z)} days of (4, 70, 100) spatial tensors")

start_time = time.time()
best_loss = 1e9

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    
    for x_0, cond in loader:
        x_0 = x_0.to(DEVICE)
        cond = cond.to(DEVICE)
        B = x_0.shape[0]
        
        t = torch.randint(0, ddpm.T, (B,), device=DEVICE).long()
        x_noisy, noise = ddpm.q_sample(x_0, t)
        
        noise_pred = model(x_noisy, t, cond)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * B
        
    scheduler.step()
    avg_loss = total_loss / len(dataset)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_loss': best_loss,
            'n_params': n_params
        }, str(CHECKPOINT))
        
    if epoch % 20 == 0 or epoch == EPOCHS:
        elapsed = time.time() - start_time
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:3d}/{EPOCHS:3d}] | Loss: {avg_loss:.5f} (Best: {best_loss:.5f}) | LR: {lr_cur:.2e} | Elapsed: {elapsed:.1f}s")

print("=" * 75)
print(f"✅ Training completed! Best Loss: {best_loss:.5f}")
print(f"✅ Checkpoint saved to: {CHECKPOINT}")
print("=" * 75)
