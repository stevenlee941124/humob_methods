"""
===============================================================================
HuMob 2026 Hybrid: Step 4 - Train 2D Spatial Diffusion Model on GPU
===============================================================================
"""
import sys, time, pickle, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from spatial_diffusion import SpatialUNet2D, SpatialDDPM

DATA_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_dataset.npz'
OUT_CKPT = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_hybrid_checkpoint.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EPOCHS = 120
LR = 3e-4

print("=" * 80)
print(f"[Step 4] Training 2D Spatial-Temporal Diffusion Model on {DEVICE.upper()}")
print("=" * 80)

data = np.load(DATA_NPZ)
z_spatial = data['z_spatial']          # (292, 4, 70, 100)
cond_features = data['cond_features']  # (292, 4)

class SpatialDataset(Dataset):
    def __init__(self, z_arr, c_arr):
        self.z = torch.tensor(z_arr, dtype=torch.float32)
        self.c = torch.tensor(c_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.c[idx]

dataset = SpatialDataset(z_spatial, cond_features)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

model = SpatialUNet2D(in_ch=4, cond_dim=4, base_ch=32, time_dim=128).to(DEVICE)
ddpm = SpatialDDPM(T=1000, device=DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

print(f"Dataset samples: {len(dataset)}, Batches per epoch: {len(dataloader)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("Starting training...")

start_time = time.time()
best_loss = 1e9

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x_0, c_cond in dataloader:
        x_0 = x_0.to(DEVICE)
        c_cond = c_cond.to(DEVICE)
        B = x_0.shape[0]

        t = torch.randint(0, ddpm.T, (B,), device=DEVICE).long()
        noise = torch.randn_like(x_0)
        x_t = ddpm.q_sample(x_0, t, noise)

        pred_noise = model(x_t, t, c_cond)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B

    scheduler.step()
    avg_loss = total_loss / len(dataset)

    if epoch % 20 == 0 or epoch == EPOCHS:
        elapsed = time.time() - start_time
        print(f"  Epoch [{epoch:>3}/{EPOCHS}] | MSE Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {elapsed:.1f}s")

OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    'model': model.state_dict(),
    'epoch': EPOCHS,
    'loss': avg_loss
}, OUT_CKPT)

print(f"\n✅ Training completed! Model saved to: {OUT_CKPT}")
print("=" * 80)
