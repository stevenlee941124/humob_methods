import pickle, torch, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PACKAGE_ROOT = Path("humob2026_destination_diffusion")
sys_path = str(PACKAGE_ROOT / 'src')
import sys
if sys_path not in sys.path: sys.path.insert(0, sys_path)
from multi_channel_diffusion import MultiChannelSpatialUNet, MultiChannelDDPM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# We just want to sample Z for 14 days and print the values for 39_46-39_46
blind_cond = np.zeros((14, 4), dtype=np.float32)
for j in range(14):
    dt = datetime(2024, 2, 1) + timedelta(days=j)
    blind_cond[j, 0] = dt.weekday() / 6.0
    blind_cond[j, 1] = 0.0
    blind_cond[j, 2] = (dt.month - 1) / 11.0
    blind_cond[j, 3] = (31 + j) / 365.0

model = MultiChannelSpatialUNet(in_channels=1476, latent_channels=64, cond_dim=4, time_dim=128).to(DEVICE)
ckpt = torch.load(PACKAGE_ROOT / 'data' / 'outputs' / 'ddpm_1476_checkpoint.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
ddpm = MultiChannelDDPM(T=1000, device=DEVICE)

print("Sampling Z...")
with torch.no_grad():
    z = ddpm.ddim_sample(model, (14, 1476, 70, 100), c_cond=torch.tensor(blind_cond, device=DEVICE), n_steps=50).cpu().numpy()

with open(PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl', 'rb') as f: meta = pickle.load(f)
c_idx = next(r['c_idx'] for r in meta['active_routes'] if r['pair_key'] == '39_46-39_46')
ox = next(r['ox'] for r in meta['active_routes'] if r['pair_key'] == '39_46-39_46')
oy = next(r['oy'] for r in meta['active_routes'] if r['pair_key'] == '39_46-39_46')

z_i = z[:, c_idx, ox, oy]
print("Z values for 14 days:")
print(np.round(z_i, 2))
