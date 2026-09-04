"""
===============================================================================
HuMob 2026: Step 3b - Train Per-Origin Destination Map Flow Matching Model
===============================================================================
Flow Matching 訓練腳本。
Loss = E_t [ ||v_θ(x_t, t, c) - (x_0 - ε)||² ]
其中 x_t = (1-t)*ε + t*x_0，t ~ U(0,1)
===============================================================================
"""
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[警告] tqdm 未安裝，將使用簡易進度。可用 pip install tqdm 安裝。", flush=True)

sys.stdout.reconfigure(encoding='utf-8')

# 防止 Windows 在訓練期間進入睡眠
try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from origin_flow_matching import OriginDestFlowUNet, OriginFlowMatching

NPZ_PATH   = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_dataset.npz'
CHECKPOINT = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint_ep5.pt'

# ── 超參數 ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 256   # 64→256：steps/epoch 縮 4 倍，攤薄 random data access 開銷
EPOCHS     = 5     # 肘點早停 (Elbow point at epoch 5)
LR         = 3e-4
BASE_CH    = 32
TIME_DIM   = 128
COND_DIM   = 8
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 75, flush=True)
print(f"[Step 3b] Training Per-Origin Flow Matching on {DEVICE.upper()}", flush=True)
print("=" * 75, flush=True)

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("1/3 Loading dataset...", flush=True)
t_load = time.time()

# 🔑 from_numpy 共享記憶體（不複製）→ peak RAM 從 16 GB 降到 8 GB
#    torch.tensor() 會複製整個 7.94 GB，造成記憶體不足與 swap 地獄
raw            = np.load(str(NPZ_PATH))
sample_z_np    = np.array(raw['sample_z'],    dtype=np.float32, copy=False)
sample_cond_np = np.array(raw['sample_cond'], dtype=np.float32, copy=False)
del raw  # 釋放 npz 物件
print(f"  解壓完成: {time.time()-t_load:.1f}s", flush=True)

sample_z    = torch.from_numpy(sample_z_np)    # 共享記憶體，不複製
sample_cond = torch.from_numpy(sample_cond_np)
print(f"  torch tensor 建立完成: {time.time()-t_load:.1f}s", flush=True)

N_SAMPLES       = len(sample_z)
STEPS_PER_EPOCH = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
print(f"✅ 樣本數: {N_SAMPLES:,}  |  每 epoch {STEPS_PER_EPOCH:,} 步", flush=True)

dataset = TensorDataset(sample_z, sample_cond)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     drop_last=False, pin_memory=False,
                     num_workers=0)

# ── 模型 ──────────────────────────────────────────────────────────────────────
model = OriginDestFlowUNet(cond_dim=COND_DIM, base_ch=BASE_CH, time_dim=TIME_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ OriginDestFlowUNet 參數量: {n_params:,}", flush=True)
print(f"✅ Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | Steps/epoch: {STEPS_PER_EPOCH:,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── 訓練 ──────────────────────────────────────────────────────────────────────
print("\n2/3 Flow Matching Training...", flush=True)
start_time = time.time()
best_loss  = 1e9

global_step = 0
step_loss_history = []
epoch_loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t_epoch    = time.time()

    # ── 進度條：tqdm 或簡易版 ────────────────────────────────────────────────
    if HAS_TQDM:
        pbar      = tqdm(loader,
                         desc=f"Ep {epoch:3d}/{EPOCHS}",
                         unit='step',
                         dynamic_ncols=True,
                         leave=True)
        data_iter = pbar
    else:
        data_iter     = loader
        print_every   = max(1, STEPS_PER_EPOCH // 10)  # 每完成 10% 印一行

    for step_i, (x0, cond) in enumerate(data_iter):
        x0   = x0.to(DEVICE, non_blocking=True)    # (B, 1, 70, 100)
        cond = cond.to(DEVICE, non_blocking=True)   # (B, 8)
        B    = x0.shape[0]

        # Flow Matching：隨機 t ∈ [0,1]，線性插值，計算目標 vector field
        t                = torch.rand(B, device=DEVICE)
        x_t, target      = OriginFlowMatching.get_xt_and_target(x0, t)
        v_pred           = model(x_t, t, cond)
        loss             = F.mse_loss(v_pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        global_step += 1
        if global_step % 10 == 0:
            step_loss_history.append({'step': global_step, 'loss': float(loss.item())})

        # 即時顯示
        if HAS_TQDM:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'best': f'{best_loss:.4f}',
            })
        else:
            if (step_i + 1) % print_every == 0:
                pct = (step_i + 1) / STEPS_PER_EPOCH * 100
                elapsed_ep = time.time() - t_epoch
                print(f"  Ep {epoch:3d} [{pct:5.1f}%] "
                      f"step {step_i+1}/{STEPS_PER_EPOCH} | "
                      f"loss {loss.item():.4f} | "
                      f"{elapsed_ep:.0f}s elapsed", flush=True)

    # ── epoch 結束：印出摘要 + ETA ────────────────────────────────────────────
    scheduler.step()
    avg_loss  = total_loss / N_SAMPLES
    epoch_loss_history.append({'epoch': epoch, 'loss': float(avg_loss)})
    epoch_sec = time.time() - t_epoch
    remaining = epoch_sec * (EPOCHS - epoch)   # 以本 epoch 速度估算剩餘時間
    lr_cur    = optimizer.param_groups[0]['lr']

    saved_mark = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        ckpt_data = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'best_loss': best_loss,
            'n_params':  n_params,
            'cond_dim':  COND_DIM,
            'base_ch':   BASE_CH,
            'time_dim':  TIME_DIM,
        }
        torch.save(ckpt_data, str(CHECKPOINT))
        # 同步另存該 epoch 專屬權重 (讓 ep4 與 ep5 都能自由選用)
        ep_specific_ckpt = CHECKPOINT.parent / f"origin_fm_checkpoint_ep{epoch}.pt"
        torch.save(ckpt_data, str(ep_specific_ckpt))
        saved_mark = f" [saved ep{epoch}]"

    rem_h = int(remaining // 3600)
    rem_m = int((remaining % 3600) // 60)
    print(
        f"Epoch [{epoch:3d}/{EPOCHS}] "
        f"loss={avg_loss:.6f} (best={best_loss:.6f}){saved_mark} | "
        f"LR={lr_cur:.1e} | "
        f"{epoch_sec:.0f}s/ep | "
        f"ETA {rem_h}h{rem_m:02d}m",
        flush=True
    )

print("=" * 75, flush=True)
print(f"✅ Training Done! Best Loss: {best_loss:.6f}", flush=True)
print(f"✅ Checkpoint → {CHECKPOINT}", flush=True)

# ── 🌟 自動繪製並儲存 Loss vs Steps / Epochs 曲線圖 ───────────────────────
import json
import pandas as pd
import matplotlib.pyplot as plt

history_json = PACKAGE_ROOT / 'data' / 'outputs' / 'loss_history_ep5.json'
loss_png     = PACKAGE_ROOT / 'data' / 'outputs' / 'loss_step_curve_ep5.png'

with open(history_json, 'w', encoding='utf-8') as f:
    json.dump({'step_loss': step_loss_history, 'epoch_loss': epoch_loss_history}, f, indent=2)
print(f"✅ Loss 數據日誌 → {history_json}")

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
fig.patch.set_facecolor('#0f172a')

# 子圖 1: Loss vs Steps
steps = [x['step'] for x in step_loss_history]
losses = [x['loss'] for x in step_loss_history]
ax1.set_facecolor('#1e293b')
ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax1.plot(steps, losses, color='#10b981', alpha=0.35, linewidth=1.0, label='Raw Batch Loss')

if len(losses) > 10:
    ema = pd.Series(losses).ewm(span=30).mean().values
    ax1.plot(steps, ema, color='#34d399', linewidth=2.2, label='EMA Smoothed (span=30)')

ax1.set_title("Flow Matching Loss vs Steps", color='#f8fafc', fontsize=13, fontweight='bold')
ax1.set_xlabel("Training Steps", color='#cbd5e1')
ax1.set_ylabel("CNF Vector Field MSE Loss", color='#cbd5e1')
ax1.legend(facecolor='#1e293b', edgecolor='#475569')

# 子圖 2: Loss vs Epochs
eps = [x['epoch'] for x in epoch_loss_history]
ep_losses = [x['loss'] for x in epoch_loss_history]
ax2.set_facecolor('#1e293b')
ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax2.plot(eps, ep_losses, color='#f43f5e', marker='o', linewidth=2.2, markersize=6, label='Epoch Avg Loss')
ax2.set_title("Flow Matching Loss vs Epochs", color='#f8fafc', fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch", color='#cbd5e1')
ax2.set_ylabel("Average Loss", color='#cbd5e1')
ax2.legend(facecolor='#1e293b', edgecolor='#475569')

plt.tight_layout()
plt.savefig(loss_png, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✅ Loss Steps 曲線圖已繪製 → {loss_png}")
print("=" * 75, flush=True)
