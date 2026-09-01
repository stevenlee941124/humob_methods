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
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from origin_flow_matching import OriginDestFlowUNet, OriginFlowMatching

NPZ_PATH   = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_dataset.npz'
CHECKPOINT = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_checkpoint.pt'

# ── 超參數 ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 256   # 64→256：steps/epoch 縮 4 倍，攤薄 random data access 開銷
EPOCHS     = 20    # 80→20：Flow Matching 收斂比 DDPM 快，20 epoch 足夠
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
    epoch_sec = time.time() - t_epoch
    remaining = epoch_sec * (EPOCHS - epoch)   # 以本 epoch 速度估算剩餘時間
    lr_cur    = optimizer.param_groups[0]['lr']

    saved_mark = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'best_loss': best_loss,
            'n_params':  n_params,
            'cond_dim':  COND_DIM,
            'base_ch':   BASE_CH,
            'time_dim':  TIME_DIM,
        }, str(CHECKPOINT))
        saved_mark = " [saved]"

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
print("=" * 75, flush=True)
