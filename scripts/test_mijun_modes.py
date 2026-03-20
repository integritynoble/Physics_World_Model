#!/usr/bin/env python3
"""Test MiJUN with the fixed MambaLayer modes on KAIST scene01."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core', 'pwm_core', 'recon'))

import numpy as np
import h5py
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
from cassi_arch.MiJUN import MiJUN, shift_3d, shift_back_3d

model = MiJUN(stage=5, n_bands=28, step=2).to(device)
state = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(state['model'], strict=True)
model.eval()
print("Checkpoint loaded (305 keys, strict=True)")

# Load KAIST scene01
cassi_dir = os.path.join(ROOT, "datasets", "benchmark", "cassi", "standard")
h5_file = os.path.join(cassi_dir, "standard_cassi_00.h5")
with h5py.File(h5_file, 'r') as hf:
    x_true = hf['x_true'][:].astype(np.float32)
    mask_2d = hf['mask'][:].astype(np.float32)

nC, step = 28, 2
H, W = 256, 256

# ---- Test 1: MambaLayer on CORRECT 256x256 spatial input ----
print("\n=== Test 1: MambaLayer modes on 256x256 ===")
mamba_layer = None
for m in model.PP.modules():
    if m.__class__.__name__ == 'MambaLayer':
        mamba_layer = m
        break

# 256x256 spatial, C=28 (after LNLT embedding)
test_in = torch.randn(1, 28, 256, 256, device=device) * 0.1
with torch.no_grad():
    B, C, H_, W_ = test_in.shape

    # Check Mamba A_log values
    A_log = mamba_layer.mamba.A_log
    A = -torch.exp(A_log)
    print(f"Mode 0 Mamba: d_model={mamba_layer.mamba.d_model}, d_inner={mamba_layer.mamba.d_inner}")
    print(f"  A range: [{A.min().item():.4f}, {A.max().item():.4f}]")
    print(f"  A_log range: [{A_log.min().item():.4f}, {A_log.max().item():.4f}]")

    # Mode 0: (B, H*W, C) — d_model=28, L=65536
    x0 = test_in.permute(0, 2, 3, 1).reshape(B, H_ * W_, C)
    x0_normed = mamba_layer.norm(x0)
    print(f"  norm(x0) range: [{x0_normed.min().item():.4f}, {x0_normed.max().item():.4f}]")
    x0 = mamba_layer.mamba(x0_normed)
    x0 = x0.reshape(B, H_, W_, C).permute(0, 3, 1, 2)
    print(f"Mode 0 (L={H_*W_}): range [{x0.min().item():.4f}, {x0.max().item():.4f}]")

    # Mode 1: (B, C*W, H) — d_model=256, L=7168
    x1_in = test_in.permute(0, 2, 1, 3).reshape(B, H_, C * W_)
    x1_flat = x1_in.transpose(1, 2)
    x1_out = mamba_layer.mamba2(mamba_layer.norm2(x1_flat))
    x1 = x1_out.transpose(1, 2).reshape(B, H_, C, W_).permute(0, 2, 1, 3)
    print(f"Mode 1 (L={C*W_}): range [{x1.min().item():.4f}, {x1.max().item():.4f}]")

    # Mode 2: (B, C*H, W) — d_model=256, L=7168
    x2_in = test_in.permute(0, 3, 1, 2).reshape(B, W_, C * H_)
    x2_flat = x2_in.transpose(1, 2)
    x2_out = mamba_layer.mamba2(mamba_layer.norm2(x2_flat))
    x2 = x2_out.transpose(1, 2).reshape(B, W_, C, H_).permute(0, 2, 3, 1)
    print(f"Mode 2 (L={C*H_}): range [{x2.min().item():.4f}, {x2.max().item():.4f}]")

    avg = (x0 + x1 + x2) / 3.0
    print(f"Average: range [{avg.min().item():.4f}, {avg.max().item():.4f}]")

# ---- Test 2: Check where in the sequential scan Mode 0 starts diverging ----
print("\n=== Test 2: Mode 0 scan divergence analysis ===")
with torch.no_grad():
    x0_in = test_in.permute(0, 2, 3, 1).reshape(B, H_ * W_, C)
    x0_normed = mamba_layer.norm(x0_in)

    mamba = mamba_layer.mamba
    # Run Mamba step by step to find where explosion starts
    xz = mamba.in_proj(x0_normed)
    x_inner, z = xz.chunk(2, dim=-1)
    x_conv = x_inner.transpose(1, 2)
    x_conv = mamba.conv1d(x_conv)[:, :, :H_*W_]
    x_conv = torch.nn.functional.silu(x_conv).transpose(1, 2)

    x_dbl = mamba.x_proj(x_conv)
    dt, B_ssm, C_ssm = x_dbl.split(
        [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    dt = mamba.dt_proj(dt)
    dt = torch.nn.functional.softplus(dt)

    A = -torch.exp(mamba.A_log)

    print(f"  dt range: [{dt.min().item():.4f}, {dt.max().item():.4f}]")
    print(f"  dt mean: {dt.mean().item():.4f}")
    print(f"  x_conv range: [{x_conv.min().item():.4f}, {x_conv.max().item():.4f}]")

    # Check dA values
    dA_sample = torch.exp(dt[:, 0].unsqueeze(-1) * A.unsqueeze(0))
    print(f"  dA[0] range: [{dA_sample.min().item():.6f}, {dA_sample.max().item():.6f}]")

    # Run scan for first 1000 steps and check state
    batch_sz = x_conv.shape[0]
    d_inner = mamba.d_inner
    d_state = A.shape[1]
    h = torch.zeros(batch_sz, d_inner, d_state, device=device, dtype=torch.float64)
    A_64 = A.double()

    checkpoints = [100, 500, 1000, 5000, 10000, 30000, 65536]
    for t in range(min(65536, H_ * W_)):
        dt_t = dt[:, t].double()
        dA_t = torch.exp(dt_t.unsqueeze(-1) * A_64.unsqueeze(0))
        dBu = dt_t.unsqueeze(-1) * B_ssm[:, t].double().unsqueeze(1) * x_conv[:, t].double().unsqueeze(-1)
        h = dA_t * h + dBu
        y_t = (h * C_ssm[:, t].double().unsqueeze(1)).sum(-1)

        if (t + 1) in checkpoints:
            print(f"  Step {t+1}: h=[{h.min().item():.2f}, {h.max().item():.2f}], "
                  f"y=[{y_t.min().item():.2f}, {y_t.max().item():.2f}]")

# ---- Test 3: PP denoiser on clean input ----
print("\n=== Test 3: PP denoiser on clean x_true (256x256) ===")
x_true_t = torch.from_numpy(x_true.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
with torch.no_grad():
    noise_level = torch.ones(1, 1, H, W, device=device) * 0.01
    test_in_pp = torch.cat([noise_level, x_true_t], dim=1)  # (1, 29, 256, 256)
    pp_out = model.PP(test_in_pp)
    print(f"PP input range: [{test_in_pp[:, 1:].min().item():.4f}, {test_in_pp[:, 1:].max().item():.4f}]")
    print(f"PP output: {pp_out.shape}, range [{pp_out.min().item():.4f}, {pp_out.max().item():.4f}]")

    x_hat = pp_out[0].cpu().numpy().transpose(1, 2, 0)
    mse = np.mean((x_true - x_hat) ** 2)
    psnr_pp = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"PP denoiser PSNR on clean input: {psnr_pp:.2f} dB")
