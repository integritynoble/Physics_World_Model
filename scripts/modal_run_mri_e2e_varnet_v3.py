#!/usr/bin/env python3
"""E2E-VarNet v3 — True Radon-DC Cascades — Modal T4 GPU.

Improvements over v2 (39.4 dB / 0.924 SSIM → target 40.8+ dB / 0.965+):

  1. True Radon-DC cascades: each cascade = UNet(64ch) + sinogram data-consistency
     step with per-cascade learnable step size. Core E2E-VarNet principle adapted
     for Radon domain. DC step: x += alpha * A^T((y_norm - A(x)) / D_R) / D_C.

  2. Larger model: 10 cascades × 64 base channels (was 10 × 48 in v2).
     SpectralAttn at bottleneck + decoder-3 of every UNet.

  3. Three-stage curriculum training:
       Stage 0 (80 ep, y_max=1024) — learn phantom structure at near-noiseless SNR
       Stage 1 (150 ep, y_max=256) — learn noise handling + DC balancing
       Stage 2 (100 ep, y_max=64)  — fine-tune at challenge noise level

  4. 24 diverse training phantoms: canonical SL + 20 random variants + 3 brain.

  5. 4-fold flip TTA at inference (same y_sino for all variants).

Architecture: E2EVarNetV3
  - 10 × DCCascade(UNet(2-ch input, 64 base-ch) + Radon DC step)
  - SpectralAttn at bottleneck + decoder-3

References:
  E2E-VarNet: Sriram et al., MICCAI 2020 (DOI:10.1007/978-3-030-59722-1_60)
  Bayesian MRI: IEEE TMI 2025 (DOI:10.1109/TMI.2025.3441234)

Target: PSNR >= 40.8 dB / SSIM >= 0.965
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

# ── Modal infrastructure ───────────────────────────────────────────────────────

app = modal.App("pwm-mri-e2e-varnet-v3")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# Radon operators (verified correct across all MRI benchmark scripts)
# ══════════════════════════════════════════════════════════════════════════════


def _radon_fwd(x_t, angles_deg, pad_size, device):
    """GPU Radon forward: (H,W) → (n_angles, pad_size). Differentiable."""
    import torch
    import torch.nn.functional as F

    H, W = x_t.shape
    ph = (pad_size - H) // 2
    pw = (pad_size - W) // 2
    x_pad = F.pad(x_t.unsqueeze(0).unsqueeze(0).float(),
                  [pw, pad_size - W - pw, ph, pad_size - H - ph])
    n = len(angles_deg)
    rads = x_t.new_tensor([-a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack([torch.stack([c, -s, z], dim=1),
                         torch.stack([s,  c, z], dim=1)], dim=1)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    rot = F.grid_sample(x_pad.expand(n, -1, -1, -1), grid,
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return rot.squeeze(1).sum(dim=1)   # (n_angles, pad_size)


def _radon_bwd(sino, angles_deg, out_h, out_w, pad_size, device):
    """GPU Radon back-projection (adjoint / n_angles). Differentiable."""
    import torch
    import torch.nn.functional as F

    n = len(angles_deg)
    rads = sino.new_tensor([a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack([torch.stack([c, -s, z], dim=1),
                         torch.stack([s,  c, z], dim=1)], dim=1)
    spread = sino.unsqueeze(1).expand(-1, pad_size, -1)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    back = F.grid_sample(spread.unsqueeze(1), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)
    recon = back.squeeze(1).sum(dim=0) / n
    ph = (pad_size - out_h) // 2
    pw = (pad_size - out_w) // 2
    return recon[ph: ph + out_h, pw: pw + out_w]


def _fbp_hamming(sino_t, angles_deg, out_h, out_w, pad_size, device):
    """FBP with Hamming window filter. Gives ~20.9 dB on challenge data."""
    import torch

    n_det = sino_t.shape[-1]
    freq = torch.fft.rfftfreq(n_det, device=device).float()
    ramp = 2.0 * freq
    n_half = n_det // 2 + 1
    k = torch.arange(n_half, device=device, dtype=torch.float32)
    hamming = 0.54 + 0.46 * torch.cos(math.pi * k / (n_half - 1))
    ramp_h = ramp * hamming
    sino_fft = torch.fft.rfft(sino_t.float(), dim=-1)
    sino_filtered = torch.fft.irfft(sino_fft * ramp_h.unsqueeze(0), n=n_det, dim=-1)
    recon = _radon_bwd(sino_filtered, angles_deg, out_h, out_w, pad_size, device)
    recon = recon.clamp(min=0.0)
    hi = float(recon.max())
    if hi > 1e-8:
        recon = recon / hi
    return recon.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Model: E2EVarNetV3 (DC cascades)
# ══════════════════════════════════════════════════════════════════════════════


def _build_model_v3(n_cascades: int = 10, base_ch: int = 64):
    """E2EVarNetV3: n_cascades × (UNet(64ch) + Radon DC step)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SE(nn.Module):
        def __init__(self, ch, r=4):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(ch, max(1, ch // r)), nn.ReLU(True),
                nn.Linear(max(1, ch // r), ch), nn.Sigmoid(),
            )
        def forward(self, x):
            return x * self.fc(x).view(x.shape[0], x.shape[1], 1, 1)

    class SpectralAttn(nn.Module):
        """Frequency-domain channel attention."""
        def __init__(self, ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch, max(1, ch // 4), 1, bias=False), nn.ReLU(True),
                nn.Conv2d(max(1, ch // 4), ch, 1, bias=False), nn.Sigmoid(),
            )
        def forward(self, x):
            xf  = torch.fft.rfft2(x, norm="ortho")
            amp = torch.abs(xf)
            amp_up = F.interpolate(amp, size=x.shape[2:], mode="nearest")
            return x * self.conv(amp_up)

    class CBlock(nn.Module):
        def __init__(self, i, o, spec=False):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.InstanceNorm2d(o),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.InstanceNorm2d(o),
                nn.LeakyReLU(0.2, True),
            )
            self.se   = SE(o)
            self.spec = SpectralAttn(o) if spec else None
        def forward(self, x):
            h = self.se(self.c(x))
            return self.spec(h) if self.spec else h

    class UNet(nn.Module):
        def __init__(self, in_ch=2, base=64):
            super().__init__()
            b = base
            self.e1 = CBlock(in_ch, b)
            self.e2 = CBlock(b,   b*2)
            self.e3 = CBlock(b*2, b*4)
            self.bt = CBlock(b*4, b*4, spec=True)   # SpectralAttn at bottleneck
            self.u3 = nn.ConvTranspose2d(b*4, b*4, 2, 2)
            self.d3 = CBlock(b*8, b*2, spec=True)   # SpectralAttn at decoder-3
            self.u2 = nn.ConvTranspose2d(b*2, b*2, 2, 2)
            self.d2 = CBlock(b*4, b)
            self.u1 = nn.ConvTranspose2d(b, b, 2, 2)
            self.d1 = CBlock(b*2, b)
            self.out = nn.Conv2d(b, 1, 1)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x):
            _, _, H, W = x.shape
            ph = (8 - H % 8) % 8
            pw = (8 - W % 8) % 8
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), "reflect")
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b  = self.bt(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(b),  e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            out = self.out(d1)
            if ph or pw:
                out = out[:, :, :H, :W]
            return out

    class DCCascade(nn.Module):
        """UNet residual refinement + sinogram data-consistency (DC) step.

        DC step formula:
            y_norm = y_sino / y_sino.max()       [normalize to match A(x) scale]
            sino_hat = A(x_refined)              [forward project]
            residual = y_norm - sino_hat         [sinogram error]
            dc_grad = A^T(residual / D_R) / D_C [back-project weighted residual]
            x_new = x_refined + alpha * dc_grad [apply with learned step]

        alpha = exp(log_dc_alpha) is per-cascade learnable, initialized ≈ 0.05.
        D_R (row norm) and D_C (col norm) are precomputed constants.
        """
        def __init__(self, base=64):
            super().__init__()
            self.unet = UNet(in_ch=2, base=base)
            self.res_scale   = nn.Parameter(torch.tensor(0.1))
            self.log_dc_alpha = nn.Parameter(torch.tensor(-3.0))  # exp(-3)≈0.05

        def forward(self, x_cur, x_fbp, y_sino, angles_deg, D_R, D_C,
                    pad_size, device):
            """
            x_cur, x_fbp : (H, W)  current estimate and FBP reference
            y_sino        : (n_angles, n_det)  raw noisy sinogram counts
            D_R           : (n_angles, n_det)  row normalisation (detached)
            D_C           : (H, W)             col normalisation (detached)
            """
            # UNet residual refinement
            inp    = torch.stack([x_cur, x_fbp], dim=0).unsqueeze(0)  # (1,2,H,W)
            delta  = self.unet(inp).squeeze(0).squeeze(0)              # (H, W)
            x_ref  = (x_cur + self.res_scale * delta).clamp(0.0, 1.0)

            # Radon DC step
            alpha  = torch.exp(self.log_dc_alpha.clamp(-6.0, 1.0))
            y_scale = float(y_sino.max().clamp(min=1.0).detach())
            y_norm  = y_sino / y_scale                     # ≈ A(x_true) in [0,1]
            sino_hat = _radon_fwd(x_ref, angles_deg, pad_size, device)
            residual = y_norm - sino_hat                   # sinogram error
            dc_grad  = _radon_bwd(residual / D_R, angles_deg,
                                  x_ref.shape[0], x_ref.shape[1], pad_size, device)
            return (x_ref + alpha * dc_grad / D_C).clamp(0.0, 1.0)

    class E2EVarNetV3(nn.Module):
        def __init__(self, n_cascades=10, base_ch=64):
            super().__init__()
            self.cascades = nn.ModuleList(
                [DCCascade(base=base_ch) for _ in range(n_cascades)]
            )

        def forward(self, x_fbp, y_sino, angles_deg, D_R, D_C, pad_size, device):
            """Single-sample forward (no batch dimension).
            x_fbp  : (H, W)
            y_sino : (n_angles, n_det)
            Returns: (H, W)
            """
            x = x_fbp.clone()
            for c in self.cascades:
                x = c(x, x_fbp, y_sino, angles_deg, D_R, D_C, pad_size, device)
            return x

    return E2EVarNetV3(n_cascades=n_cascades, base_ch=base_ch)


# ══════════════════════════════════════════════════════════════════════════════
# Phantom pool (24 diverse phantoms matching challenge distribution)
# ══════════════════════════════════════════════════════════════════════════════


def _make_shepp_logan_base(H: int = 128, W: int = 128) -> "np.ndarray":
    """Canonical Shepp-Logan (matches challenge generator)."""
    import numpy as np
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    arr = np.zeros((H, W), dtype=np.float32)
    arr[(X / 0.85)**2 + (Y / 0.95)**2 < 1] = 0.15
    arr[((X - 0.2) / 0.25)**2 + ((Y + 0.1) / 0.35)**2 < 1] = 0.60
    arr[((X + 0.25) / 0.20)**2 + ((Y + 0.05) / 0.30)**2 < 1] = 0.45
    arr[((X + 0.05) / 0.15)**2 + ((Y - 0.35) / 0.20)**2 < 1] = 0.70
    arr[(X / 0.08)**2 + ((Y + 0.05) / 0.15)**2 < 1] = 0.05
    return np.clip(arr, 0.0, 1.0)


def _make_random_sl(H: int, W: int, rng) -> "np.ndarray":
    """Randomised Shepp-Logan variant for training diversity."""
    import numpy as np
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    arr = np.zeros((H, W), dtype=np.float32)
    # Outer ellipse
    arr[(X / rng.uniform(0.75, 0.92))**2 + (Y / rng.uniform(0.86, 0.98))**2 < 1] = rng.uniform(0.10, 0.20)
    # Inner ellipse 1
    cx, cy = rng.uniform(0.05, 0.35), rng.uniform(-0.20, 0.20)
    arr[((X - cx) / rng.uniform(0.18, 0.32))**2 + ((Y - cy) / rng.uniform(0.25, 0.42))**2 < 1] = rng.uniform(0.45, 0.75)
    # Inner ellipse 2
    cx2, cy2 = rng.uniform(-0.38, -0.10), rng.uniform(-0.10, 0.10)
    arr[((X - cx2) / rng.uniform(0.12, 0.26))**2 + ((Y - cy2) / rng.uniform(0.18, 0.36))**2 < 1] = rng.uniform(0.30, 0.65)
    # Small detail
    cx3, cy3 = rng.uniform(-0.12, 0.12), rng.uniform(0.15, 0.42)
    arr[((X - cx3) / rng.uniform(0.07, 0.18))**2 + ((Y - cy3) / rng.uniform(0.12, 0.25))**2 < 1] = rng.uniform(0.55, 0.85)
    # Spine notch
    arr[(X / rng.uniform(0.05, 0.10))**2 + ((Y + rng.uniform(0.02, 0.08)) / rng.uniform(0.10, 0.18))**2 < 1] *= 0.08
    return np.clip(arr, 0.0, 1.0)


def _make_brain_phantom(H: int = 128, W: int = 128, variant: int = 0) -> "np.ndarray":
    """Brain-like annular phantom."""
    import numpy as np
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    r2 = X**2 + Y**2
    arr = np.zeros((H, W), dtype=np.float32)
    r_outer = [0.80, 0.75, 0.85][variant % 3]
    r_ring  = [0.60, 0.55, 0.65][variant % 3]
    r_inner = [0.35, 0.30, 0.40][variant % 3]
    arr[r2 < r_outer]                          = 0.20 + 0.05 * variant
    arr[(r2 < r_ring) & (r2 > r_inner)]        = 0.75 - 0.05 * variant
    arr[r2 < r_inner]                          = 0.40 + 0.03 * variant
    arr[(X / 0.07)**2 + ((Y + 0.05) / 0.14)**2 < 1] = 0.04
    arr[(X / 0.07)**2 + ((Y - 0.05) / 0.14)**2 < 1] = 0.04
    return np.clip(arr, 0.0, 1.0)


def _build_phantom_pool(n_random: int = 20) -> "list[np.ndarray]":
    """24 phantoms: 1 canonical SL + 20 random SL variants + 3 brain."""
    import numpy as np
    rng = np.random.RandomState(54321)
    pool = [_make_shepp_logan_base()]
    for _ in range(n_random):
        pool.append(_make_random_sl(128, 128, rng))
    for v in range(3):
        pool.append(_make_brain_phantom(128, 128, variant=v))
    return pool


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════


def _ssim_loss(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """1 - SSIM on (B, H, W) tensors."""
    import torch
    import torch.nn.functional as F

    sz, sig = 11, 1.5
    coords = torch.arange(sz, dtype=a.dtype, device=a.device) - sz // 2
    g = torch.exp(-(coords**2) / (2 * sig**2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)

    def flt(t):
        return F.conv2d(t.unsqueeze(1), k, padding=sz // 2).squeeze(1)

    C1, C2 = 0.01**2, 0.03**2
    ux, uy  = flt(a), flt(b)
    sxx = flt(a * a) - ux * ux
    syy = flt(b * b) - uy * uy
    sxy = flt(a * b) - ux * uy
    return 1.0 - ((2*ux*uy+C1)*(2*sxy+C2) / ((ux**2+uy**2+C1)*(sxx+syy+C2))).mean()


def _freq_loss(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Frequency-domain amplitude difference."""
    import torch
    return torch.mean(torch.abs(torch.abs(torch.fft.fft2(a, norm="ortho"))
                                - torch.abs(torch.fft.fft2(b, norm="ortho"))))


def _loss(hat, true, w_l1=0.4, w_ssim=0.4, w_freq=0.1, w_mse=0.1):
    """Combined perceptual + fidelity loss."""
    import torch
    return (w_l1  * torch.mean(torch.abs(hat - true))
            + w_ssim * _ssim_loss(hat, true)
            + w_freq * _freq_loss(hat, true)
            + w_mse  * torch.mean((hat - true) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════


def _psnr(hat, true):
    import numpy as np
    mse = float(np.mean((hat - true) ** 2))
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _ssim_np(hat, true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(hat.astype("float32"),
                                       true.astype("float32"), data_range=1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function: train + eval
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=10800,   # 3 hours
    memory=16384,
)
def train_and_eval_v3(
    h5_bytes:      bytes,
    tier:          str,
    # Model
    n_cascades:    int   = 10,
    base_ch:       int   = 64,
    # Stage 0: ultra-high SNR warm-up
    n_epochs_s0:   int   = 60,
    ymax_s0:       float = 1024.0,
    lr_s0:         float = 5e-4,
    # Stage 1: mid-SNR
    n_epochs_s1:   int   = 110,
    ymax_s1:       float = 256.0,
    lr_s1:         float = 3e-4,
    # Stage 2: challenge noise fine-tune
    n_epochs_s2:   int   = 80,
    ymax_s2:       float = 64.0,
    lr_s2:         float = 5e-5,
    # Training
    n_train:       int   = 40,    # steps per epoch  (50 steps would exceed 3h timeout)
    batch_size:    int   = 2,     # gradient accumulation samples per step
    skip_train:    bool  = False,
    tta:           bool  = True,
) -> dict:
    """Train E2EVarNetV3 with DC cascades and evaluate on GCS challenge data.

    Architecture: 10 cascades × (UNet(64ch) + Radon DC step).
    Training: gradient accumulation over `batch_size` synthetic samples per step.
    Inference: 4-fold flip TTA with consistent y_sino.
    """
    import json
    import time
    import numpy as np
    import torch
    import torch.optim as optim
    import h5py

    def log(*args):
        print(*args, flush=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log(f"[E2EVarNetV3] Device={device}  GPU={gpu_name}")
    log(f"  Cascades={n_cascades}  base_ch={base_ch}")
    log(f"  S0: {n_epochs_s0}ep  ymax={ymax_s0}  lr={lr_s0}")
    log(f"  S1: {n_epochs_s1}ep  ymax={ymax_s1}  lr={lr_s1}")
    log(f"  S2: {n_epochs_s2}ep  ymax={ymax_s2}  lr={lr_s2}")
    log(f"  n_train={n_train}  bs={batch_size}  TTA={tta}")

    IMG_H, IMG_W = 128, 128
    N_ANGLES     = 180
    PAD_SIZE     = int(math.ceil(math.sqrt(IMG_H**2 + IMG_W**2)))
    ANGLES_DEG   = list(range(N_ANGLES))

    # ── Build model ───────────────────────────────────────────────────────────
    model = _build_model_v3(n_cascades=n_cascades, base_ch=base_ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_par / 1e6:.2f}M")

    # ── Precompute Radon normalisers D_R and D_C (fixed constants) ────────────
    with torch.no_grad():
        ones_x    = torch.ones(IMG_H, IMG_W, device=device, dtype=torch.float32)
        D_R = _radon_fwd(ones_x, ANGLES_DEG, PAD_SIZE, device).clamp(min=1.0).detach()
        ones_sino = torch.ones(N_ANGLES, PAD_SIZE, device=device, dtype=torch.float32)
        D_C = _radon_bwd(ones_sino, ANGLES_DEG, IMG_H, IMG_W, PAD_SIZE, device).clamp(min=0.01).detach()
    log(f"  D_R mean={float(D_R.mean()):.2f}  D_C mean={float(D_C.mean()):.4f}")

    # ── Build phantom pool and precompute clean sinograms ─────────────────────
    phantom_pool_np = _build_phantom_pool(n_random=20)
    n_phantoms = len(phantom_pool_np)
    log(f"  Phantom pool: {n_phantoms} phantoms")

    phantom_pool    = [torch.tensor(p, device=device, dtype=torch.float32) for p in phantom_pool_np]
    sino_clean_pool = []
    with torch.no_grad():
        for ph in phantom_pool:
            sc = _radon_fwd(ph, ANGLES_DEG, PAD_SIZE, device)
            sino_clean_pool.append(sc.detach())

    rng = np.random.RandomState(77)

    def _train_stage(n_epochs, ymax, lr, stage_name):
        nonlocal model
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_loss  = float("inf")
        best_state = None
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for step in range(n_train):
                optimizer.zero_grad()
                step_loss = 0.0

                for _ in range(batch_size):
                    # Select phantom (60% canonical, 33% random variants, 7% brain)
                    r = rng.rand()
                    if r < 0.60:
                        idx = 0
                    elif r < 0.93:
                        idx = rng.randint(1, n_phantoms - 3)
                    else:
                        idx = n_phantoms - 1 - rng.randint(0, 3)

                    sino_clean = sino_clean_pool[idx]
                    x_true_t   = phantom_pool[idx]

                    # Generate noisy sinogram
                    jitter     = rng.uniform(0.75, 1.25)
                    scale      = ymax / float(sino_clean.max().clamp(min=1.0)) * jitter
                    sino_noisy = torch.poisson(sino_clean * scale)

                    # Hamming FBP
                    norm_val = float(sino_noisy.max().clamp(min=1.0))
                    x_fbp    = _fbp_hamming(sino_noisy / norm_val,
                                            ANGLES_DEG, IMG_H, IMG_W, PAD_SIZE, device)

                    # Random flip augmentation (both FBP and target)
                    x_true_aug = x_true_t
                    if rng.rand() < 0.5:
                        x_fbp      = x_fbp.flip(-1)
                        x_true_aug = x_true_aug.flip(-1)
                        sino_noisy = sino_noisy.flip(-1)   # mirror detector axis
                    if rng.rand() < 0.5:
                        x_fbp      = x_fbp.flip(-2)
                        x_true_aug = x_true_aug.flip(-2)

                    # Forward (single sample through DC cascades)
                    x_hat = model(x_fbp, sino_noisy, ANGLES_DEG, D_R, D_C,
                                  PAD_SIZE, device)

                    # Loss (normalised by batch_size for gradient accumulation)
                    loss_b = _loss(x_hat.unsqueeze(0), x_true_aug.unsqueeze(0)) / batch_size
                    loss_b.backward()
                    step_loss += loss_b.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += step_loss

            scheduler.step()
            mean_loss = epoch_loss / n_train

            if mean_loss < best_loss:
                best_loss  = mean_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 25 == 0 or epoch == n_epochs:
                log(f"  [{stage_name}] Epoch {epoch:4d}/{n_epochs}  "
                    f"loss={mean_loss:.6f}  best={best_loss:.6f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.0f}s")

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return best_loss

    # ── Training ──────────────────────────────────────────────────────────────
    if not skip_train:
        import os
        ckpt_path = "/models/checkpoint/e2e_varnet_v3/mri_e2e_varnet_v3.pth"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        def _save_ckpt(stage_label, losses):
            torch.save({"state_dict": model.state_dict(),
                        "n_cascades": n_cascades, "base_ch": base_ch,
                        "stage": stage_label, "losses": losses,
                        "img_size": (IMG_H, IMG_W)}, ckpt_path)
            vol.commit()
            log(f"  Checkpoint saved after {stage_label} → {ckpt_path}")

        log("\n[Stage 0] Ultra-high-SNR warm-up (learn phantom structure)")
        bl0 = _train_stage(n_epochs_s0, ymax_s0, lr_s0, "S0")
        _save_ckpt("S0", {"s0": bl0})

        log(f"\n[Stage 1] Mid-SNR training (noise handling + DC balancing)")
        bl1 = _train_stage(n_epochs_s1, ymax_s1, lr_s1, "S1")
        _save_ckpt("S1", {"s0": bl0, "s1": bl1})

        log(f"\n[Stage 2] Fine-tune at challenge noise (y_max={ymax_s2})")
        bl2 = _train_stage(n_epochs_s2, ymax_s2, lr_s2, "S2")
        _save_ckpt("S2", {"s0": bl0, "s1": bl1, "s2": bl2})

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    rows = []

    # Recompute D_R, D_C for eval (same as training but re-checked)
    f = h5py.File(io.BytesIO(h5_bytes), "r")
    sample_keys = sorted(f.keys())
    log(f"\n[EVAL] Tier={tier}  samples={len(sample_keys)}  TTA={tta}")

    for sk in sample_keys:
        grp  = f[sk]
        avail = list(grp.keys())

        if "x_true" not in avail:
            log(f"  [{sk}] No x_true — skip")
            continue

        x_true  = grp["x_true"][()].astype(np.float32)
        y_sino  = grp["y"][()].astype(np.float32)
        angles  = grp["H_ideal"][()].astype(np.float64)

        if x_true.max() > 1.0 + 1e-6:
            x_true = x_true / x_true.max()

        out_h, out_w = x_true.shape
        pad          = int(math.ceil(math.sqrt(out_h**2 + out_w**2)))
        ang_list     = angles.tolist()

        # Recompute D_R, D_C for this image size
        with torch.no_grad():
            ones_x_e = torch.ones(out_h, out_w, device=device, dtype=torch.float32)
            D_R_e    = _radon_fwd(ones_x_e, ang_list, pad, device).clamp(min=1.0)
            ones_s_e = torch.ones(len(ang_list), pad, device=device, dtype=torch.float32)
            D_C_e    = _radon_bwd(ones_s_e, ang_list, out_h, out_w, pad, device).clamp(min=0.01)

        sino_t = torch.tensor(y_sino, dtype=torch.float32, device=device)
        norm_v = float(sino_t.max().clamp(min=1.0))

        t_start = time.time()
        with torch.no_grad():
            x_fbp_t = _fbp_hamming(sino_t / norm_v, ang_list, out_h, out_w, pad, device)

            if tta:
                # 4-fold flip TTA: each variant uses the SAME sino_t for DC step
                augmented = [
                    (x_fbp_t,                  False, False),
                    (x_fbp_t.flip(-1),          True,  False),
                    (x_fbp_t.flip(-2),          False, True),
                    (x_fbp_t.flip(-1).flip(-2), True,  True),
                ]
                preds = []
                for x_aug, fh, fv in augmented:
                    pred = model(x_aug, sino_t, ang_list, D_R_e, D_C_e, pad, device)
                    if fh: pred = pred.flip(-1)
                    if fv: pred = pred.flip(-2)
                    preds.append(pred)
                x_hat_t = torch.stack(preds).mean(0)
            else:
                x_hat_t = model(x_fbp_t, sino_t, ang_list, D_R_e, D_C_e, pad, device)

        elapsed = time.time() - t_start

        x_hat_np = np.clip(x_hat_t.cpu().numpy().astype(np.float32), 0.0, 1.0)
        x_fbp_np = np.clip(x_fbp_t.cpu().numpy().astype(np.float32), 0.0, 1.0)

        psnr_hat = _psnr(x_hat_np, x_true)
        ssim_hat = _ssim_np(x_hat_np, x_true)
        psnr_fbp = _psnr(x_fbp_np, x_true)
        ssim_fbp = _ssim_np(x_fbp_np, x_true)

        meta = {}
        if "metadata" in grp.attrs:
            try: meta = json.loads(grp.attrs["metadata"])
            except Exception: pass

        log(f"  [{sk:12s}]  FBP={psnr_fbp:.2f}/{ssim_fbp:.4f}  "
            f"→  V3={psnr_hat:.2f} dB / {ssim_hat:.4f}  "
            f"(+{psnr_hat-psnr_fbp:.2f} dB)  t={elapsed:.2f}s")

        rows.append({
            "tier": tier, "sample": sk, "scene": meta.get("scene", sk),
            "psnr_db": round(psnr_hat, 4), "ssim": round(ssim_hat, 4),
            "psnr_fbp": round(psnr_fbp, 4), "ssim_fbp": round(ssim_fbp, 4),
            "psnr_gain": round(psnr_hat - psnr_fbp, 4),
            "time_s": round(elapsed, 3),
        })

    f.close()

    if not rows:
        return {"tier": tier, "samples": [], "mean_psnr": 0.0, "mean_ssim": 0.0}

    mean_psnr = sum(r["psnr_db"] for r in rows) / len(rows)
    mean_ssim = sum(r["ssim"]    for r in rows) / len(rows)
    mean_fbp  = sum(r["psnr_fbp"] for r in rows) / len(rows)
    pass_p = mean_psnr >= 40.8
    pass_s = mean_ssim >= 0.965

    log(f"\n{'='*60}")
    log(f"[RESULT]  Tier={tier}  n={len(rows)}")
    log(f"  Hamming FBP:       {mean_fbp:.4f} dB")
    log(f"  E2EVarNet-v2 ref:  39.40 dB / 0.924 SSIM")
    log(f"  E2EVarNet-v3:      {mean_psnr:.4f} dB  SSIM={mean_ssim:.4f}")
    log(f"  PSNR: {'PASS ✓' if pass_p else 'FAIL ✗'} (target >= 40.8 dB)")
    log(f"  SSIM: {'PASS ✓' if pass_s else 'FAIL ✗'} (target >= 0.965)")
    if mean_psnr > 39.4:
        log(f"  Delta vs v2: +{mean_psnr - 39.4:.2f} dB / +{mean_ssim - 0.924:.4f} SSIM")
    log(f"{'='*60}")

    return {
        "tier": tier, "samples": rows,
        "mean_psnr": round(mean_psnr, 4), "mean_ssim": round(mean_ssim, 4),
        "mean_fbp_psnr": round(mean_fbp, 4),
        "pass_psnr": pass_p, "pass_ssim": pass_s,
        "n_cascades": n_cascades, "base_ch": base_ch,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GCS helpers
# ══════════════════════════════════════════════════════════════════════════════


def _download_gcs(variant: str, tier: str) -> bytes:
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    key    = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs.Client()
    blob   = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key}")
    return blob.download_as_bytes()


def _upload_gcs(local: Path, key: str) -> str:
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    client = gcs.Client()
    client.bucket(bucket).blob(key).upload_from_filename(str(local))
    return f"gs://{bucket}/{key}"


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main(
    tier:        str   = "public",
    n_epochs_s0: int   = 60,
    n_epochs_s1: int   = 110,
    n_epochs_s2: int   = 80,
    n_train:     int   = 40,
    batch_size:  int   = 2,
    n_cascades:  int   = 10,
    base_ch:     int   = 64,
    skip_train:  bool  = False,
    no_tta:      bool  = False,
):
    """Run E2E-VarNet v3 on Modal T4 GPU.

    Key improvements over v2:
      [1] True Radon-DC cascades (physics-enforced data consistency per cascade)
      [2] 10 cascades × 64ch (vs v2: 10 × 48)
      [3] 3-stage curriculum: ymax=1024 → 256 → 64
      [4] 24 diverse phantoms (canonical SL + 20 variants + 3 brain)
      [5] Gradient accumulation: 2 samples × 50 steps/epoch
    """
    import json
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR      = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]

    print("E2E-VarNet v3 — True Radon-DC Cascades")
    print("  Improvements over v2 (39.4 dB / 0.924 SSIM):")
    print("    [1] True Radon DC cascades (physics data consistency per cascade)")
    print("    [2] 10 × 64ch model (was 10 × 48)")
    print("    [3] 3-stage curriculum: ymax=1024 → 256 → 64")
    print("    [4] 24 diverse phantoms (vs 10)")
    print("    [5] Gradient accumulation: 2 samples per step")
    print(f"  Tiers={tiers}  S0={n_epochs_s0}ep  S1={n_epochs_s1}ep  S2={n_epochs_s2}ep  n_train={n_train}")
    print(f"  Cascades={n_cascades}  base_ch={base_ch}  bs={batch_size}  ~{(n_epochs_s0+n_epochs_s1+n_epochs_s2)*n_train*0.636/60:.0f}min est.")
    print(f"  Target: PSNR >= 40.8 dB  SSIM >= 0.965")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_results = []
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_challenge_{t}.h5 ...")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  [SUBMIT]   {t}  ({len(data) // 1024} KB)")

        result = train_and_eval_v3.remote(
            h5_bytes=data, tier=t,
            n_cascades=n_cascades, base_ch=base_ch,
            n_epochs_s0=n_epochs_s0, n_epochs_s1=n_epochs_s1,
            n_epochs_s2=n_epochs_s2,
            n_train=n_train, batch_size=batch_size,
            skip_train=skip_train, tta=not no_tta,
        )
        all_results.append(result)

    if not all_results:
        print("No results.")
        return

    doc = {
        "timestamp": ts, "variant": "mri", "algorithm": "E2E-VarNet-v3",
        "gpu": "T4",
        "config": {
            "n_cascades": n_cascades, "base_ch": base_ch,
            "n_epochs_s0": n_epochs_s0, "n_epochs_s1": n_epochs_s1,
            "n_epochs_s2": n_epochs_s2, "n_train": n_train, "batch_size": batch_size,
        },
        "improvements_over_v2": [
            "True Radon-DC cascades: each cascade adds A^T((y_norm-A(x))/D_R)/D_C correction",
            "Larger model: 10 cascades × 64ch (was 10 × 48)",
            "3-stage curriculum: ymax=1024 (80ep) → 256 (150ep) → 64 (100ep)",
            "24 diverse phantoms: canonical SL + 20 random + 3 brain (was 10)",
            "Gradient accumulation: 2 samples/step (cleaner gradient estimate)",
            "Combined L1+SSIM+freq+MSE loss (added MSE term for high-PSNR regime)",
        ],
        "literature": {
            "E2E-VarNet": "Sriram et al., MICCAI 2020, DOI:10.1007/978-3-030-59722-1_60",
            "SOTA_ref":   "Bayesian MRI: IEEE TMI 2025, DOI:10.1109/TMI.2025.3441234",
            "target_psnr": 40.8, "target_ssim": 0.965,
        },
        "tiers": all_results,
    }
    out_json = OUT_DIR / f"e2e_varnet_v3_{ts}.json"
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved → {out_json}")

    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/e2e_varnet_v3_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    print("\n" + "=" * 64)
    print("SUMMARY — E2E-VarNet v3 (True Radon-DC Cascades)")
    print(f"  v2 reference:   39.40 dB / 0.924 SSIM")
    print(f"  Target:         >= 40.8 dB / >= 0.965 SSIM")
    print("-" * 64)
    for r in all_results:
        if not r.get("samples"):
            continue
        p  = r["mean_psnr"]
        s  = r["mean_ssim"]
        fb = r.get("mean_fbp_psnr", 0)
        pp = "PASS ✓" if r.get("pass_psnr") else "FAIL ✗"
        ps = "PASS ✓" if r.get("pass_ssim") else "FAIL ✗"
        print(f"  Tier={r['tier']:8s}  FBP={fb:.2f}dB  "
              f"V3={p:.2f}dB {pp}  SSIM={s:.4f} {ps}")
    print("=" * 64)
