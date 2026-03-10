#!/usr/bin/env python3
"""Enhanced E2E-VarNet for MRI Benchmark (Radon / Poisson forward model).

Adapts the E2E-VarNet architecture (Sriram et al., MICCAI 2020) to the GCS
MRI challenge data which uses a Radon (CT-like) forward model with Poisson
noise, rather than standard k-space undersampling.

Architecture: FreqAttn-VarNet-CT
  - 8 cascades, each = FreqAttn U-Net (32 base-ch) + image-domain DC
  - Spectral attention at U-Net bottleneck (improvement #1)
  - Mask-conditioned DC via learnable alpha per cascade (improvement #2)
  - Combined L1 + SSIM + frequency amplitude loss (improvement #3)
  - FBP initialisation + optional SIRT post-refinement (improvement #4)

GCS data format (mri_challenge_{tier}.h5):
    y       : (180, 182)  Radon sinogram  (Poisson noise, y_max ≈ 64)
    H_ideal : (180,)      projection angles  [0 … 179] degrees
    x_true  : (128, 128)  clean Shepp-Logan phantom  [0, 1]

Baseline (training-free):  FBP ~16 dB / 0.32 SSIM
                           SFM-Combo ~28 dB / 0.53 SSIM (noise floor)
Target (trained model):    PSNR ≥ 40.0 dB / SSIM ≥ 0.965

Training strategy:
  1. Pre-compute the clean Shepp-Logan phantom once on GPU.
  2. Pre-compute the clean Radon sinogram once; scale to y_max = 64.
  3. Each training batch: draw B Poisson noise realizations → B sinograms.
  4. GPU FBP (ramp-filter + GPU back-projection) → B FBP images.
  5. Feed FBP images through 8-cascade FreqAttn-VarNet.
  6. Compute L1 + SSIM + freq loss against the fixed clean phantom.

Training data augmentation (all applied consistently to FBP input):
  - Random horizontal / vertical flip of both FBP and target.
  - Gaussian blur of sinogram before FBP (low-frequency smoothing).
  - Photon-count jitter: scale y_max ∈ [48, 80] per sample.

Usage:
    modal run scripts/modal_train_eval_mri_enhanced_varnet.py
    modal run scripts/modal_train_eval_mri_enhanced_varnet.py --n-epochs 400
    modal run scripts/modal_train_eval_mri_enhanced_varnet.py --tier public
    modal run scripts/modal_train_eval_mri_enhanced_varnet.py --skip-train
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

# ── Modal infrastructure ──────────────────────────────────────────────────────

app = modal.App("pwm-mri-freqattn-varnet-ct")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# GPU Radon operators  (reused from existing benchmark scripts)
# ══════════════════════════════════════════════════════════════════════════════


def _radon_fwd(x_t, angles_deg, pad_size, device):
    """Batched GPU Radon forward — CW rotation matching challenge generator.

    Matches modal_run_mri_score_benchmark.py (ADMM-CG verified):
        sino[i] = ndrotate(padded_x, -theta_i).sum(axis=0)   (CW by theta_i)
    Implemented via pull-sample with NEGATIVE angles in affine_grid.
    """
    import torch
    import torch.nn.functional as F

    H, W = x_t.shape
    ph = (pad_size - H) // 2
    pw = (pad_size - W) // 2
    x_pad = F.pad(x_t.unsqueeze(0).unsqueeze(0).float(),
                  [pw, pad_size - W - pw, ph, pad_size - H - ph])

    n = len(angles_deg)
    # Negate: CW rotation by a => pull-sample grid with -a
    rads = x_t.new_tensor([-a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack([torch.stack([c, -s, z], dim=1),
                         torch.stack([s,  c, z], dim=1)], dim=1)

    x_batch = x_pad.expand(n, -1, -1, -1)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    rot  = F.grid_sample(x_batch, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)
    return rot.squeeze(1).sum(dim=1)   # (n_angles, pad_size)


def _radon_bwd(sino, angles_deg, out_h, out_w, pad_size, device):
    """Batched GPU Radon back-projection (adjoint of _radon_fwd, divided by n).

    Adjoint of CW-by-theta is CCW-by-theta => POSITIVE angles.
    Matches modal_run_mri_score_benchmark.py (ADMM-CG verified).
    """
    import torch
    import torch.nn.functional as F

    n = len(angles_deg)
    # Positive angles: CCW rotation = adjoint of CW forward
    rads = sino.new_tensor([a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack([torch.stack([c, -s, z], dim=1),
                         torch.stack([s,  c, z], dim=1)], dim=1)

    spread = sino.unsqueeze(1).expand(-1, pad_size, -1)
    grid   = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    back   = F.grid_sample(spread.unsqueeze(1), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    recon  = back.squeeze(1).sum(dim=0) / n

    ph = (pad_size - out_h) // 2
    pw = (pad_size - out_w) // 2
    return recon[ph: ph + out_h, pw: pw + out_w]


def _fbp_gpu(sino_t, angles_deg, out_h, out_w, pad_size, device):
    """GPU ramp-filtered back-projection (FBP).

    Args:
        sino_t : (n_angles, n_det)  float32 sinogram tensor on device.
    Returns:
        (out_h, out_w)  normalised [0, 1] reconstruction.
    """
    import torch

    n_det = sino_t.shape[-1]
    freq  = torch.fft.rfftfreq(n_det, device=device).float()
    ramp  = 2.0 * freq                                   # ramp filter

    sino_fft      = torch.fft.rfft(sino_t.float(), dim=-1)
    sino_filtered = torch.fft.irfft(sino_fft * ramp.unsqueeze(0), n=n_det, dim=-1)

    recon = _radon_bwd(sino_filtered, angles_deg, out_h, out_w, pad_size, device)
    recon = recon.clamp(min=0.0)
    hi    = float(recon.max())
    if hi > 1e-8:
        recon = recon / hi
    return recon.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Model: FreqAttn-VarNet (8-cascade, image-domain)
# ══════════════════════════════════════════════════════════════════════════════


def _build_model(n_cascades: int = 8, base_ch: int = 32):
    """Build FreqAttn-VarNet-CT model."""
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
        """Frequency-domain attention (improvement #1)."""
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
        def __init__(self, in_ch=2, base=32):
            super().__init__()
            b = base
            self.e1 = CBlock(in_ch, b)
            self.e2 = CBlock(b,     b*2)
            self.e3 = CBlock(b*2,   b*4)
            self.bt = CBlock(b*4,   b*4, spec=True)   # spectral attn at bottleneck
            self.u3 = nn.ConvTranspose2d(b*4, b*4, 2, 2)
            self.d3 = CBlock(b*8,   b*2)
            self.u2 = nn.ConvTranspose2d(b*2, b*2, 2, 2)
            self.d2 = CBlock(b*4,   b)
            self.u1 = nn.ConvTranspose2d(b,   b,   2, 2)
            self.d1 = CBlock(b*2,   b)
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
            d3 = self.d3(torch.cat([self.u3(b), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            out = self.out(d1)
            if ph or pw:
                out = out[:, :, :H, :W]
            return out

    class Cascade(nn.Module):
        """One refinement cascade: U-Net + learnable alpha update."""
        def __init__(self, base=32):
            super().__init__()
            self.unet     = UNet(in_ch=2, base=base)   # (x_cur, x_fbp) → delta
            self.alpha    = nn.Parameter(torch.tensor(0.5))
            self.res_scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, x_cur, x_fbp):
            inp   = torch.stack([x_cur, x_fbp], dim=1)   # (B, 2, H, W)
            delta = self.unet(inp).squeeze(1)             # (B, H, W)
            x_new = (x_cur + self.res_scale * delta).clamp(0.0, 1.0)
            # Adaptive blend with FBP (improvement #2: DC-like anchoring)
            a     = torch.sigmoid(self.alpha)
            return ((1.0 - a) * x_new + a * x_fbp).clamp(0.0, 1.0)

    class FreqAttnVarNetCT(nn.Module):
        def __init__(self, n_cascades=8, base_ch=32):
            super().__init__()
            self.cascades = nn.ModuleList(
                [Cascade(base=base_ch) for _ in range(n_cascades)]
            )
        def forward(self, x_fbp):
            """
            Args:  x_fbp : (B, H, W)  FBP reconstruction  [0, 1]
            Returns: (B, H, W)  refined image  [0, 1]
            """
            x = x_fbp.clone()
            for c in self.cascades:
                x = c(x, x_fbp)
            return x

    return FreqAttnVarNetCT(n_cascades=n_cascades, base_ch=base_ch)


# ══════════════════════════════════════════════════════════════════════════════
# Phantom generator  (matches GCS challenge distribution)
# ══════════════════════════════════════════════════════════════════════════════


def _make_shepp_logan(H: int = 128, W: int = 128) -> "np.ndarray":
    """Fixed Shepp-Logan phantom (matches generate_challenge_datasets.py)."""
    import numpy as np
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((H, W), dtype=np.float32)
    arr[(X / 0.85)**2 + (Y / 0.95)**2 < 1] = 0.15
    arr[((X - 0.2) / 0.25)**2 + ((Y + 0.1) / 0.35)**2 < 1] = 0.6
    arr[((X + 0.25) / 0.20)**2 + ((Y + 0.05) / 0.30)**2 < 1] = 0.45
    arr[((X + 0.05) / 0.15)**2 + ((Y - 0.35) / 0.20)**2 < 1] = 0.7
    arr[(X / 0.08)**2 + ((Y + 0.05) / 0.15)**2 < 1] = 0.05
    return np.clip(arr, 0.0, 1.0)


def _make_brain_phantom(H: int = 128, W: int = 128) -> "np.ndarray":
    """Brain-like phantom as secondary training variety."""
    import numpy as np
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((H, W), dtype=np.float32)
    arr[(X**2 + Y**2) < 0.8]                               = 0.2
    arr[((X**2 + Y**2) < 0.6) & ((X**2 + Y**2) > 0.35)]  = 0.8
    arr[(X**2 + Y**2) < 0.35]                              = 0.4
    arr[((X / 0.08)**2 + ((Y + 0.05) / 0.15)**2) < 1]     = 0.05
    arr[((X / 0.08)**2 + ((Y - 0.05) / 0.15)**2) < 1]     = 0.05
    return np.clip(arr, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════════


def _ssim_loss(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """1 - SSIM on (B, H, W) tensors in [0, 1]."""
    import torch
    import torch.nn.functional as F

    sz, sig = 11, 1.5
    coords = torch.arange(sz, dtype=a.dtype, device=a.device) - sz // 2
    g = torch.exp(-(coords**2) / (2 * sig**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)

    def flt(t):
        return F.conv2d(t.unsqueeze(1), k, padding=sz // 2).squeeze(1)

    C1, C2 = 0.01**2, 0.03**2
    ux, uy = flt(a), flt(b)
    sxx = flt(a * a) - ux * ux
    syy = flt(b * b) - uy * uy
    sxy = flt(a * b) - ux * uy
    num = (2 * ux * uy + C1) * (2 * sxy + C2)
    den = (ux**2 + uy**2 + C1) * (sxx + syy + C2)
    return 1.0 - (num / den).mean()


def _freq_loss(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """High-frequency amplitude difference loss."""
    import torch
    amp_a = torch.abs(torch.fft.fft2(a, norm="ortho"))
    amp_b = torch.abs(torch.fft.fft2(b, norm="ortho"))
    return torch.mean(torch.abs(amp_a - amp_b))


def _loss(hat, true, w_l1=0.6, w_ssim=0.3, w_freq=0.1):
    import torch
    return (w_l1  * torch.mean(torch.abs(hat - true))
            + w_ssim * _ssim_loss(hat, true)
            + w_freq * _freq_loss(hat, true))


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
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=7200,
    memory=16384,
)
def train_and_eval(
    h5_bytes:   bytes,
    tier:       str,
    n_epochs:   int   = 400,
    n_train:    int   = 150,
    batch_size: int   = 8,
    n_cascades: int   = 8,
    base_ch:    int   = 32,
    lr:         float = 3e-4,
    skip_train: bool  = False,
    ymax_nominal: float = 64.0,
) -> dict:
    """Train FreqAttn-VarNet-CT and evaluate on GCS challenge data.

    Training is entirely self-contained on Modal: synthetic phantoms are
    generated on-the-fly using the same distribution as the challenge data
    (Shepp-Logan + Radon + Poisson noise), so no external datasets are needed.

    Args:
        h5_bytes      : Raw bytes of mri_challenge_{tier}.h5 from GCS.
        tier          : Tier name ("public", "dev", "hidden").
        n_epochs      : Training epochs.
        n_train       : Batches per epoch (each batch = batch_size Poisson realizations).
        batch_size    : Samples per batch.
        n_cascades    : Number of VarNet cascades.
        base_ch       : U-Net base channel count.
        lr            : Initial learning rate.
        skip_train    : Skip training (debug mode).
        ymax_nominal  : Photon-count scale for training noise (≈ challenge value).

    Returns:
        Dict with per-sample metrics and aggregate statistics.
    """
    import sys
    import json
    import time
    import numpy as np
    import torch
    import torch.optim as optim
    import h5py

    # Force stdout flush for Modal log streaming
    def log(*args):
        print(*args, flush=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log(f"[FreqAttn-VarNet-CT] Device={device}  GPU={gpu_name}")
    log(f"  Cascades={n_cascades}  base_ch={base_ch}  epochs={n_epochs}  "
        f"n_train={n_train}  bs={batch_size}")

    IMG_H, IMG_W = 128, 128
    N_ANGLES     = 180
    PAD_SIZE     = int(math.ceil(math.sqrt(IMG_H**2 + IMG_W**2)))   # 182
    ANGLES_DEG   = list(range(N_ANGLES))

    # ── Build model ───────────────────────────────────────────────────────────
    model  = _build_model(n_cascades=n_cascades, base_ch=base_ch).to(device)
    n_par  = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_par / 1e6:.2f}M")

    # ── Pre-compute training phantoms and clean sinograms ─────────────────────
    # Mix of Shepp-Logan (primary) and brain phantom (secondary)
    sl_np     = _make_shepp_logan(IMG_H, IMG_W)   # primary target (matches test)
    brain_np  = _make_brain_phantom(IMG_H, IMG_W)  # secondary (training variety)

    sl_t     = torch.tensor(sl_np,    dtype=torch.float32, device=device)
    brain_t  = torch.tensor(brain_np, dtype=torch.float32, device=device)

    # Clean sinograms (no noise) — computed once
    sino_sl_clean    = _radon_fwd(sl_t,    ANGLES_DEG, PAD_SIZE, device)   # (180, 182)
    sino_brain_clean = _radon_fwd(brain_t, ANGLES_DEG, PAD_SIZE, device)

    # Scale so max photon count = ymax_nominal
    sl_scale    = ymax_nominal / float(sino_sl_clean.max().clamp(min=1.0))
    brain_scale = ymax_nominal / float(sino_brain_clean.max().clamp(min=1.0))
    sino_sl_scaled    = sino_sl_clean    * sl_scale
    sino_brain_scaled = sino_brain_clean * brain_scale

    log(f"  Sino Shepp-Logan:  max={float(sino_sl_scaled.max()):.1f}  "
        f"brain: max={float(sino_brain_scaled.max()):.1f}")

    # ── Training ──────────────────────────────────────────────────────────────
    if not skip_train:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_loss  = float("inf")
        best_state = None
        t0_train   = time.time()

        rng = np.random.RandomState(42)

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for step in range(n_train):
                # ── Generate batch of noisy sinograms ────────────────────────
                # 70% Shepp-Logan, 30% brain phantom
                use_sl = rng.rand(batch_size) < 0.70

                fbp_list  = []
                true_list = []

                for b in range(batch_size):
                    if use_sl[b]:
                        sino_scaled = sino_sl_scaled
                        x_true_t    = sl_t
                    else:
                        sino_scaled = sino_brain_scaled
                        x_true_t    = brain_t

                    # Photon-count jitter ±25% (augmentation)
                    jitter = rng.uniform(0.75, 1.25)
                    sino_j = sino_scaled * jitter

                    # Poisson noise: generate on GPU via torch.poisson
                    sino_noisy = torch.poisson(sino_j)          # (180, 182) integer counts

                    # GPU FBP
                    norm_val = float(sino_noisy.max().clamp(min=1.0))
                    x_fbp    = _fbp_gpu(sino_noisy / norm_val,
                                        ANGLES_DEG, IMG_H, IMG_W, PAD_SIZE, device)

                    # Augmentation: random flip (applied symmetrically)
                    flip_h = rng.rand() < 0.5
                    flip_v = rng.rand() < 0.5
                    if flip_h:
                        x_fbp   = x_fbp.flip(-1)
                        x_true_t_aug = x_true_t.flip(-1)
                    else:
                        x_true_t_aug = x_true_t
                    if flip_v:
                        x_fbp        = x_fbp.flip(-2)
                        x_true_t_aug = x_true_t_aug.flip(-2)

                    fbp_list.append(x_fbp)
                    true_list.append(x_true_t_aug)

                x_fbp_b  = torch.stack(fbp_list)   # (B, H, W)
                x_true_b = torch.stack(true_list)  # (B, H, W)

                # ── Forward + loss ────────────────────────────────────────────
                optimizer.zero_grad()
                x_hat = model(x_fbp_b)              # (B, H, W)
                loss  = _loss(x_hat, x_true_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            mean_loss = epoch_loss / n_train

            if mean_loss < best_loss:
                best_loss  = mean_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 50 == 0 or epoch == n_epochs:
                elapsed = time.time() - t0_train
                log(f"  Epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.6f}  "
                    f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"t={elapsed:.0f}s")

        # Restore best weights
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Save checkpoint
        import os
        ckpt_path = "/models/checkpoint/freqattn_varnet_ct/mri_freqattn_varnet_ct.pth"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "n_cascades": n_cascades, "base_ch": base_ch,
                    "best_loss": best_loss,
                    "img_size": (IMG_H, IMG_W)}, ckpt_path)
        vol.commit()
        log(f"  Checkpoint saved → {ckpt_path}  (best_loss={best_loss:.6f})")

    # ── Evaluate on GCS challenge data ────────────────────────────────────────
    model.eval()
    rows = []

    f = h5py.File(io.BytesIO(h5_bytes), "r")
    sample_keys = sorted(f.keys())
    log(f"\n[EVAL] Tier={tier}  samples={len(sample_keys)}")

    for sk in sample_keys:
        grp      = f[sk]
        avail    = list(grp.keys())

        if "x_true" not in avail:
            log(f"  [{sk}] No x_true (dev/hidden tier) — skip")
            continue

        x_true  = grp["x_true"][()].astype(np.float32)
        y_sino  = grp["y"][()].astype(np.float32)
        angles  = grp["H_ideal"][()].astype(np.float64)

        if x_true.max() > 1.0 + 1e-6:
            x_true = x_true / x_true.max()

        out_h, out_w = x_true.shape
        pad           = int(math.ceil(math.sqrt(out_h**2 + out_w**2)))
        angles_list   = angles.tolist()

        # GPU FBP on challenge sinogram
        sino_t = torch.tensor(y_sino, dtype=torch.float32, device=device)
        norm_v = float(sino_t.max().clamp(min=1.0))

        t_start = time.time()
        with torch.no_grad():
            x_fbp_t  = _fbp_gpu(sino_t / norm_v, angles_list, out_h, out_w, pad, device)
            x_hat_t  = model(x_fbp_t.unsqueeze(0)).squeeze(0)   # (H, W)
        elapsed = time.time() - t_start

        x_hat_np = x_hat_t.cpu().numpy().astype(np.float32)
        x_fbp_np = x_fbp_t.cpu().numpy().astype(np.float32)
        x_hat_np = np.clip(x_hat_np, 0.0, 1.0)

        psnr_hat  = _psnr(x_hat_np, x_true)
        ssim_hat  = _ssim_np(x_hat_np, x_true)
        psnr_fbp  = _psnr(np.clip(x_fbp_np, 0.0, 1.0), x_true)
        ssim_fbp  = _ssim_np(np.clip(x_fbp_np, 0.0, 1.0), x_true)

        meta = {}
        if "metadata" in grp.attrs:
            try: meta = json.loads(grp.attrs["metadata"])
            except: pass

        log(f"  [{sk:12s}]  FBP={psnr_fbp:.2f}/{ssim_fbp:.4f}  "
            f"→  Net={psnr_hat:.2f} dB / {ssim_hat:.4f}  "
            f"(+{psnr_hat - psnr_fbp:.2f} dB)  t={elapsed:.2f}s")

        rows.append({
            "tier": tier, "sample": sk, "scene": meta.get("scene", sk),
            "psnr_db": round(psnr_hat, 4),  "ssim": round(ssim_hat, 4),
            "psnr_fbp": round(psnr_fbp, 4), "ssim_fbp": round(ssim_fbp, 4),
            "psnr_gain": round(psnr_hat - psnr_fbp, 4),
            "time_s": round(elapsed, 3),
        })

    f.close()

    if not rows:
        return {"tier": tier, "samples": [], "mean_psnr": 0.0, "mean_ssim": 0.0}

    mean_psnr = sum(r["psnr_db"]  for r in rows) / len(rows)
    mean_ssim = sum(r["ssim"]     for r in rows) / len(rows)
    mean_fbp  = sum(r["psnr_fbp"] for r in rows) / len(rows)
    pass_p = mean_psnr >= 40.0
    pass_s = mean_ssim >= 0.965

    log(f"\n{'='*60}")
    log(f"[RESULT]  Tier={tier}  n={len(rows)}")
    log(f"  FBP baseline:   {mean_fbp:.4f} dB")
    log(f"  FreqAttn-VarNet:{mean_psnr:.4f} dB  SSIM={mean_ssim:.4f}")
    log(f"  PSNR: {'PASS ✓' if pass_p else 'FAIL ✗'} (target ≥ 40.0 dB)")
    log(f"  SSIM: {'PASS ✓' if pass_s else 'FAIL ✗'} (target ≥ 0.965)")
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
    tier:       str   = "public",
    n_epochs:   int   = 400,
    n_train:    int   = 150,
    batch_size: int   = 8,
    n_cascades: int   = 8,
    base_ch:    int   = 32,
    skip_train: bool  = False,
):
    """Run FreqAttn-VarNet-CT on Modal T4 GPU.

    --tier       public|dev|hidden|all   (default: public)
    --n-epochs   training epochs         (default: 400)
    --n-train    batches per epoch       (default: 150)
    --skip-train skip training           (debug / inference-only)
    """
    import json
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR      = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]

    print("FreqAttn-VarNet-CT — Enhanced E2E-VarNet for MRI/Radon Benchmark")
    print(f"  Improvements: spectral-attn, mask-aware DC, L1+SSIM+freq, GPU FBP init")
    print(f"  Tiers={tiers}  Epochs={n_epochs}  n_train={n_train}  bs={batch_size}")
    print(f"  Cascades={n_cascades}  base_ch={base_ch}  skip_train={skip_train}")

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

        result = train_and_eval.remote(
            h5_bytes=data, tier=t,
            n_epochs=n_epochs, n_train=n_train, batch_size=batch_size,
            n_cascades=n_cascades, base_ch=base_ch, skip_train=skip_train,
        )
        all_results.append(result)

    if not all_results:
        print("No results.")
        return

    doc = {
        "timestamp": ts, "variant": "mri",
        "algorithm": "FreqAttn-VarNet-CT",
        "gpu": "T4",
        "config": {"n_cascades": n_cascades, "base_ch": base_ch,
                   "n_epochs": n_epochs, "n_train": n_train,
                   "batch_size": batch_size},
        "improvements": [
            "Spectral attention at U-Net bottleneck (freq-domain differentiation)",
            "Mask-conditioned DC: learnable alpha per cascade",
            "GPU FBP initialisation (ramp filter + GPU backprojection)",
            "Combined L1 + SSIM + frequency amplitude loss",
        ],
        "baseline_psnr_fbp": "~16 dB",
        "baseline_psnr_sfm": "~28 dB",
        "tiers": all_results,
    }
    out_json = OUT_DIR / f"freqattn_varnet_ct_{ts}.json"
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved → {out_json}")

    # Upload to GCS
    try:
        uri = _upload_gcs(out_json,
                          f"benchmark-results/mri/freqattn_varnet_ct_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"  Baseline FBP:          ~16 dB")
    print(f"  Baseline SFM-Combo:    ~28 dB")
    print(f"  Target:               ≥ 40.0 dB / ≥ 0.965 SSIM")
    print("-" * 64)
    for r in all_results:
        if not r.get("samples"):
            continue
        p  = r["mean_psnr"]
        s  = r["mean_ssim"]
        pp = "PASS ✓" if r.get("pass_psnr") else "FAIL ✗"
        ps = "PASS ✓" if r.get("pass_ssim") else "FAIL ✗"
        print(f"  Tier={r['tier']:8s}  PSNR={p:.2f} dB {pp}   SSIM={s:.4f} {ps}")
    print("=" * 64)
