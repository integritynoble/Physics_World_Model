#!/usr/bin/env python3
"""PromptMR-SFM: Spatial-Frequency Joint MRI Reconstruction — Modal T4/A10 GPU.

Implements PromptMR++ combining four improvements over the broken Score-MRI-INR baseline:

  Phase 1 — Filtered Back-Projection (FBP):
    Fast initialization using analytical Radon inversion with Hamming filter.
    Provides a physics-consistent starting point in < 0.5 s.

  Phase 2 — PnP-GD with DRUNet (Spatial Branch):
    Plug-and-Play gradient descent alternating between:
      (a) Radon data-consistency gradient step  [fixes geometry/physics]
      (b) DRUNet MMSE denoising                 [removes noise/artifacts]
    Geometric sigma schedule: σ_max → σ_min over N_pnp iterations.
    Inspired by the spatial-domain Mamba branch of MMR-Mamba (MIA 2025).
    Reference: Chan et al., SIAM J. Imaging Sci. 2017.

  Phase 3 — Frequency-Domain Amplitude Refinement (Frequency Branch):
    Split FFT of PnP result and FBP into amplitude + phase.
    For high-frequency components (|ω| > threshold), blend FBP amplitude
    (phase-accurate, detail-rich) into PnP phase (artifact-free, smooth).
    Inspired by the frequency-domain branch of MMR-Mamba (2025).
    Reference: Zhao et al., Med. Image Anal. 2025.

  Phase 4 — INR Finalization with Corrected DC Loss (Perceptual Branch):
    Parameterize image through SIREN implicit neural representation.
    KEY BUG FIX: DC loss computed on x_cur (INR render) NOT x_hat (detached),
    ensuring ∇_θ L_DC flows back through the INR parameters.
    Multi-term loss: λ_DC·||A(x_cur)/s - y_t||² + λ_SSIM·(1-SSIM(x_cur,x_hat))
                   + λ_LPIPS·LPIPS(x_cur, x_hat)
    Reference: MR-IPT (Sci. Reports 2025), Score-MRI (Chung & Ye, MedIA 2022).

Dataset format (MRI challenge HDF5):
    y       : (180, N_det) parallel-beam sinogram (rotation-based Radon)
    H_ideal : (180,)       projection angles in degrees [0..179]
    x_true  : (128, 128)   ground-truth phantom in [0, 1]

Usage:
    modal run scripts/modal_run_mri_promptsfm_benchmark.py
    modal run scripts/modal_run_mri_promptsfm_benchmark.py --tier public
    modal run scripts/modal_run_mri_promptsfm_benchmark.py --algo fbp,pnp_drunet,prompt_mri_sfm
    modal run scripts/modal_run_mri_promptsfm_benchmark.py --tier all --algo all
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

# ── Modal infrastructure ──────────────────────────────────────────────────────

app = modal.App("pwm-mri-promptsfm-benchmark")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "h5py",
        "scikit-image",
        "deepinv",
        "lpips",
        "Pillow",
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SIREN Implicit Neural Representation
# ══════════════════════════════════════════════════════════════════════════════


def _build_siren(hidden_dim: int = 256, n_layers: int = 5):
    """Build a SIREN MLP: (x, y) → pixel value.

    SIREN (Sitzmann et al., NeurIPS 2020) uses sinusoidal activations.
    Larger hidden_dim and n_layers than baseline for higher capacity.
    """
    import torch
    import torch.nn as nn
    import math as _math

    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, is_first=False, omega=30.0):
            super().__init__()
            self.omega = omega
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                if is_first:
                    bound = 1.0 / in_f
                else:
                    bound = _math.sqrt(6.0 / in_f) / omega
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega * self.linear(x))

    layers: list[nn.Module] = [SineLayer(2, hidden_dim, is_first=True)]
    for _ in range(n_layers - 1):
        layers.append(SineLayer(hidden_dim, hidden_dim))
    layers.append(nn.Linear(hidden_dim, 1))
    with torch.no_grad():
        bound = _math.sqrt(6.0 / hidden_dim) / 30.0
        layers[-1].weight.uniform_(-bound, bound)
        layers[-1].bias.zero_()

    return nn.Sequential(*layers)


def _make_coord_grid(H: int, W: int, device) -> "torch.Tensor":
    """Normalized coordinate grid in [-1, 1]^2, shape (H*W, 2)."""
    import torch
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def _inr_render(inr, coords: "torch.Tensor", H: int, W: int) -> "torch.Tensor":
    """Rasterize INR to (H, W) tensor (no activation — sigmoid applied outside)."""
    return inr(coords).reshape(H, W)


# ══════════════════════════════════════════════════════════════════════════════
# GPU Radon operators (rotation-based, matches challenge generator exactly)
# ══════════════════════════════════════════════════════════════════════════════


def _radon_fwd(x_t, angles_deg, pad_size: int, device):
    """GPU parallel-beam Radon forward.

    Matches _forward_radon_fast in generate_challenge_datasets.py:
        sino[i] = rotate(padded_x, -θ_i).sum(axis=0)
    """
    import torch
    import torch.nn.functional as F

    H, W = x_t.shape
    pad_h = (pad_size - H) // 2
    pad_w = (pad_size - W) // 2
    x_pad = F.pad(
        x_t.unsqueeze(0).unsqueeze(0).float(),
        [pad_w, pad_size - W - pad_w, pad_h, pad_size - H - pad_h],
    )

    n_angles = len(angles_deg)
    sino = torch.zeros(n_angles, pad_size, device=device, dtype=torch.float32)

    for i, angle in enumerate(angles_deg):
        rad = float(-angle * math.pi / 180.0)
        c, s = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]],
                              device=device, dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), x_pad.shape, align_corners=True)
        rot = F.grid_sample(x_pad, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        sino[i] = rot.squeeze().sum(dim=0)
    return sino


def _radon_bwd(sino, angles_deg, out_h: int, out_w: int, pad_size: int, device):
    """GPU Radon back-projection (adjoint of _radon_fwd)."""
    import torch
    import torch.nn.functional as F

    n_angles = len(angles_deg)
    recon = torch.zeros(pad_size, pad_size, device=device, dtype=torch.float32)

    for i, angle in enumerate(angles_deg):
        rad = float(angle * math.pi / 180.0)
        c, s = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]],
                              device=device, dtype=torch.float32)
        spread = sino[i].unsqueeze(0).expand(pad_size, -1)
        grid = F.affine_grid(
            theta.unsqueeze(0), (1, 1, pad_size, pad_size), align_corners=True
        )
        back = F.grid_sample(
            spread.unsqueeze(0).unsqueeze(0), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        recon += back.squeeze()

    recon /= n_angles
    ph = (pad_size - out_h) // 2
    pw = (pad_size - out_w) // 2
    return recon[ph: ph + out_h, pw: pw + out_w]


# ══════════════════════════════════════════════════════════════════════════════
# SSIM differentiable loss
# ══════════════════════════════════════════════════════════════════════════════


def _ssim_loss(x: "torch.Tensor", y: "torch.Tensor",
               window_size: int = 11,
               C1: float = 0.01 ** 2,
               C2: float = 0.03 ** 2) -> "torch.Tensor":
    """Differentiable SSIM loss = 1 - SSIM(x, y). Inputs: (1,1,H,W) in [0,1]."""
    import torch
    import torch.nn.functional as F

    coords = torch.arange(window_size, device=x.device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    ssim_map = (
        (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    )
    return 1.0 - ssim_map.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: FBP baseline
# ══════════════════════════════════════════════════════════════════════════════


def _fbp_recon(y_sino, angles_deg, out_h: int, out_w: int):
    """Filtered back-projection via skimage iradon (Hamming filter)."""
    import numpy as np
    from skimage.transform import iradon
    from PIL import Image as PILImage

    y_norm = y_sino / max(float(y_sino.max()), 1e-8)
    recon = iradon(y_norm.T, theta=angles_deg,
                   filter_name="hamming", interpolation="linear")
    if recon.shape != (out_h, out_w):
        img = PILImage.fromarray(np.clip(recon, 0, None).astype(np.float32))
        img = img.resize((out_w, out_h), PILImage.BILINEAR)
        recon = np.array(img)
    lo, hi = float(recon.min()), float(recon.max())
    if hi > lo + 1e-8:
        recon = (recon - lo) / (hi - lo)
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: PnP-GD with DRUNet (spatial-domain branch)
# ══════════════════════════════════════════════════════════════════════════════


def _pnp_gd_phase(
    x_init,
    y_sino,
    angles_deg,
    denoiser,
    device,
    pad_size: int,
    out_h: int,
    out_w: int,
    n_iter: int = 40,
    lr: float = 0.04,
    sigma_start: float = 0.15,
    sigma_end: float = 0.008,
):
    """PnP gradient descent: Radon DC step + DRUNet denoising.

    Uses a geometric sigma schedule for DRUNet (coarse → fine denoising).
    Each iteration:
        x ← x - lr * A^T(A(x)/s - y_t)   [Radon data-consistency gradient]
        x ← DRUNet(x, σ_i)                [plug-and-play denoising]
        x ← clip(x, 0, 1)
    """
    import torch
    import torch.nn.functional as F

    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)

    x = torch.tensor(x_init, device=device, dtype=torch.float32)

    # Geometric sigma schedule: exponential decay
    log_s = math.log(sigma_start)
    log_e = math.log(sigma_end)

    for i in range(n_iter):
        frac = i / max(n_iter - 1, 1)
        sigma = math.exp(log_s + (log_e - log_s) * frac)

        # --- Radon data-consistency gradient step ---
        x_g = x.detach().requires_grad_(True)
        sino_hat = _radon_fwd(x_g, angles_deg, pad_size, device)
        hat_scale = sino_hat.detach().max().clamp(min=1e-8)
        dc_loss = F.mse_loss(sino_hat / hat_scale, y_t)
        dc_loss.backward()
        x = (x - lr * x_g.grad.detach()).clamp(0.0, 1.0)

        # --- DRUNet denoising (plug-and-play) ---
        with torch.no_grad():
            x_4d = x.unsqueeze(0).unsqueeze(0)
            x = denoiser(x_4d, sigma).squeeze().clamp(0.0, 1.0)

        if i % 10 == 0 or i == n_iter - 1:
            with torch.no_grad():
                sino_check = _radon_fwd(x, angles_deg, pad_size, device)
                dc_v = float(F.mse_loss(sino_check / sino_check.max().clamp(1e-8), y_t))
            print(f"      [PnP-GD {i:3d}/{n_iter}]  σ={sigma:.4f}  DC={dc_v:.5f}")

    return x.detach().cpu().numpy().astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Frequency-Domain Amplitude Refinement (frequency branch)
# ══════════════════════════════════════════════════════════════════════════════


def _freq_domain_refine(
    x_pnp,
    x_fbp,
    device,
    blend_freq_thresh: float = 0.35,
    blend_sharpness: float = 15.0,
    global_blend: float = 0.25,
):
    """Frequency-domain amplitude refinement.

    For high-frequency components (|ω| > threshold), blend FBP amplitude
    (phase-accurate, rich in fine structure) into PnP phase (smooth, artifact-free).
    This recovers fine structural details that DRUNet may have over-smoothed.

    Inspired by the frequency-domain branch of MMR-Mamba (Zhao et al., MIA 2025).

    Args:
        x_pnp: PnP-GD result, numpy (H, W), values in [0, 1]
        x_fbp: FBP result, numpy (H, W), values in [0, 1]
        blend_freq_thresh: normalized frequency threshold [0, 0.5] for blend
        blend_sharpness: sigmoid sharpness of the blend mask
        global_blend: final spatial blend weight (0→keep PnP, 1→freq-refined)

    Returns:
        Refined numpy array (H, W) in [0, 1]
    """
    import torch

    H, W = x_pnp.shape
    x_p = torch.tensor(x_pnp, device=device, dtype=torch.float32)
    x_f = torch.tensor(x_fbp, device=device, dtype=torch.float32)

    # 2D FFT (rfft2 for efficiency)
    X_p = torch.fft.rfft2(x_p)   # (H, W//2+1) complex
    X_f = torch.fft.rfft2(x_f)   # (H, W//2+1) complex

    # Build normalized frequency grid
    freq_u = torch.fft.fftfreq(H, device=device)        # (H,) in [-0.5, 0.5)
    freq_v = torch.fft.rfftfreq(W, device=device)       # (W//2+1,) in [0, 0.5]
    FU, FV = torch.meshgrid(freq_u, freq_v, indexing="ij")
    freq_r = (FU ** 2 + FV ** 2).sqrt()                  # (H, W//2+1)

    # Smooth high-frequency blend mask: 0 (low-freq → keep PnP) → 1 (high-freq → blend FBP)
    alpha = torch.sigmoid((freq_r - blend_freq_thresh) * blend_sharpness)

    # Blend amplitudes: low-freq from PnP, high-freq blend in FBP amplitude
    amp_p = X_p.abs()
    amp_f = X_f.abs()
    amp_blend = (1.0 - alpha) * amp_p + alpha * amp_f

    # Reconstruct with PnP phase but blended amplitude
    phase_p = torch.angle(X_p)
    X_refined = torch.polar(amp_blend, phase_p)
    x_refined = torch.fft.irfft2(X_refined, s=(H, W)).clamp(0.0, 1.0)

    # Soft spatial blend: mostly trust PnP, use freq-refined for detail correction
    x_out = (1.0 - global_blend) * x_p + global_blend * x_refined
    return x_out.clamp(0.0, 1.0).detach().cpu().numpy().astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: INR Finalization with Corrected DC Loss
# ══════════════════════════════════════════════════════════════════════════════


def _inr_finalize_phase(
    x_init,
    y_sino,
    angles_deg,
    denoiser,
    lpips_fn,
    device,
    pad_size: int,
    out_h: int,
    out_w: int,
    # INR architecture (larger than baseline)
    inr_hidden: int = 256,
    inr_layers: int = 5,
    # Pretrain
    inr_pretrain_steps: int = 100,
    inr_pretrain_lr: float = 5e-4,
    # Diffusion loop
    n_steps: int = 120,
    sigma_max: float = 0.12,
    sigma_min: float = 0.002,
    inr_steps_per_diffusion: int = 8,
    inr_lr: float = 2e-4,
    # Loss weights
    lam_dc: float = 1.0,
    lam_ssim: float = 0.08,
    lam_perc: float = 0.06,
):
    """INR fine-tuning with CORRECTED data-consistency gradient.

    BUG FIX vs baseline score_mri_inr:
        Original: dc_loss = ||A(x_hat)/s - y_t||^2  (x_hat is DETACHED — no grad!)
        Fixed:    dc_loss = ||A(x_cur)/s - y_t||^2  (x_cur flows gradient to INR)

    The INR now correctly enforces Radon data-consistency, not just perceptual
    similarity to a noisy denoiser output.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    coords = _make_coord_grid(out_h, out_w, device)

    # Build and pretrain SIREN from x_init
    inr = _build_siren(inr_hidden, inr_layers).to(device)
    opt_pre = torch.optim.Adam(inr.parameters(), lr=inr_pretrain_lr)
    x_init_t = torch.tensor(x_init, device=device, dtype=torch.float32)

    for step in range(inr_pretrain_steps):
        opt_pre.zero_grad()
        x_inr = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))
        F.mse_loss(x_inr, x_init_t).backward()
        opt_pre.step()

    with torch.no_grad():
        x_check = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))
        pre_mse = float(F.mse_loss(x_check, x_init_t))
    print(f"      [INR pretrain] MSE={pre_mse:.5f}  "
          f"PSNR≈{float(-10 * math.log10(pre_mse + 1e-12)):.1f} dB")

    # Fixed y normalization (not adaptive per-step for stability)
    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)

    # Cosine sigma schedule
    def _sigma(i: int) -> float:
        frac = i / max(n_steps - 1, 1)
        return sigma_min + 0.5 * (sigma_max - sigma_min) * (1.0 + math.cos(math.pi * frac))

    for step_i in range(n_steps):
        sigma = _sigma(step_i)

        # Current INR render (detached copy for denoiser input)
        with torch.no_grad():
            x_render = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))

        # DRUNet score estimate (no gradient through denoiser)
        with torch.no_grad():
            inp = x_render.unsqueeze(0).unsqueeze(0)
            inp_noisy = (inp + sigma * 0.25 * torch.randn_like(inp)).clamp(0.0, 1.0)
            x_hat = denoiser(inp_noisy, sigma).squeeze().clamp(0.0, 1.0)

        # Adaptive learning rate: larger sigma → larger lr (coarse adjustment)
        step_lr = inr_lr * (sigma / sigma_max)
        opt = torch.optim.Adam(inr.parameters(), lr=step_lr)

        for _ in range(inr_steps_per_diffusion):
            opt.zero_grad()

            # x_cur: current INR render WITH gradient (flows to INR params)
            x_cur = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))

            # --- Data consistency on x_cur (CORRECTED — gradient flows to INR) ---
            sino_cur = _radon_fwd(x_cur, angles_deg, pad_size, device)
            cur_scale = sino_cur.detach().max().clamp(min=1e-8)
            dc_loss = F.mse_loss(sino_cur / cur_scale, y_t)

            # --- SSIM structural loss ---
            x_cur_4d = x_cur.unsqueeze(0).unsqueeze(0)
            x_hat_4d = x_hat.unsqueeze(0).unsqueeze(0).detach()
            ssim_l = _ssim_loss(x_cur_4d, x_hat_4d)

            # --- LPIPS perceptual loss (VGG) ---
            x_cur_3c = x_cur_4d.expand(-1, 3, -1, -1) * 2.0 - 1.0
            x_hat_3c = x_hat_4d.expand(-1, 3, -1, -1) * 2.0 - 1.0
            perc_l = lpips_fn(x_cur_3c, x_hat_3c).mean()

            total_loss = (lam_dc * dc_loss
                          + lam_ssim * ssim_l
                          + lam_perc * perc_l)
            total_loss.backward()
            opt.step()

        if step_i % 30 == 0 or step_i == n_steps - 1:
            with torch.no_grad():
                x_check = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))
            print(f"      [INR {step_i:3d}/{n_steps}]  σ={sigma:.4f}  "
                  f"DC={float(dc_loss):.5f}  SSIM_l={float(ssim_l):.4f}  "
                  f"LPIPS={float(perc_l):.4f}")

    with torch.no_grad():
        x_final = torch.sigmoid(_inr_render(inr, coords, out_h, out_w))
    return x_final.cpu().numpy().astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# PromptMR-SFM: orchestrates all four phases
# ══════════════════════════════════════════════════════════════════════════════


def prompt_mri_sfm(
    y_sino,
    angles_deg,
    device,
    denoiser,
    lpips_fn,
    pad_size: int,
    out_h: int,
    out_w: int,
):
    """PromptMR++ Spatial-Frequency Joint MRI Reconstruction.

    Four-phase pipeline:
        1. FBP                   (~0.3s)  — physics-consistent initialization
        2. PnP-GD / DRUNet       (~15s)   — spatial-domain artifact removal
        3. Freq-domain refinement (~0.1s) — high-freq detail recovery
        4. INR finalization       (~50s)  — perceptual + DC fine-tuning
    Total: ~65s per sample on T4.
    """
    import numpy as np

    print("      [Phase 1] FBP initialization ...")
    x_fbp = _fbp_recon(y_sino, angles_deg, out_h, out_w)
    print(f"      [Phase 1] FBP range=[{x_fbp.min():.3f}, {x_fbp.max():.3f}]")

    print("      [Phase 2] PnP-GD with DRUNet (40 iterations) ...")
    x_pnp = _pnp_gd_phase(
        x_fbp, y_sino, angles_deg, denoiser, device,
        pad_size, out_h, out_w,
        n_iter=40, lr=0.04,
        sigma_start=0.15, sigma_end=0.008,
    )
    print(f"      [Phase 2] PnP range=[{x_pnp.min():.3f}, {x_pnp.max():.3f}]")

    print("      [Phase 3] Frequency-domain amplitude refinement ...")
    x_freq = _freq_domain_refine(
        x_pnp, x_fbp, device,
        blend_freq_thresh=0.35,
        blend_sharpness=15.0,
        global_blend=0.25,
    )
    print(f"      [Phase 3] Freq-refined range=[{x_freq.min():.3f}, {x_freq.max():.3f}]")

    print("      [Phase 4] INR finalization (120 steps × 8 inner) ...")
    x_final = _inr_finalize_phase(
        x_freq, y_sino, angles_deg, denoiser, lpips_fn,
        device, pad_size, out_h, out_w,
        inr_hidden=256, inr_layers=5,
        inr_pretrain_steps=100, inr_pretrain_lr=5e-4,
        n_steps=120, sigma_max=0.12, sigma_min=0.002,
        inr_steps_per_diffusion=8, inr_lr=2e-4,
        lam_dc=1.0, lam_ssim=0.08, lam_perc=0.06,
    )

    return x_final


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════


def _psnr(x_hat, x_true):
    import numpy as np
    mse = float(((x_hat - x_true) ** 2).mean())
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _ssim_np(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        x_hat.astype("float32"), x_true.astype("float32"), data_range=1.0
    ))


def _consistency(x_hat, y_sino, angles_deg, pad_size: int, device):
    """Radon data-consistency score in [0, 1]."""
    import torch
    x_t = torch.tensor(x_hat, device=device, dtype=torch.float32)
    sino_hat = _radon_fwd(x_t, angles_deg, pad_size, device)
    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)
    hat_scale = float(sino_hat.max()) if float(sino_hat.max()) > 0 else 1.0
    diff = float((sino_hat / hat_scale - y_t).norm())
    y_norm = float(y_t.norm())
    return float(max(0.0, 1.0 - diff / y_norm)) if y_norm > 1e-8 else 0.0


def _composite(psnr: float, ssim: float, cons: float) -> float:
    return 0.4 * min(1.0, max(0.0, (psnr - 10.0) / 40.0)) + 0.4 * ssim + 0.2 * cons


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=3600,
    memory=16384,
)
def run_mri_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    """Run MRI reconstruction algorithms on T4 GPU."""
    import json
    import time
    import h5py
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[{tier}] Device: {device}  GPU: {gpu_name}")

    # Load DRUNet denoiser (shared across algorithms)
    denoiser = None
    need_denoiser = any(a in algos for a in ("pnp_drunet", "prompt_mri_sfm"))
    if need_denoiser:
        try:
            import deepinv as dinv
            path = "/models/checkpoint/DRUNet/drunet_deepinv_gray_finetune_26k.pth"
            denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, nb=4).to(device)
            ckpt = torch.load(path, map_location=device, weights_only=False)
            denoiser.load_state_dict(ckpt)
            denoiser.eval()
            print("[DRUNet] Loaded from /models")
        except Exception as exc:
            print(f"[DRUNet] FAILED: {exc}")

    # Load LPIPS (VGG perceptual loss)
    lpips_fn = None
    if "prompt_mri_sfm" in algos:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device)
            lpips_fn.eval()
            print("[LPIPS] Loaded (VGG)")
        except Exception as exc:
            print(f"[LPIPS] FAILED: {exc}")

    rows = []
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        for sk in sorted(f.keys()):
            grp = f[sk]
            x_true = grp["x_true"][()].astype(np.float32)
            y_sino = grp["y"][()].astype(np.float64)
            angles_deg = grp["H_ideal"][()].astype(np.float64)

            try:
                meta = json.loads(grp.attrs.get("metadata", "{}"))
            except Exception:
                meta = {}
            scene_name = meta.get("scene", sk)

            out_h, out_w = x_true.shape
            pad_size = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))

            if x_true.max() > 1.0 + 1e-6:
                x_true /= x_true.max()

            print(f"\n  [{tier}] {sk} ({scene_name})  "
                  f"img={out_h}×{out_w}  sino={y_sino.shape}  pad={pad_size}")

            for algo in algos:
                t0 = time.time()
                try:
                    if algo == "fbp":
                        x_hat = _fbp_recon(y_sino, angles_deg, out_h, out_w)

                    elif algo == "pnp_drunet":
                        if denoiser is None:
                            print(f"    [{algo}] SKIP — denoiser not loaded")
                            continue
                        x_fbp = _fbp_recon(y_sino, angles_deg, out_h, out_w)
                        x_hat = _pnp_gd_phase(
                            x_fbp, y_sino, angles_deg, denoiser, device,
                            pad_size, out_h, out_w,
                            n_iter=40, lr=0.04,
                            sigma_start=0.15, sigma_end=0.008,
                        )

                    elif algo == "prompt_mri_sfm":
                        if denoiser is None or lpips_fn is None:
                            print(f"    [{algo}] SKIP — denoiser or LPIPS not loaded")
                            continue
                        x_hat = prompt_mri_sfm(
                            y_sino, angles_deg, device, denoiser, lpips_fn,
                            pad_size, out_h, out_w,
                        )

                    else:
                        print(f"    [{algo}] Unknown, skipping")
                        continue

                except Exception as exc:
                    import traceback
                    print(f"    [{algo}] ERROR: {exc}")
                    traceback.print_exc()
                    continue

                elapsed = time.time() - t0
                x_hat_f = np.clip(x_hat, 0.0, 1.0).astype(np.float32)

                psnr = _psnr(x_hat_f, x_true)
                ssim = _ssim_np(x_hat_f, x_true)
                cons = _consistency(x_hat_f, y_sino, angles_deg, pad_size, device)
                score = _composite(psnr, ssim, cons)

                print(f"    [{algo:18s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                      f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")

                rows.append({
                    "tier": tier,
                    "scene": sk,
                    "scene_name": scene_name,
                    "algo": algo,
                    "psnr_db": round(psnr, 4),
                    "ssim": round(ssim, 4),
                    "consistency": round(cons, 4),
                    "score": round(score, 4),
                    "time_s": round(elapsed, 2),
                })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Local helpers: GCS download / upload
# ══════════════════════════════════════════════════════════════════════════════


def _download_gcs(variant: str, tier: str) -> bytes:
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    key = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs.Client()
    blob = client.bucket(bucket).blob(key)
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
def main(tier: str = "public", algo: str = "all"):
    """Run PromptMR-SFM MRI benchmark on Modal T4.

    --tier  public|dev|hidden|all   (default: public)
    --algo  fbp|pnp_drunet|prompt_mri_sfm|all   (default: all)
    """
    import csv
    import json
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    ALL_ALGOS = ["fbp", "pnp_drunet", "prompt_mri_sfm"]

    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("=" * 72)
    print("PromptMR-SFM: Spatial-Frequency Joint MRI Reconstruction")
    print("  Phase 1: FBP initialization")
    print("  Phase 2: PnP-GD with DRUNet (spatial branch)")
    print("  Phase 3: Frequency-domain amplitude refinement")
    print("  Phase 4: INR finalization with corrected DC loss")
    print(f"  Tiers : {tiers}")
    print(f"  Algos : {algos}")
    print("=" * 72)

    futures = {}
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_challenge_{t}.h5 ...")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  [SUBMIT]   {t}  ({len(data) // 1024} KB)")
        futures[t] = run_mri_gpu.spawn(data, t, algos)

    all_rows = []
    for t, fut in futures.items():
        print(f"  [WAITING]  {t} ...")
        rows = fut.get()
        all_rows.extend(rows)
        print(f"  [DONE]     {t}: {len(rows)} results")

    if not all_rows:
        print("No results collected.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"prompt_mri_sfm_{ts}.json"
    out_csv  = OUT_DIR / f"prompt_mri_sfm_{ts}.csv"

    doc = {
        "timestamp": ts,
        "variant": "mri",
        "tiers": tiers,
        "algos": algos,
        "gpu": "T4",
        "algorithm": "PromptMR-SFM",
        "improvements": [
            "Phase 1: FBP initialization",
            "Phase 2: PnP-GD DRUNet (spatial branch)",
            "Phase 3: Frequency-domain amplitude refinement",
            "Phase 4: INR + corrected DC gradient + LPIPS + SSIM",
        ],
        "bug_fixes": [
            "DC loss now computed on x_cur (INR render) not x_hat (detached)",
            "Gradient correctly flows from DC loss through INR parameters",
        ],
        "scenes": all_rows,
    }
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved → {out_json}")

    with open(out_csv, "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved → {out_csv}")

    # Upload result to GCS
    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/prompt_mri_sfm_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload failed: {e}")

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY — mean metrics per (tier, algo)")
    print("=" * 72)
    print(f"{'tier':8s}  {'algo':20s}  {'PSNR':>7s}  {'SSIM':>6s}  "
          f"{'Cons':>6s}  {'Score':>6s}  N")
    print("-" * 72)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        p  = sum(r["psnr_db"]    for r in rs) / len(rs)
        s  = sum(r["ssim"]       for r in rs) / len(rs)
        c  = sum(r["consistency"] for r in rs) / len(rs)
        sc = sum(r["score"]      for r in rs) / len(rs)
        print(f"{t:8s}  {a:20s}  {p:7.2f}  {s:6.4f}  {c:6.4f}  {sc:6.4f}  {len(rs)}")
    print("=" * 72)

    # Quality assessment vs targets
    sfm_rows = [r for r in all_rows if r["algo"] == "prompt_mri_sfm"]
    if sfm_rows:
        mp = sum(r["psnr_db"] for r in sfm_rows) / len(sfm_rows)
        ms = sum(r["ssim"]    for r in sfm_rows) / len(sfm_rows)
        print(f"\nPromptMR-SFM:   PSNR = {mp:.2f} dB   SSIM = {ms:.4f}")
        print(f"Excellent target: PSNR ≥ 40.50 dB   SSIM ≥ 0.9700")
        psnr_ok = mp >= 40.5
        ssim_ok = ms >= 0.97
        print(f"  PSNR {'PASS' if psnr_ok else 'FAIL'}   SSIM {'PASS' if ssim_ok else 'FAIL'}")
        if not (psnr_ok and ssim_ok):
            print("\n  [INFO] If targets not met, consider:")
            print("    - Increase n_iter in PnP phase (40 → 60)")
            print("    - Increase n_steps in INR phase (120 → 200)")
            print("    - Reduce lr in PnP phase (0.04 → 0.02) for stability")
            print("    - Upgrade to A10 GPU for larger SIREN (hidden_dim=512)")
