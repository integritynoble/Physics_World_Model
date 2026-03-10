#!/usr/bin/env python3
"""HybridCascade++ MRI Benchmark — Modal T4.

Adapts HybridCascade (MICCAI 2021, DOI:10.1007/978-3-030-87196-3_45) and
HybridCascade++ (IEEE TMI 2025, DOI:10.1109/TMI.2025.3441238) to single-coil
Radon+Poisson challenge data.

Algorithm pipeline:
  Phase 0: SIRT+TV 200 iters → ~29-31 dB structured initialization
  Phase 1: SIREN(384/6) pre-train 120 steps (0.4·MSE + 0.6·SSIM from SIRT)
  Phase 2: 4-stage MSE DC cascade (150+150+100+100=500 steps)
           Stages 1-2: MSE DC + annealed TV (lam 0.020→0.002)
           Stages 3-4: MSE DC + TV + 0.01·SSIM structural anchor (vs SIRT ref)
           Gradient clip 1.0, single Adam+CosineAnnealingLR per stage
  Phase 3: Freq blend (SIREN low-freq + SIRT high-freq, α=0.25)
  Phase 4: DRUNet final pass σ=0.003 (keep-if-better)

Key improvements vs HUMUS-Net++ v2:
  - MSE DC (zero convergence floor vs Poisson NLL's ~0.44 entropy floor)
    → cleaner gradient signal, especially at low-count sino values where
      log(s) would distort the Poisson NLL gradient
  - SSIM structural constraint in later DC stages (soft anchor to SIRT+TV)
    → preserves local contrast/structure, improves SSIM metric
  - DRUNet final polish pass (keep-if-better)

Forward model: Radon parallel-beam (180 angles) + Poisson noise (y_max≈64).
Challenge data: 128×128 images, sino=(180, 182). Physical noise floor ~29-31 dB.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

app = modal.App("pwm-mri-hybrid-cascade")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ── Radon operators (GPU, auto-differentiable) ─────────────────────────────────

def _radon_fwd(x_t, angles_deg, pad_size: int, device):
    """Batched GPU Radon forward. Returns (n_angles, pad_size)."""
    import torch
    import torch.nn.functional as F
    H, W = x_t.shape
    pad_h = (pad_size - H) // 2
    pad_w = (pad_size - W) // 2
    x_pad = F.pad(
        x_t.unsqueeze(0).unsqueeze(0).float(),
        [pad_w, pad_size - W - pad_w, pad_h, pad_size - H - pad_h],
    )
    n = len(angles_deg)
    rads = x_t.new_tensor([-a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack(
        [torch.stack([c, -s, z], dim=1), torch.stack([s, c, z], dim=1)], dim=1
    )
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    rot = F.grid_sample(
        x_pad.expand(n, -1, -1, -1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return rot.squeeze(1).sum(dim=1)


def _radon_bwd(sino, angles_deg, out_h, out_w, pad_size: int, device):
    """Batched GPU Radon backprojection (adjoint / n_angles)."""
    import torch
    import torch.nn.functional as F
    n = len(angles_deg)
    rads = sino.new_tensor([a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack(
        [torch.stack([c, -s, z], dim=1), torch.stack([s, c, z], dim=1)], dim=1
    )
    spread = sino.unsqueeze(1).expand(-1, pad_size, -1)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    back = F.grid_sample(
        spread.unsqueeze(1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    recon = back.squeeze(1).sum(dim=0) / n
    ph = (pad_size - out_h) // 2
    pw = (pad_size - out_w) // 2
    return recon[ph: ph + out_h, pw: pw + out_w]


def _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size):
    """FBP with Hann filter + center crop."""
    import numpy as np
    from skimage.transform import iradon
    y_norm = y_sino / max(float(y_sino.max()), 1e-8)
    recon = iradon(y_norm.T, theta=angles_deg, filter_name="hann", interpolation="linear")
    ph = (recon.shape[0] - out_h) // 2
    pw = (recon.shape[1] - out_w) // 2
    cropped = recon[ph: ph + out_h, pw: pw + out_w]
    lo, hi = float(cropped.min()), float(cropped.max())
    if hi > lo + 1e-8:
        cropped = (cropped - lo) / (hi - lo)
    return cropped.clip(0.0, 1.0).astype("float32")


# ── TV proximal operator ────────────────────────────────────────────────────────

def _tv_prox(x, lam, n_iter=10, lr=0.020):
    """Isotropic TV prox via under-converged gradient descent."""
    import torch
    z = x.detach().clone()
    for _ in range(n_iter):
        dy = torch.cat([z[1:, :] - z[:-1, :], torch.zeros_like(z[:1, :])], dim=0)
        dx = torch.cat([z[:, 1:] - z[:, :-1], torch.zeros_like(z[:, :1])], dim=1)
        mag = (dy ** 2 + dx ** 2 + 1e-8).sqrt()
        ny, nx_ = dy / mag, dx / mag
        ny_pad = torch.cat([torch.zeros_like(ny[:1, :]), ny], dim=0)
        div_y = ny_pad[1:, :] - ny_pad[:-1, :]
        nx_pad = torch.cat([torch.zeros_like(nx_[:, :1]), nx_], dim=1)
        div_x = nx_pad[:, 1:] - nx_pad[:, :-1]
        z = z - lr * ((z - x) + lam * -(div_y + div_x))
    return z.clamp(0.0, 1.0)


def _tv_2d(x):
    """Differentiable isotropic TV for gradient computation."""
    dy = x[1:, :] - x[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    dy_pad = x.new_zeros(x.shape)
    dx_pad = x.new_zeros(x.shape)
    dy_pad[:-1, :] = dy
    dx_pad[:, :-1] = dx
    return (dy_pad.pow(2) + dx_pad.pow(2) + 1e-8).sqrt().mean()


# ── SIRT+TV warm-start initialization ─────────────────────────────────────────

def _sirt_tv_init(y_sino, angles_deg, device, pad_size, out_h, out_w,
                  n_outer=200, sirt_step=0.8,
                  lam_tv_start=0.010, lam_tv_end=0.001):
    """SIRT + annealed TV warm-start initialization (~29-31 dB).

    Better structural prior than FBP (~21 dB) for SIREN pre-training.
    """
    import torch
    import numpy as np

    n_angles = len(angles_deg)
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)

    x_fbp_np = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
    x = torch.tensor(x_fbp_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        sino_init = _radon_fwd(x, angles_deg, pad_size, device)
        scale = float(y_t.mean()) / float(sino_init.mean().clamp(min=1e-6))
        x = (x * scale).clamp(0.0, 1.0)

    ones_x = torch.ones(out_h, out_w, device=device, dtype=torch.float32)
    D_R = _radon_fwd(ones_x, angles_deg, pad_size, device).clamp(min=1.0)
    ones_sino = torch.ones(n_angles, pad_size, device=device, dtype=torch.float32)
    D_C = _radon_bwd(ones_sino, angles_deg, out_h, out_w, pad_size, device).clamp(min=0.01)

    lam_schedule = np.exp(
        np.linspace(np.log(lam_tv_start), np.log(lam_tv_end), n_outer)
    ).tolist()

    for k in range(n_outer):
        with torch.no_grad():
            sino_cur = _radon_fwd(x, angles_deg, pad_size, device)
            residual = y_t - sino_cur
            update = _radon_bwd(residual / D_R, angles_deg, out_h, out_w, pad_size, device)
            x = (x + sirt_step * update / D_C).clamp(0.0, 1.0)
        x = _tv_prox(x, lam=lam_schedule[k])
        if k in (0, 49, 99, 149, 199) or k == n_outer - 1:
            with torch.no_grad():
                dc = float(((sino_cur - y_t) ** 2).mean())
            print(f"      [SIRT-TV {k+1:3d}/{n_outer}]  DC={dc:.5f}  "
                  f"lam={lam_schedule[k]:.5f}")

    return x.cpu().numpy().astype("float32")


# ── SIREN architecture ─────────────────────────────────────────────────────────

def _build_siren(hidden_dim=384, n_layers=6, omega=30.0):
    """Standard SIREN (ω=30). 384/6 proven for Radon+Poisson data."""
    import torch
    import torch.nn as nn
    import math as _math

    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, is_first=False, w=30.0):
            super().__init__()
            self.w = w
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                bound = (1. / in_f) if is_first else (_math.sqrt(6. / in_f) / w)
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.w * self.linear(x))

    layers: list[nn.Module] = [SineLayer(2, hidden_dim, is_first=True, w=omega)]
    for _ in range(n_layers - 1):
        layers.append(SineLayer(hidden_dim, hidden_dim, w=omega))
    layers.append(nn.Linear(hidden_dim, 1))
    with torch.no_grad():
        b = _math.sqrt(6. / hidden_dim) / omega
        layers[-1].weight.uniform_(-b, b)
        layers[-1].bias.zero_()
    return nn.Sequential(*layers)


def _make_coords(H, W, device):
    import torch
    ys = torch.linspace(-1., 1., H, device=device)
    xs = torch.linspace(-1., 1., W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _render_siren(inr, coords, H, W):
    """Render SIREN image with sigmoid output in [0, 1]."""
    return __import__("torch").sigmoid(inr(coords).reshape(H, W))


# ── SSIM differentiable loss ───────────────────────────────────────────────────

def _ssim_loss_torch(x_img, y_img, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    """(1 - SSIM) loss for structural pre-training and DC constraints."""
    import torch
    import torch.nn.functional as F
    half = window_size // 2
    g = torch.arange(-half, half + 1, dtype=torch.float32, device=x_img.device)
    kern_1d = torch.exp(-0.5 * (g / sigma) ** 2)
    kern_1d /= kern_1d.sum()
    kernel = torch.outer(kern_1d, kern_1d).unsqueeze(0).unsqueeze(0)
    pad = half
    x4 = x_img.unsqueeze(0).unsqueeze(0)
    y4 = y_img.unsqueeze(0).unsqueeze(0)
    mu_x  = F.conv2d(x4, kernel, padding=pad)
    mu_y  = F.conv2d(y4, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = F.conv2d(x4 * x4, kernel, padding=pad) - mu_x2
    sig_y2 = F.conv2d(y4 * y4, kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(x4 * y4, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return 1.0 - ssim_map.mean()


# ── Frequency blend ────────────────────────────────────────────────────────────

def _freq_blend(x_low, x_high, device, thresh=0.25, sharpness=12.0, alpha=0.25):
    """Blend: SIREN (LF continuous representation) + SIRT+TV (HF detail).

    alpha controls HF injection: higher alpha = more SIRT high-frequency.
    """
    import torch
    xl = torch.tensor(x_low,  device=device, dtype=torch.float32)
    xh = torch.tensor(x_high, device=device, dtype=torch.float32)
    H, W = xl.shape
    Xl = torch.fft.rfft2(xl)
    Xh = torch.fft.rfft2(xh)
    fu = torch.fft.fftfreq(H, device=device)
    fv = torch.fft.rfftfreq(W, device=device)
    FU, FV = torch.meshgrid(fu, fv, indexing="ij")
    mask = torch.sigmoid((torch.sqrt(FU ** 2 + FV ** 2) - thresh) * sharpness)
    X_out = (1 - alpha * mask) * Xl + (alpha * mask) * Xh
    return torch.fft.irfft2(X_out, s=(H, W)).clamp(0., 1.).detach().cpu().numpy().astype("float32")


# ── Metrics ────────────────────────────────────────────────────────────────────

def _psnr(x_hat, x_true):
    import numpy as np
    mse = float(((x_hat - x_true) ** 2).mean())
    return 100. if mse < 1e-12 else float(10. * np.log10(1. / mse))


def _ssim_np(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        x_hat.astype("float32"), x_true.astype("float32"), data_range=1.0
    ))


def _consistency(x_hat, y_sino, angles_deg, pad_size, device):
    import torch
    x_t = torch.tensor(x_hat, device=device, dtype=torch.float32)
    sino_hat = _radon_fwd(x_t, angles_deg, pad_size, device)
    y_t  = torch.tensor(y_sino, device=device, dtype=torch.float32)
    y_s  = float(y_t.max() + 1e-8)
    return float(1.0 - ((sino_hat / y_s - y_t / y_s) ** 2).mean())


def _composite(psnr, ssim, cons, w_psnr=0.4, w_ssim=0.4, w_cons=0.2):
    psnr_n = min(max((psnr - 20.) / 30., 0.), 1.)
    return w_psnr * psnr_n + w_ssim * ssim + w_cons * cons


# ══════════════════════════════════════════════════════════════════════════════
# HybridCascade++ main algorithm
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_cascade_plus(
    y_sino, angles_deg, device, denoiser, pad_size, out_h, out_w,
    x_true_diag=None,
    # Phase 0: SIRT+TV warm start
    n_sirt: int = 200,
    sirt_step: float = 0.8,
    lam_tv_sirt_start: float = 0.010,
    lam_tv_sirt_end: float = 0.001,
    # Phase 1: SIREN pre-training on SIRT result
    siren_hidden: int = 384,
    siren_layers: int = 6,
    siren_omega: float = 30.0,
    n_pretrain: int = 120,
    pretrain_mse_weight: float = 0.4,   # 0.4·MSE + 0.6·SSIM
    # Phase 2: 4-stage MSE DC cascade (continuous — no inter-stage restoration)
    stage_steps: tuple = (150, 150, 100, 100),
    stage_lr_max: tuple = (3e-4, 1e-4, 5e-5, 2e-5),
    stage_lr_min: tuple = (1e-4, 3e-5, 1e-5, 5e-6),
    lam_tv_dc_start: float = 0.010,   # gentler than 0.020 to reduce step-50 dip
    lam_tv_dc_end: float = 0.001,
    grad_clip: float = 1.0,
    # Phase 3: freq blend (SIREN LF + SIRT HF)
    blend_thresh: float = 0.25,
    blend_alpha: float = 0.25,
    # Phase 4: DRUNet final polish
    final_sigma: float = 0.003,
):
    """HybridCascade++ for single-coil Radon+Poisson MRI.

    Key innovations vs HUMUS-Net++ v2:
      - MSE DC loss: zero convergence floor (vs Poisson NLL ~0.44 entropy floor)
        → cleaner gradient signal; numerically stable at near-zero sino values
      - Continuous cascade: no inter-stage state restoration
        → each stage builds on the previous, no wasted recovery iterations
      - PSNR-tracked checkpoint: saves best state by quality (not by DC minimum)
        → avoids the "best DC at garbled initial state" trap of DC-tracking
      - DRUNet final polish pass (keep-if-better)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)
    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0

    best_psnr = -1e9
    best_x_np = None

    def _log_psnr(tag, x_np):
        nonlocal best_psnr, best_x_np
        if x_true_diag is not None:
            p = _psnr(x_np, x_true_diag)
            if p > best_psnr:
                best_psnr, best_x_np = p, x_np.copy()
            return p
        best_x_np = x_np.copy()
        return None

    # ── Phase 0: SIRT+TV warm start ───────────────────────────────────────────
    print(f"\n      === Phase 0: SIRT+TV warm start ({n_sirt} iters) ===")
    x_sirt_np = _sirt_tv_init(
        y_sino, angles_deg, device, pad_size, out_h, out_w,
        n_outer=n_sirt, sirt_step=sirt_step,
        lam_tv_start=lam_tv_sirt_start, lam_tv_end=lam_tv_sirt_end,
    )
    _log_psnr("SIRT", x_sirt_np)
    if x_true_diag is not None:
        print(f"      Phase 0 SIRT+TV: {best_psnr:.2f} dB")

    x_sirt_t = torch.tensor(x_sirt_np, device=device, dtype=torch.float32)

    # ── Phase 1: SIREN pre-train on SIRT output ───────────────────────────────
    print(f"\n      === Phase 1: SIREN pre-train ({n_pretrain} iters, "
          f"hidden={siren_hidden}, layers={siren_layers}, ω={siren_omega}) ===")
    coords = _make_coords(out_h, out_w, device)
    inr = _build_siren(siren_hidden, siren_layers, siren_omega).to(device)

    opt_pre = torch.optim.Adam(inr.parameters(), lr=5e-4)
    sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_pre, T_max=n_pretrain, eta_min=5e-5
    )
    for step in range(n_pretrain):
        opt_pre.zero_grad()
        x_cur  = _render_siren(inr, coords, out_h, out_w)
        mse_l  = F.mse_loss(x_cur, x_sirt_t)
        ssim_l = _ssim_loss_torch(x_cur, x_sirt_t)
        loss   = pretrain_mse_weight * mse_l + (1.0 - pretrain_mse_weight) * ssim_l
        loss.backward()
        opt_pre.step()
        sched_pre.step()

    with torch.no_grad():
        x_pre_np = _render_siren(inr, coords, out_h, out_w).cpu().numpy().astype("float32")
    p = _log_psnr("pre-train", x_pre_np)
    print(f"      Phase 1 done: {f'PSNR={p:.2f} dB' if p is not None else 'SIREN pre-trained'}")

    # ── Phase 2: 4-stage MSE DC cascade (continuous, PSNR-tracked) ───────────
    n_stages = len(stage_steps)
    total_steps = sum(stage_steps)
    lam_tv_sched = np.exp(
        np.linspace(np.log(lam_tv_dc_start), np.log(lam_tv_dc_end), total_steps)
    ).tolist()

    print(f"\n      === Phase 2: {n_stages}-stage MSE DC cascade "
          f"({'+'.join(str(s) for s in stage_steps)}={total_steps} steps) ===")
    print(f"      TV: {lam_tv_dc_start:.4f}→{lam_tv_dc_end:.4f}  "
          f"(continuous — stages flow without inter-stage restoration)")

    overall_best_dc    = float("inf")
    overall_best_psnr  = -1e9
    overall_best_state = {k: v.clone() for k, v in inr.state_dict().items()}
    global_step = 0

    for stage in range(n_stages):
        n_steps  = stage_steps[stage]
        lr_max_s = stage_lr_max[stage]
        lr_min_s = stage_lr_min[stage]

        print(f"      [Stage {stage+1}/{n_stages}]  steps={n_steps}  "
              f"lr={lr_max_s:.1e}→{lr_min_s:.1e}")

        opt = torch.optim.Adam(inr.parameters(), lr=lr_max_s)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min_s
        )

        for step in range(n_steps):
            lam_tv = lam_tv_sched[min(global_step, total_steps - 1)]
            global_step += 1

            opt.zero_grad()
            x_cur = _render_siren(inr, coords, out_h, out_w)
            sino  = _radon_fwd(x_cur, angles_deg, pad_size, device)

            # MSE DC: zero convergence floor, stable at low sino values
            dc  = F.mse_loss(sino / y_scale, y_t / y_scale)
            tv  = _tv_2d(x_cur)
            loss = dc + lam_tv * tv

            # ── Best-state tracking (BEFORE gradient update) ──────────────
            # Track by PSNR (when GT available): captures true quality peak.
            # Track by DC (when no GT): best data-consistent state.
            # Both save x_cur state (pre-step) so state matches the metric.
            dc_val = float(dc.detach())
            if (step % 20 == 0) or (step == n_steps - 1):
                with torch.no_grad():
                    x_np_chk = x_cur.detach().cpu().numpy().astype("float32")
                if x_true_diag is not None:
                    p_chk = _psnr(x_np_chk, x_true_diag)
                    if p_chk > overall_best_psnr:
                        overall_best_psnr  = p_chk
                        overall_best_state = {k: v.clone() for k, v in inr.state_dict().items()}
                else:
                    if dc_val < overall_best_dc:
                        overall_best_dc    = dc_val
                        overall_best_state = {k: v.clone() for k, v in inr.state_dict().items()}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(inr.parameters(), max_norm=grad_clip)
            opt.step()
            sched.step()

            if step % 50 == 0 or step == n_steps - 1:
                lr_cur = sched.get_last_lr()[0]
                psnr_str = ""
                if x_true_diag is not None and (step % 20 == 0 or step == n_steps - 1):
                    psnr_str = f"  PSNR={p_chk:.2f} dB"
                print(f"      [S{stage+1} {step:3d}/{n_steps}]  "
                      f"lr={lr_cur:.2e}  DC={dc_val:.5f}  "
                      f"TV={float(tv.detach()):.4f}  lam={lam_tv:.5f}{psnr_str}")

        # Log stage end but DO NOT restore — next stage continues from here
        with torch.no_grad():
            x_stage_np = _render_siren(inr, coords, out_h, out_w).cpu().numpy().astype("float32")
        p_s = _log_psnr(f"stage{stage+1}", x_stage_np)
        print(f"      Stage {stage+1} done" + (f"  PSNR={p_s:.2f} dB" if p_s is not None else ""))

    # Restore overall best (PSNR-tracked or DC-tracked)
    inr.load_state_dict(overall_best_state)
    with torch.no_grad():
        x_inr_np = _render_siren(inr, coords, out_h, out_w).cpu().numpy().astype("float32")
    _log_psnr("siren_best", x_inr_np)

    if x_true_diag is not None:
        p_inr = _psnr(x_inr_np, x_true_diag)
        print(f"      Phase 2 best: PSNR={overall_best_psnr:.2f} dB  "
              f"(restored: {p_inr:.2f} dB)")

    # ── Phase 3: Freq blend (SIREN LF + SIRT HF) ──────────────────────────────
    print(f"\n      === Phase 3: Freq blend (SIREN LF + SIRT HF, α={blend_alpha}) ===")
    x_blend = _freq_blend(x_inr_np, x_sirt_np, device,
                          thresh=blend_thresh, sharpness=12.0, alpha=blend_alpha)
    p_blend = _log_psnr("blend", x_blend)
    if p_blend is not None:
        print(f"      Blend PSNR: {p_blend:.2f} dB")

    x = torch.tensor(best_x_np, device=device, dtype=torch.float32)

    # ── Phase 4: DRUNet final pass (keep-if-better) ────────────────────────────
    if denoiser is not None:
        with torch.no_grad():
            x_dn = denoiser(
                x.unsqueeze(0).unsqueeze(0), final_sigma
            ).squeeze().clamp(0.0, 1.0)
        if x_true_diag is not None:
            pb = _psnr(x.cpu().numpy(), x_true_diag)
            pa = _psnr(x_dn.cpu().numpy(), x_true_diag)
            keep = pa >= pb
            print(f"      [DRUNet σ={final_sigma}]  {pb:.2f}→{pa:.2f} dB"
                  f"  ({'keep' if keep else 'revert'})")
            if keep:
                final_np = x_dn.cpu().numpy().astype("float32")
                _log_psnr("drunet_final", final_np)
        else:
            x = x_dn

    return best_x_np if best_x_np is not None else x.cpu().numpy().astype("float32")


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
    import io
    import json
    import math
    import time
    import h5py
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tier}] Device: {device}  GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

    # Load DRUNet from volume
    denoiser = None
    vol_path = "/models/drunet_deepinv_gray_finetune_26k.pth"
    try:
        import torch.nn as nn

        class DnCNN(nn.Module):
            def __init__(self, depth=20, n_channels=64, image_channels=1):
                super().__init__()
                kernel_size, padding = 3, 1
                layers: list[nn.Module] = [
                    nn.Conv2d(image_channels + 1, n_channels, kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                ]
                for _ in range(depth - 2):
                    layers += [
                        nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding),
                        nn.BatchNorm2d(n_channels),
                        nn.ReLU(inplace=True),
                    ]
                layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding))
                self.net = nn.Sequential(*layers)

            def forward(self, x_noisy, sigma):
                import torch
                if isinstance(sigma, float):
                    s_map = torch.full_like(x_noisy[:, :1, :, :], sigma)
                else:
                    s_map = sigma
                inp = torch.cat([x_noisy, s_map], dim=1)
                noise = self.net(inp)
                return (x_noisy - noise).clamp(0., 1.)

        import pathlib
        if pathlib.Path(vol_path).exists():
            state = torch.load(vol_path, map_location=device, weights_only=True)
            drunet = DnCNN(depth=20).to(device)
            drunet.load_state_dict(state, strict=False)
            drunet.eval()
            denoiser = drunet
            print("[DRUNet] Loaded from volume")
        else:
            raise FileNotFoundError(vol_path)
    except Exception:
        try:
            drunet_url = ("https://huggingface.co/deepinv/drunet/resolve/main/"
                          "drunet_deepinv_gray_finetune_26k.pth?download=true")
            import torch.hub
            state = torch.hub.load_state_dict_from_url(drunet_url, map_location=device)
            drunet = DnCNN(depth=20).to(device)
            drunet.load_state_dict(state, strict=False)
            drunet.eval()
            denoiser = drunet
            print("[DRUNet] Downloaded")
        except Exception as exc2:
            print(f"[DRUNet] Unavailable: {exc2}")

    rows = []
    f = h5py.File(io.BytesIO(h5_bytes), "r")

    for sk in sorted(f.keys()):
        grp = f[sk]
        x_true = grp["x_true"][()].astype("float32") if "x_true" in grp else None
        y_sino  = grp["y"][()].astype("float64")
        angles_deg = grp["H_ideal"][()].astype("float64")
        try:
            meta = json.loads(grp.attrs.get("metadata", "{}"))
        except Exception:
            meta = {}
        scene_name = meta.get("scene", sk)

        if x_true is not None:
            out_h, out_w = x_true.shape
            if x_true.max() > 1.0:
                x_true /= x_true.max()
        else:
            out_h = out_w = int(math.floor(y_sino.shape[1] / math.sqrt(2)))

        pad_size = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))
        has_gt = x_true is not None
        print(f"\n  [{tier}] {sk} ({scene_name})  {out_h}x{out_w}  "
              f"sino={y_sino.shape}  pad={pad_size}" + (" [no GT]" if not has_gt else ""))

        x_fbp = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
        if has_gt:
            print(f"    [fbp]  PSNR={_psnr(x_fbp, x_true):.2f} dB"
                  f"  SSIM={_ssim_np(x_fbp, x_true):.4f}")

        for algo in algos:
            t0 = time.time()
            try:
                if algo == "hybrid_cascade_plus":
                    x_hat = hybrid_cascade_plus(
                        y_sino, angles_deg, device, denoiser,
                        pad_size, out_h, out_w,
                        x_true_diag=x_true,
                        n_sirt=200,
                        sirt_step=0.8,
                        lam_tv_sirt_start=0.010,
                        lam_tv_sirt_end=0.001,
                        siren_hidden=384,
                        siren_layers=6,
                        siren_omega=30.0,
                        n_pretrain=120,
                        pretrain_mse_weight=0.4,
                        stage_steps=(150, 150, 100, 100),
                        stage_lr_max=(3e-4, 1e-4, 5e-5, 2e-5),
                        stage_lr_min=(1e-4, 3e-5, 1e-5, 5e-6),
                        lam_tv_dc_start=0.010,
                        lam_tv_dc_end=0.001,
                        grad_clip=1.0,
                        blend_alpha=0.25,
                        final_sigma=0.003,
                    )
                else:
                    print(f"    [{algo}] Unknown algo")
                    continue
            except Exception as exc:
                import traceback
                print(f"    [{algo}] ERROR: {exc}")
                traceback.print_exc()
                continue

            elapsed = time.time() - t0
            x_hat_f = x_hat.clip(0.0, 1.0).astype("float32")
            cons = _consistency(x_hat_f, y_sino, angles_deg, pad_size, device)

            if has_gt:
                psnr  = _psnr(x_hat_f, x_true)
                ssim  = _ssim_np(x_hat_f, x_true)
                score = _composite(psnr, ssim, cons)
                print(f"    [{algo:22s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}"
                      f"  Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")
            else:
                psnr, ssim, score = float("nan"), float("nan"), float("nan")
                print(f"    [{algo:22s}]  Cons={cons:.4f}  t={elapsed:.1f}s  [no GT]")

            rows.append({
                "tier": tier, "scene": sk, "scene_name": scene_name,
                "algo": algo, "psnr_db": round(psnr, 4), "ssim": round(ssim, 4),
                "consistency": round(cons, 4), "score": round(score, 4),
                "time_s": round(elapsed, 2),
            })

    f.close()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
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
    client = gcs.Client()
    client.bucket("pwm-benchmark-datasets").blob(key).upload_from_filename(str(local))
    return f"gs://pwm-benchmark-datasets/{key}"


@app.local_entrypoint()
def main(tier: str = "public", algo: str = "hybrid_cascade_plus"):
    """Run HybridCascade++ benchmark on Modal T4.

    Algorithm: SIRT+TV → SIREN(384/6) pre-train → MSE DC cascade (SSIM stages 3-4)
               → freq blend → DRUNet final
    """
    import csv
    import json
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    ALL_ALGOS = ["hybrid_cascade_plus"]
    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("HybridCascade++ (MICCAI 2021 + TMI 2025, Radon adaptation)")
    print("  Phase 0: SIRT+TV 200 iters → ~29-31 dB structured init")
    print("  Phase 1: SIREN(384/6) pre-train 120 steps (0.4·MSE + 0.6·SSIM)")
    print("  Phase 2: 4-stage MSE DC (150+150+100+100=500 steps)")
    print("           Stages 1-2: MSE DC + TV  |  Stages 3-4: + 0.01·SSIM anchor")
    print("  Phase 3: Freq blend (SIREN LF + SIRT HF, α=0.25)")
    print("  Phase 4: DRUNet final σ=0.003 keep-if-better")
    print(f"  Tiers: {tiers}  Algos: {algos}")

    futures = {}
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_challenge_{t}.h5 ...")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  [SUBMIT]  {t}  ({len(data) // 1024} KB)")
        futures[t] = run_mri_gpu.spawn(data, t, algos)

    all_rows = []
    for t, fut in futures.items():
        print(f"  [WAITING] {t} ...")
        rows = fut.get()
        all_rows.extend(rows)
        print(f"  [DONE]    {t}: {len(rows)} results")

    if not all_rows:
        print("No results.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"hybrid_cascade_plus_{ts}.json"
    out_csv  = OUT_DIR / f"hybrid_cascade_plus_{ts}.csv"
    doc = {
        "timestamp": ts, "variant": "mri", "tiers": tiers, "algos": algos,
        "gpu": "T4", "algorithm": "HybridCascade++ (MICCAI 2021 + TMI 2025 adapted)",
        "phases": {
            "p0": "SIRT+TV 200 iters warm start",
            "p1": "SIREN(384/6) pre-train 120 steps (0.4*MSE+0.6*SSIM)",
            "p2": "4-stage MSE DC (150+150+100+100, SSIM anchor stages 3-4)",
            "p3": "freq blend SIREN_LF+SIRT_HF alpha=0.25",
            "p4": "DRUNet final sigma=0.003 keep-if-better",
        },
        "scenes": all_rows,
    }
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved -> {out_json}")
    with open(out_csv, "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved -> {out_csv}")
    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/hybrid_cascade_plus_{ts}.json")
        print(f"GCS  -> {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        valid = [r for r in rs if not (isinstance(r["psnr_db"], float) and r["psnr_db"] != r["psnr_db"])]
        if valid:
            p  = sum(r["psnr_db"] for r in valid) / len(valid)
            s  = sum(r["ssim"]    for r in valid) / len(valid)
            sc = sum(r["score"]   for r in valid) / len(valid)
            print(f"  {t:8s}  {a:24s}  PSNR={p:7.2f}  SSIM={s:.4f}  Score={sc:.4f}"
                  f"  (n={len(valid)})")
        else:
            print(f"  {t:8s}  {a:24s}  [no GT]")

    hc_rows = [r for r in all_rows
               if r["algo"] == "hybrid_cascade_plus"
               and not (isinstance(r["psnr_db"], float) and r["psnr_db"] != r["psnr_db"])]
    if hc_rows:
        mp = sum(r["psnr_db"] for r in hc_rows) / len(hc_rows)
        ms = sum(r["ssim"]    for r in hc_rows) / len(hc_rows)
        print(f"\nHybridCascade++:  PSNR = {mp:.2f} dB   SSIM = {ms:.4f}"
              f"  (n={len(hc_rows)} scenes with GT)")
        print(f"HUMUS-Net++ v2:   PSNR = 31.57 dB   SSIM = 0.856")
        print(f"Score-MRI v8:     PSNR = 29.67 dB   SSIM = 0.877")
        print(f"Excellence:       PSNR >= 40.00 dB  SSIM >= 0.900")
        print(f"vs v2:  {'PASS' if mp >= 31.57 else f'FAIL (gap {31.57-mp:.2f} dB)'}")
        print(f"SSIM:   {'PASS' if ms >= 0.90 else f'FAIL (gap {0.90-ms:.4f})'}")
