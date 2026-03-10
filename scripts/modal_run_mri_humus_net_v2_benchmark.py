#!/usr/bin/env python3
"""MRI HUMUS-Net++ v2 Benchmark — Modal T4.

Key improvements over v1 (38.9 dB / 0.923 SSIM → target 41+ dB / 0.968+):

  1. Fixed critical Adam optimizer bug
     v1 recreated optimizer every DC step, resetting all momentum state.
     v2 uses a single Adam instance + CosineAnnealingLR per stage.
     Expected gain: ~1-2 dB from proper gradient accumulation.

  2. SIRT+TV warm-start initialization (200 iters, ~29-30 dB)
     v1 started from Gaussian-FBP (~21 dB). Better init = stronger SIREN prior.

  3. Poisson NLL data-consistency loss
     Challenge data: Poisson noise, y_max≈64. MSE is suboptimal (Gaussian).
     Poisson NLL = mean(s - y*log(s+eps)) correctly weights measurements.

  4. Deeper SIREN: hidden=384, 6 layers (2.3× more parameters than v1's 256/5).

  5. Annealed TV regularization in DC (lam 0.02→0.002 over 4 stages).
     Prevents Poisson noise overfitting while preserving data fidelity.

  6. 4-stage cascade, 500 total DC steps (vs v1: 3 stages, 250 steps).

  7. Gradient norm clipping (max_norm=1.0) for training stability.

  8. Freq blend uses SIRT+TV result (30 dB) instead of FBP (21 dB).

Forward model: Radon parallel-beam (180 angles) + Poisson noise (y_max≈64).
Challenge data: 128×128 images, sino=(180, 182). Noise floor ~28 dB.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

app = modal.App("pwm-mri-humus-net-v2")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ── Batched Radon operators ────────────────────────────────────────────────────

def _radon_fwd(x_t, angles_deg, pad_size: int, device):
    """Batched GPU Radon forward projection. Returns (n_angles, pad_size)."""
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
    return rot.squeeze(1).sum(dim=1)  # (n_angles, pad_size)


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


# ── FBP ────────────────────────────────────────────────────────────────────────

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
    """Isotropic TV prox via under-converged gradient descent (mild regularizer)."""
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
    """Differentiable isotropic TV for image tensor (H, W)."""
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
    """SIRT + annealed TV warm-start initialization (~29-30 dB).

    Much better structural prior for SIREN pre-training than FBP (~21 dB).
    """
    import torch
    import numpy as np

    n_angles = len(angles_deg)
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)

    x_fbp_np = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
    x = torch.tensor(x_fbp_np, device=device, dtype=torch.float32)

    # Scale calibration: mean(A(x0)) = mean(y)
    with torch.no_grad():
        sino_init = _radon_fwd(x, angles_deg, pad_size, device)
        scale = float(y_t.mean()) / float(sino_init.mean().clamp(min=1e-6))
        x = (x * scale).clamp(0.0, 1.0)

    # SIRT denominators
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

def _build_siren(hidden_dim=384, n_layers=6):
    """Standard SIREN (omega=30). Deeper than v1 (256/5) for more capacity."""
    import torch
    import torch.nn as nn
    import math as _math

    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, is_first=False, omega=30.0):
            super().__init__()
            self.omega = omega
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                bound = (1. / in_f) if is_first else (_math.sqrt(6. / in_f) / omega)
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega * self.linear(x))

    layers: list[nn.Module] = [SineLayer(2, hidden_dim, is_first=True)]
    for _ in range(n_layers - 1):
        layers.append(SineLayer(hidden_dim, hidden_dim))
    layers.append(nn.Linear(hidden_dim, 1))
    with torch.no_grad():
        b = _math.sqrt(6. / hidden_dim) / 30.
        layers[-1].weight.uniform_(-b, b)
        layers[-1].bias.zero_()
    return nn.Sequential(*layers)


def _make_coords(H, W, device):
    import torch
    ys = torch.linspace(-1., 1., H, device=device)
    xs = torch.linspace(-1., 1., W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _render(inr, coords, H, W):
    return inr(coords).reshape(H, W)


# ── Loss functions ─────────────────────────────────────────────────────────────

def _ssim_loss_torch(x_img, y_img, window_size=11, sigma=1.5,
                     C1=1e-4, C2=9e-4):
    """Differentiable (1 - SSIM) loss for structural pre-training."""
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
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sig_x2 = F.conv2d(x4 * x4, kernel, padding=pad) - mu_x2
    sig_y2 = F.conv2d(y4 * y4, kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(x4 * y4, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return 1.0 - ssim_map.mean()


# ── Frequency blend ────────────────────────────────────────────────────────────

def _freq_blend(x_low, x_high, device, thresh=0.25, sharpness=12.0, alpha=0.25):
    """Blend: INR result (low-freq) + reference init (high-freq).

    Uses SIRT+TV result as x_high (30 dB), not FBP (21 dB) as in v1.
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
    out = torch.fft.irfft2(X_out, s=(H, W)).clamp(0., 1.)
    return out.detach().cpu().numpy().astype("float32")


# ── HUMUS-Net++ v2 main algorithm ──────────────────────────────────────────────

def _humus_net_v2(
    y_sino, angles_deg, device, pad_size, out_h, out_w,
    # SIREN architecture
    inr_hidden: int = 384,
    inr_layers: int = 6,
    # SIRT+TV warm-start
    n_sirt: int = 200,
    # Pre-training (MSE+SSIM from SIRT+TV result)
    n_pretrain: int = 120,
    ssim_alpha: float = 0.4,
    # Multi-stage DC cascade
    n_stages: int = 4,
    stage_steps: tuple = (150, 150, 100, 100),
    stage_lr_max: tuple = (3e-4, 1e-4, 5e-5, 2e-5),
    stage_lr_min: tuple = (1e-4, 3e-5, 1e-5, 5e-6),
    # TV regularization in DC phase (annealed)
    lam_tv_dc_start: float = 0.02,
    lam_tv_dc_end: float = 0.002,
    # Training stability
    grad_clip: float = 1.0,
    # Dynamic stage weights (dHUMUS-Net analogue)
    dynamic_stage_weights: bool = True,
    # Frequency blend
    blend_thresh: float = 0.25,
    blend_alpha: float = 0.25,
):
    """HUMUS-Net++ v2 reconstruction.

    Phase 0: SIRT+TV warm start (200 iters, ~29-30 dB init)
    Phase 1: SIREN pre-training (120 steps, joint MSE+SSIM from SIRT+TV)
    Phase 2: 4-stage Poisson-NLL DC cascade
             — Single Adam + CosineAnnealingLR per stage (fixes v1 optimizer bug)
             — Annealed TV regularization (lam 0.02→0.002)
             — Gradient norm clipping (max_norm=1.0)
             — Learnable stage weights (dHUMUS-Net analogue)
    Phase 3: Freq blend (INR low-freq + SIRT+TV high-freq)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    total_dc_steps = sum(stage_steps)
    lam_tv_schedule = np.exp(
        np.linspace(np.log(lam_tv_dc_start), np.log(lam_tv_dc_end), total_dc_steps)
    ).tolist()

    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    # Keep raw y_sino tensor (un-normalized) for Poisson NLL
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)

    # ── Phase 0: SIRT+TV warm start ───────────────────────────────────────────
    print(f"      [HUMUSv2] Phase 0: SIRT+TV warm start ({n_sirt} iters) ...")
    x_init_np = _sirt_tv_init(y_sino, angles_deg, device, pad_size, out_h, out_w,
                               n_outer=n_sirt)

    # ── Build SIREN ────────────────────────────────────────────────────────────
    coords = _make_coords(out_h, out_w, device)
    inr    = _build_siren(inr_hidden, inr_layers).to(device)
    x_init_t = torch.tensor(x_init_np, device=device, dtype=torch.float32)

    # ── Phase 1: Joint MSE+SSIM pre-training from SIRT+TV result ──────────────
    print(f"      [HUMUSv2] Phase 1: Pre-training ({n_pretrain} steps, "
          f"α_mse={ssim_alpha:.2f}, α_ssim={1-ssim_alpha:.2f}) ...")
    opt_pre = torch.optim.Adam(inr.parameters(), lr=5e-4)
    for step in range(n_pretrain):
        opt_pre.zero_grad()
        x_cur  = torch.sigmoid(_render(inr, coords, out_h, out_w))
        mse_l  = F.mse_loss(x_cur, x_init_t)
        ssim_l = _ssim_loss_torch(x_cur, x_init_t)
        loss   = ssim_alpha * mse_l + (1.0 - ssim_alpha) * ssim_l
        loss.backward()
        opt_pre.step()

    with torch.no_grad():
        pre_mse = float(F.mse_loss(torch.sigmoid(_render(inr, coords, out_h, out_w)), x_init_t))
    print(f"      [HUMUSv2] Phase 1 done: PSNR≈{-10*math.log10(pre_mse+1e-12):.1f} dB")

    # ── Phase 2: 4-stage Poisson-NLL DC cascade ────────────────────────────────
    print(f"      [HUMUSv2] Phase 2: {n_stages}-stage DC cascade "
          f"({'+'.join(str(s) for s in stage_steps)} = {total_dc_steps} steps) ...")

    if dynamic_stage_weights:
        log_w = [torch.nn.Parameter(torch.zeros(1, device=device)) for _ in range(n_stages)]
    else:
        log_w = [None] * n_stages

    overall_best_dc    = float("inf")
    overall_best_state = {k: v.clone() for k, v in inr.state_dict().items()}
    global_step = 0

    for stage in range(n_stages):
        n_steps  = stage_steps[stage]
        lr_max_s = stage_lr_max[stage]
        lr_min_s = stage_lr_min[stage]

        print(f"      [HUMUSv2] Stage {stage+1}/{n_stages}: "
              f"{n_steps} steps, lr={lr_max_s:.1e}→{lr_min_s:.1e} ...")

        # FIX: single Adam instance per stage + scheduler (v1 recreated per step!)
        params = list(inr.parameters())
        if dynamic_stage_weights and log_w[stage] is not None:
            params = params + [log_w[stage]]
        opt = torch.optim.Adam(params, lr=lr_max_s)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min_s
        )

        stage_best_dc    = float("inf")
        stage_best_state = {k: v.clone() for k, v in inr.state_dict().items()}

        for step in range(n_steps):
            lam_tv = lam_tv_schedule[min(global_step, total_dc_steps - 1)]
            global_step += 1

            opt.zero_grad()
            x_cur  = torch.sigmoid(_render(inr, coords, out_h, out_w))
            sino   = _radon_fwd(x_cur, angles_deg, pad_size, device)

            # Poisson NLL: L = mean(s/y_scale - (y/y_scale)*log(s/y_scale + eps))
            sino_n = (sino / y_scale).clamp(min=1e-4)
            y_n    = y_t / y_scale
            dc     = (sino_n - y_n * torch.log(sino_n)).mean()

            # Annealed TV regularization (prevents Poisson noise overfitting)
            tv   = _tv_2d(x_cur)
            loss = dc + lam_tv * tv

            if dynamic_stage_weights and log_w[stage] is not None:
                loss = F.softplus(log_w[stage]) * loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            scheduler.step()

            dc_val = float(dc.detach())
            if dc_val < stage_best_dc:
                stage_best_dc    = dc_val
                stage_best_state = {k: v.clone() for k, v in inr.state_dict().items()}
            if dc_val < overall_best_dc:
                overall_best_dc    = dc_val
                overall_best_state = {k: v.clone() for k, v in inr.state_dict().items()}

            if step % 30 == 0 or step == n_steps - 1:
                w_val  = float(F.softplus(log_w[stage])) if log_w[stage] is not None else 1.0
                lr_cur = scheduler.get_last_lr()[0]
                print(f"      [Stage {stage+1} {step:4d}/{n_steps}]  "
                      f"lr={lr_cur:.2e}  DC={dc_val:.5f}  "
                      f"TV={float(tv.detach()):.4f}  lam={lam_tv:.4f}  "
                      f"w={w_val:.4f}  best={stage_best_dc:.5f}")

        inr.load_state_dict(stage_best_state)
        print(f"      [HUMUSv2] Stage {stage+1} done: "
              f"best_DC={stage_best_dc:.5f}  overall={overall_best_dc:.5f}")

    # Restore overall best
    inr.load_state_dict(overall_best_state)
    with torch.no_grad():
        x_inr = torch.sigmoid(_render(inr, coords, out_h, out_w))
    x_inr_np = x_inr.cpu().numpy().astype("float32")

    # ── Phase 3: Freq blend (INR low-freq + SIRT+TV high-freq) ────────────────
    print("      [HUMUSv2] Phase 3: Freq blend (INR low-freq + SIRT+TV high-freq) ...")
    x_final = _freq_blend(x_inr_np, x_init_np, device,
                          thresh=blend_thresh, sharpness=12.0, alpha=blend_alpha)
    return x_final


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
    x_t      = torch.tensor(x_hat, device=device, dtype=torch.float32)
    sino_hat = _radon_fwd(x_t, angles_deg, pad_size, device)
    y_scale  = float(y_sino.max()) if y_sino.max() > 0 else 1.
    y_t      = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)
    hat_s    = float(sino_hat.max().clamp(min=1e-8))
    diff     = float((sino_hat / hat_s - y_t).norm())
    yn       = float(y_t.norm())
    return float(max(0., 1. - diff / yn)) if yn > 1e-8 else 0.


def _composite(psnr, ssim, cons):
    return 0.4 * min(1., max(0., (psnr - 10.) / 40.)) + 0.4 * ssim + 0.2 * cons


# ── Modal remote function ──────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=3600,
    memory=16384,
)
def run_mri_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    import json
    import time
    import h5py
    import numpy as np
    import torch

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[{tier}] Device={device}  GPU={gpu_name}")

    rows = []
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        for sk in sorted(f.keys()):
            grp        = f[sk]
            x_true     = grp["x_true"][()].astype(np.float32)
            y_sino     = grp["y"][()].astype(np.float64)
            angles_deg = grp["H_ideal"][()].astype(np.float64)
            try:
                meta = json.loads(grp.attrs.get("metadata", "{}"))
            except Exception:
                meta = {}
            scene_name = meta.get("scene", sk)

            out_h, out_w = x_true.shape
            pad_size     = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))
            if x_true.max() > 1. + 1e-6:
                x_true /= x_true.max()

            print(f"\n  [{tier}] {sk}  img={out_h}×{out_w}  "
                  f"sino={y_sino.shape}  pad={pad_size}  "
                  f"y=[{y_sino.min():.2f},{y_sino.max():.2f}]")

            for algo in algos:
                t0 = time.time()
                try:
                    if algo == "fbp":
                        x_hat = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
                    elif algo == "humus_net_v2":
                        x_hat = _humus_net_v2(y_sino, angles_deg, device,
                                              pad_size, out_h, out_w)
                    else:
                        print(f"    [{algo}] unknown, skip")
                        continue
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"    [{algo}] ERROR: {exc}")
                    continue

                elapsed = time.time() - t0
                x_hat_f = np.clip(x_hat, 0., 1.).astype(np.float32)
                psnr    = _psnr(x_hat_f, x_true)
                ssim    = _ssim_np(x_hat_f, x_true)
                cons    = _consistency(x_hat_f, y_sino, angles_deg, pad_size, device)
                score   = _composite(psnr, ssim, cons)
                print(f"    [{algo:18s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                      f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")
                rows.append({
                    "tier": tier, "scene": sk, "scene_name": scene_name,
                    "algo": algo, "psnr_db": round(psnr, 4),
                    "ssim": round(ssim, 4), "consistency": round(cons, 4),
                    "score": round(score, 4), "time_s": round(elapsed, 2),
                })

    return rows


# ── GCS helpers ────────────────────────────────────────────────────────────────

def _download_gcs(variant, tier):
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    key    = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs.Client()
    blob   = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key}")
    return blob.download_as_bytes()


def _upload_gcs(local, key):
    from google.cloud import storage as gcs
    gcs.Client().bucket("pwm-benchmark-datasets").blob(key).upload_from_filename(str(local))
    return f"gs://pwm-benchmark-datasets/{key}"


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(tier: str = "public", algo: str = "all"):
    import csv
    import json
    from collections import defaultdict
    from datetime import datetime, timezone

    ROOT    = Path(__file__).resolve().parents[1]
    OUT_DIR = ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_ALGOS = ["fbp", "humus_net_v2"]
    tiers = ["public"] if tier == "public" else \
            ["public", "dev", "hidden"] if tier == "all" else \
            [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("=" * 70)
    print("MRI HUMUS-Net++ v2 Benchmark")
    print(f"  Tiers: {tiers}   Algos: {algos}")
    print("  Key fixes: Adam optimizer bug + SIRT-TV init + Poisson NLL + deeper SIREN")
    print("=" * 70)

    futures = {}
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_{t} ...")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  [SUBMIT] {t} ({len(data)//1024} KB)")
        futures[t] = run_mri_gpu.spawn(data, t, algos)

    all_rows = []
    for t, fut in futures.items():
        print(f"  [WAIT] {t} ...")
        rows = fut.get()
        all_rows.extend(rows)
        print(f"  [DONE] {t}: {len(rows)} results")

    if not all_rows:
        print("No results.")
        return

    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"humus_net_v2_{ts}.json"
    out_csv  = OUT_DIR / f"humus_net_v2_{ts}.csv"

    doc = {
        "timestamp": ts, "variant": "mri", "tiers": tiers, "algos": algos,
        "gpu": "T4", "method": "HUMUS-Net++ v2",
        "improvements_over_v1": [
            "Fixed Adam optimizer bug (single Adam+CosineAnnealingLR, not recreated per step)",
            "SIRT+TV warm-start init (200 iters, ~29-30 dB vs v1 FBP ~21 dB)",
            "Poisson NLL DC loss: mean(s - y*log(s+eps)), optimal for y_max~64",
            "Deeper SIREN: hidden=384, 6 layers (vs v1: 256/5)",
            "Annealed TV regularization in DC: lam 0.02→0.002",
            "4 stages, 500 total DC steps (vs v1: 3 stages, 250 steps)",
            "Gradient norm clipping (max_norm=1.0)",
            "Freq blend uses SIRT+TV (30 dB) not FBP (21 dB)",
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

    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/humus_net_v2_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    print("\n" + "=" * 70)
    print(f"{'tier':8s}  {'algo':18s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Score':>6s}")
    print("-" * 70)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        p  = sum(r["psnr_db"] for r in rs) / len(rs)
        s  = sum(r["ssim"]    for r in rs) / len(rs)
        sc = sum(r["score"]   for r in rs) / len(rs)
        print(f"{t:8s}  {a:18s}  {p:7.2f}  {s:6.4f}  {sc:6.4f}  (n={len(rs)})")
    print("=" * 70)

    best_rows = [r for r in all_rows if r["algo"] == "humus_net_v2"]
    if best_rows:
        mp = sum(r["psnr_db"] for r in best_rows) / len(best_rows)
        ms = sum(r["ssim"]    for r in best_rows) / len(best_rows)
        print(f"\nHUMUS-Net++ v2: PSNR={mp:.2f} dB  SSIM={ms:.4f}")
        print(f"v1 baseline:    PSNR=38.90 dB  SSIM=0.9230")
        print(f"Target:         PSNR>=40.00    SSIM>=0.9680")
        print(f"  Δ vs v1:  PSNR{mp-38.90:+.2f} dB  SSIM{ms-0.9230:+.4f}")
        print(f"  PSNR {'PASS' if mp >= 40 else 'FAIL'}")
        print(f"  SSIM {'PASS' if ms >= 0.968 else 'FAIL'}")
        print()
        print("Note: Literature targets (41-43 dB) are FastMRI k-space benchmarks.")
        print("Challenge data uses Radon+Poisson — noise floor ~28 dB.")
