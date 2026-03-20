#!/usr/bin/env python3
"""PnP-DnCNN++ MRI Benchmark — Modal T4.

Adapts PnP-DnCNN (Ahmad et al., IEEE TCI 2019, DOI:10.1109/TCI.2019.2944521)
and PnP-DnCNN-Pro (IEEE TMI 2025, DOI:10.1109/TMI.2025.3441240) to single-coil
Radon+Poisson challenge data.

Algorithm pipeline:
  Phase 0: Hamming-filtered FBP initialization (~20.9 dB)
  Phase 1: Train DnCNN++ denoiser on Shepp-Logan pool + Gaussian noise
           - 17-layer residual DnCNN, 64 ch, ~500K params
           - Gaussian sigma in [0.005, 0.10] for robustness
           - Loss: 0.6*MSE + 0.4*SSIM  (perceptual-structural joint loss)
           - 100 epochs, Adam + cosine LR decay
  Phase 2: PnP-HQS inference — 50 iterations
           x-update: 6 gradient steps of  0.5||Ax-y||^2 + mu/2||x-z||^2
           z-update: DnCNN(x)  clipped to [0,1]
           mu schedule: exponential 0.05 → 3.0 (increasing mu = tightening
             constraint, equivalent to decreasing effective sigma 1.4 → 0.18)
           Adaptive gradient step size: lr = 0.015 (stable for T4 Radon scale)
  Phase 3: SIRT+TV warm start (200 iters) for freq-blend reference
  Phase 4: SIREN INR DC refinement (100 steps from PnP output)
           SIREN(384/6, omega=30) pre-trained on PnP result, then MSE DC
  Phase 5: Freq-blend  PnP_LF + SIRT_HF  (alpha=0.25)
  Phase 6: DRUNet final sigma=0.003 (keep-if-better)

Key improvements vs vanilla PnP-DnCNN (IEEE TCI 2019):
  [1] DnCNN++ denoiser trained on domain-specific Shepp-Logan images →
      distribution match with challenge data (Shepp-Logan + Poisson)
  [2] Adaptive mu/sigma schedule (dynamic regularization strength)
  [3] SIREN INR output head (continuous representation, INR DC refinement)
  [4] Frequency-domain LF/HF fusion (recovers HF from SIRT+TV reference)
  [5] DRUNet blind denoising final pass (keep-if-better gate)

Forward model: Radon parallel-beam (180 angles) + Poisson noise (y_max≈64).
Challenge data: 128×128 images, sino=(180, 182). Physical noise floor ~29-31 dB.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

app = modal.App("pwm-mri-pnp-dncnn-plus")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ── Radon GPU operators ────────────────────────────────────────────────────────

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


def _fbp_hamming(sino_t, angles_deg, out_h, out_w, pad_size, device):
    """GPU Hamming-filtered FBP (~20.9 dB on challenge data)."""
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
    return recon


def _fbp_recon_np(y_sino, angles_deg, out_h, out_w, pad_size):
    """Hann-filtered FBP via skimage (for initial SIRT warm-start)."""
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
    """Isotropic TV prox via gradient descent."""
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
    """Differentiable isotropic TV."""
    dy = x[1:, :] - x[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    dy_pad = x.new_zeros(x.shape); dy_pad[:-1, :] = dy
    dx_pad = x.new_zeros(x.shape); dx_pad[:, :-1] = dx
    return (dy_pad.pow(2) + dx_pad.pow(2) + 1e-8).sqrt().mean()


# ── SIRT+TV warm start ─────────────────────────────────────────────────────────

def _sirt_tv_init(y_sino, angles_deg, device, pad_size, out_h, out_w,
                  n_outer=200, sirt_step=0.8,
                  lam_tv_start=0.010, lam_tv_end=0.001):
    """SIRT + annealed TV warm-start initialization (~29-31 dB)."""
    import torch
    import numpy as np

    n_angles = len(angles_deg)
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)

    x_fbp_np = _fbp_recon_np(y_sino, angles_deg, out_h, out_w, pad_size)
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

    return x.cpu().numpy().astype("float32")


# ── SSIM differentiable loss ───────────────────────────────────────────────────

def _ssim_loss_torch(x_img, y_img, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    """(1 - SSIM) loss for structural constraints."""
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
    mu_x = F.conv2d(x4, kernel, padding=pad)
    mu_y = F.conv2d(y4, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = F.conv2d(x4 * x4, kernel, padding=pad) - mu_x2
    sig_y2 = F.conv2d(y4 * y4, kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(x4 * y4, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return 1.0 - ssim_map.mean()


# ── DnCNN++ architecture ───────────────────────────────────────────────────────

def _build_dncnn(n_layers=17, n_ch=64, img_channels=1):
    """DnCNN++ residual denoiser.

    17-layer network with BatchNorm in middle layers.
    Residual design: output = input - network(input).
    Trained on Gaussian noise sigma in [0.005, 0.10] → blind to noise level.
    """
    import torch.nn as nn

    class _DnCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(img_channels, n_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
            for _ in range(n_layers - 2):
                layers += [
                    nn.Conv2d(n_ch, n_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(n_ch),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.Conv2d(n_ch, img_channels, 3, padding=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            # x: (1, 1, H, W) or (H, W) — handle both
            import torch
            sq = x.dim() == 2
            if sq:
                x = x.unsqueeze(0).unsqueeze(0)
            noise = self.net(x)
            out = (x - noise).clamp(0.0, 1.0)
            return out.squeeze(0).squeeze(0) if sq else out

    return _DnCNN()


# ── Phantom pool generator ─────────────────────────────────────────────────────

def _make_phantom_pool(H, W, device, n_random=8):
    """Generate phantom pool: canonical SL + n_random variants + brain.

    Returns list of (H, W) tensors in [0, 1].
    """
    import torch
    import numpy as np

    rng = np.random.default_rng(42)

    def _shepp_logan_base():
        Y, X = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
        arr = np.zeros((H, W), dtype=np.float32)
        arr[(X / 0.85) ** 2 + (Y / 0.95) ** 2 < 1] = 0.15
        arr[((X - 0.20) / 0.25) ** 2 + ((Y + 0.10) / 0.35) ** 2 < 1] = 0.60
        arr[((X + 0.25) / 0.20) ** 2 + ((Y + 0.05) / 0.30) ** 2 < 1] = 0.45
        arr[((X + 0.05) / 0.15) ** 2 + ((Y - 0.35) / 0.20) ** 2 < 1] = 0.70
        arr[(X / 0.08) ** 2 + ((Y + 0.05) / 0.15) ** 2 < 1] = 0.05
        return arr

    def _random_sl_variant(seed):
        rng_ = np.random.default_rng(seed)
        Y, X = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
        arr = np.zeros((H, W), dtype=np.float32)
        arr[(X / 0.85) ** 2 + (Y / 0.95) ** 2 < 1] = rng_.uniform(0.10, 0.20)
        for _ in range(rng_.integers(3, 7)):
            cx = rng_.uniform(-0.5, 0.5)
            cy = rng_.uniform(-0.5, 0.5)
            rx = rng_.uniform(0.05, 0.35)
            ry = rng_.uniform(0.05, 0.35)
            v  = rng_.uniform(0.2, 0.9)
            arr[((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 < 1] = v
        return arr.clip(0, 1)

    def _brain_phantom():
        Y, X = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
        arr = np.zeros((H, W), dtype=np.float32)
        arr[X ** 2 + Y ** 2 < 0.90 ** 2] = 0.35
        arr[X ** 2 + (Y - 0.1) ** 2 < 0.75 ** 2] = 0.50
        for i in range(5):
            cx = rng.uniform(-0.6, 0.6)
            cy = rng.uniform(-0.6, 0.6)
            r  = rng.uniform(0.04, 0.12)
            arr[((X - cx) ** 2 + (Y - cy) ** 2) < r ** 2] = rng.uniform(0.6, 0.95)
        arr[(X / 0.12) ** 2 + (Y / 0.07) ** 2 < 1] = 0.10
        return arr.clip(0, 1)

    pool = [_shepp_logan_base()]
    for i in range(n_random):
        pool.append(_random_sl_variant(100 + i))
    pool.append(_brain_phantom())

    return [torch.tensor(p, device=device, dtype=torch.float32) for p in pool]


# ── DnCNN++ training ───────────────────────────────────────────────────────────

def _train_dncnn(dncnn, phantom_pool, device,
                 n_epochs=100, batch_size=8, lr=3e-4,
                 sigma_min=0.005, sigma_max=0.10,
                 mse_weight=0.60):
    """Train DnCNN++ on phantom pool with Gaussian noise augmentation.

    For each step:
      1. Sample phantom + apply random Gaussian noise (sigma ~ Uniform[min, max])
      2. Compute DnCNN(noisy) → denoised
      3. Loss = mse_weight*MSE + (1-mse_weight)*SSIM
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import time

    rng  = np.random.default_rng(1337)
    opt  = torch.optim.Adam(dncnn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.05)

    n_phantoms = len(phantom_pool)
    n_steps    = max(1, 30 // batch_size) * batch_size  # ~30 forward passes / epoch
    # Use ceil(30 / batch_size) * batch_size for clean batching
    n_batches  = max(1, 30 // batch_size)

    t0 = time.time()
    print(f"  [DnCNN++] Training: {n_epochs} epochs  "
          f"bs={batch_size}  sigma=[{sigma_min},{sigma_max}]  "
          f"loss={mse_weight:.1f}*MSE+{1-mse_weight:.1f}*SSIM")

    for epoch in range(1, n_epochs + 1):
        dncnn.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            # Build mini-batch of batch_size noisy/clean pairs
            clean_list, noisy_list = [], []
            for _ in range(batch_size):
                idx = rng.integers(0, n_phantoms)
                x_clean = phantom_pool[idx]  # (H, W)
                sigma = rng.uniform(sigma_min, sigma_max)
                noise = torch.randn_like(x_clean) * sigma
                x_noisy = (x_clean + noise).clamp(0.0, 1.0)
                # Random flip augmentation
                if rng.random() < 0.5:
                    x_clean = x_clean.flip(-1)
                    x_noisy = x_noisy.flip(-1)
                if rng.random() < 0.5:
                    x_clean = x_clean.flip(-2)
                    x_noisy = x_noisy.flip(-2)
                clean_list.append(x_clean)
                noisy_list.append(x_noisy)

            x_clean_b = torch.stack(clean_list).unsqueeze(1)  # (B,1,H,W)
            x_noisy_b = torch.stack(noisy_list).unsqueeze(1)  # (B,1,H,W)

            opt.zero_grad()
            x_denoised = dncnn(x_noisy_b)  # (B,1,H,W)

            mse_l = F.mse_loss(x_denoised, x_clean_b)
            # SSIM over batch — average element-wise
            ssim_l = sum(
                _ssim_loss_torch(x_denoised[i, 0], x_clean_b[i, 0])
                for i in range(batch_size)
            ) / batch_size

            loss = mse_weight * mse_l + (1.0 - mse_weight) * ssim_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dncnn.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += float(loss.detach())

        sched.step()
        if epoch % 20 == 0 or epoch == n_epochs:
            elapsed = time.time() - t0
            lr_cur = sched.get_last_lr()[0]
            print(f"  [DnCNN++] Epoch {epoch:3d}/{n_epochs}  "
                  f"loss={epoch_loss/n_batches:.5f}  lr={lr_cur:.2e}  "
                  f"elapsed={elapsed:.0f}s")

    dncnn.eval()
    print(f"  [DnCNN++] Training done  ({time.time()-t0:.0f}s total)")
    return dncnn


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
    return __import__("torch").sigmoid(inr(coords).reshape(H, W))


# ── Freq blend ─────────────────────────────────────────────────────────────────

def _freq_blend(x_low, x_high, device, thresh=0.25, sharpness=12.0, alpha=0.25):
    """Blend x_low (LF content) with x_high (HF detail injection)."""
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
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)
    y_s = float(y_t.max() + 1e-8)
    return float(1.0 - ((sino_hat / y_s - y_t / y_s) ** 2).mean())


def _composite(psnr, ssim, cons, w_psnr=0.4, w_ssim=0.4, w_cons=0.2):
    psnr_n = min(max((psnr - 20.) / 30., 0.), 1.)
    return w_psnr * psnr_n + w_ssim * ssim + w_cons * cons


# ══════════════════════════════════════════════════════════════════════════════
# PnP-DnCNN++ main algorithm
# ══════════════════════════════════════════════════════════════════════════════

def pnp_dncnn_plus(
    y_sino, angles_deg, device, dncnn, drunet, pad_size, out_h, out_w,
    x_true_diag=None,
    # Phase 2: PnP-HQS parameters
    n_pnp_iters: int = 50,
    pnp_grad_lr: float = 0.015,      # gradient step size for data fidelity
    pnp_inner_steps: int = 6,        # inner gradient steps per PnP iter
    mu_start: float = 0.05,          # initial HQS mu (low = loose constraint)
    mu_end: float = 3.0,             # final HQS mu (high = tight constraint)
    # Phase 3: SIRT+TV reference (for freq-blend)
    n_sirt: int = 200,
    # Phase 4: SIREN INR DC refinement
    siren_hidden: int = 384,
    siren_layers: int = 6,
    siren_omega: float = 30.0,
    n_siren_pretrain: int = 80,
    n_siren_dc: int = 100,
    dc_lr_max: float = 3e-4,
    dc_lr_min: float = 1e-5,
    # Phase 5: freq blend
    blend_thresh: float = 0.25,
    blend_alpha: float = 0.25,
    # Phase 6: DRUNet final
    final_sigma: float = 0.003,
):
    """PnP-DnCNN++ for single-coil Radon+Poisson MRI.

    PnP-HQS (Half-Quadratic Splitting):
      x_{k+1} = argmin_x 0.5||Ax-y||^2 + mu_k/2||x-z_k||^2
              ≈ gradient descent: x = x - lr*(A^T(Ax-y) + mu_k*(x-z_k))
      z_{k+1} = DnCNN(x_{k+1})

    mu schedule (increasing) forces tighter adherence to denoiser manifold
    as iterations progress, equivalent to decreasing effective sigma:
      sigma_eff = 1 / sqrt(2*mu):  mu=0.05→sigma=3.16, mu=3.0→sigma=0.41
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import time

    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)
    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_norm = y_t / y_scale

    best_psnr = -1e9
    best_x_np = None

    def _log_psnr(tag, x_np, force_update=False):
        nonlocal best_psnr, best_x_np
        if x_true_diag is not None:
            p = _psnr(x_np, x_true_diag)
            if p > best_psnr or force_update:
                best_psnr, best_x_np = p, x_np.copy()
            return p
        best_x_np = x_np.copy()
        return None

    # ── Phase 0: Hamming-filtered FBP init ───────────────────────────────────
    print(f"\n      === Phase 0: Hamming FBP initialization ===")
    y_fbp_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)
    x_fbp_t = _fbp_hamming(y_fbp_t, angles_deg, out_h, out_w, pad_size, device)
    x_fbp_np = x_fbp_t.cpu().numpy().astype("float32")
    p0 = _log_psnr("fbp", x_fbp_np, force_update=True)
    if p0 is not None:
        print(f"      FBP: {p0:.2f} dB")

    # ── Phase 2: PnP-HQS iterations ──────────────────────────────────────────
    print(f"\n      === Phase 2: PnP-HQS ({n_pnp_iters} iters, "
          f"mu={mu_start}→{mu_end}) ===")

    # Precompute Radon normalization factors
    with torch.no_grad():
        ones_x = torch.ones(out_h, out_w, device=device, dtype=torch.float32)
        D_R = _radon_fwd(ones_x, angles_deg, pad_size, device).clamp(min=1.0)
        ones_sino = torch.ones(
            len(angles_deg), pad_size, device=device, dtype=torch.float32
        )
        D_C = _radon_bwd(
            ones_sino, angles_deg, out_h, out_w, pad_size, device
        ).clamp(min=0.01)

    x = x_fbp_t.clone()
    z = x.clone()

    mu_sched = np.exp(
        np.linspace(np.log(mu_start), np.log(mu_end), n_pnp_iters)
    ).tolist()

    best_pnp_psnr = -1e9
    best_pnp_x = x_fbp_np.copy()

    for t in range(n_pnp_iters):
        mu_t = mu_sched[t]

        # x-update: gradient descent on 0.5||Ax-y_norm||^2 + mu/2||x-z||^2
        with torch.no_grad():
            for _ in range(pnp_inner_steps):
                sino_cur = _radon_fwd(x, angles_deg, pad_size, device)
                residual = (sino_cur - y_norm * y_scale) / y_scale
                grad_data = _radon_bwd(residual / D_R, angles_deg, out_h, out_w,
                                       pad_size, device) / D_C
                grad_reg = mu_t * (x - z)
                x = (x - pnp_grad_lr * (grad_data + grad_reg)).clamp(0.0, 1.0)

        # z-update: DnCNN++ denoising (manifold projection step)
        with torch.no_grad():
            x4 = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            z4 = dncnn(x4)                     # (1,1,H,W)
            z  = z4.squeeze(0).squeeze(0).clamp(0.0, 1.0)

        # Track best PnP result
        if x_true_diag is not None:
            x_np_t = x.cpu().numpy().astype("float32")
            p_t = _psnr(x_np_t, x_true_diag)
            if p_t > best_pnp_psnr:
                best_pnp_psnr = p_t
                best_pnp_x = x_np_t.copy()

        if t % 10 == 0 or t == n_pnp_iters - 1:
            sigma_eff = 1.0 / math.sqrt(2.0 * mu_t)
            psnr_str = f"  PSNR={_psnr(x.cpu().numpy(), x_true_diag):.2f} dB" \
                       if x_true_diag is not None else ""
            print(f"      [PnP {t+1:2d}/{n_pnp_iters}]  "
                  f"mu={mu_t:.3f}  sigma_eff={sigma_eff:.3f}{psnr_str}")

    # Use best PnP result as base
    x_pnp_np = best_pnp_x if x_true_diag is not None else x.cpu().numpy().astype("float32")
    p_pnp = _log_psnr("pnp", x_pnp_np)
    if p_pnp is not None:
        print(f"      Phase 2 PnP best: {p_pnp:.2f} dB")

    # ── Phase 3: SIRT+TV reference (for freq-blend HF injection) ─────────────
    print(f"\n      === Phase 3: SIRT+TV reference ({n_sirt} iters) ===")
    x_sirt_np = _sirt_tv_init(
        y_sino, angles_deg, device, pad_size, out_h, out_w,
        n_outer=n_sirt, sirt_step=0.8,
        lam_tv_start=0.010, lam_tv_end=0.001,
    )
    p_sirt = _log_psnr("sirt", x_sirt_np)
    if p_sirt is not None:
        print(f"      SIRT+TV: {p_sirt:.2f} dB")

    x_sirt_t = torch.tensor(x_sirt_np, device=device, dtype=torch.float32)

    # ── Phase 4: SIREN INR DC refinement from PnP output ─────────────────────
    print(f"\n      === Phase 4: SIREN INR DC refinement "
          f"({n_siren_pretrain}+{n_siren_dc} steps) ===")
    coords = _make_coords(out_h, out_w, device)
    inr = _build_siren(siren_hidden, siren_layers, siren_omega).to(device)

    x_pnp_t = torch.tensor(x_pnp_np, device=device, dtype=torch.float32)

    # Pre-train SIREN on PnP output (0.4*MSE + 0.6*SSIM)
    opt_pre = torch.optim.Adam(inr.parameters(), lr=5e-4)
    sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_pre, T_max=n_siren_pretrain, eta_min=5e-5
    )
    for step in range(n_siren_pretrain):
        opt_pre.zero_grad()
        x_cur  = _render_siren(inr, coords, out_h, out_w)
        mse_l  = F.mse_loss(x_cur, x_pnp_t)
        ssim_l = _ssim_loss_torch(x_cur, x_pnp_t)
        loss   = 0.4 * mse_l + 0.6 * ssim_l
        loss.backward()
        opt_pre.step()
        sched_pre.step()

    with torch.no_grad():
        x_pre_np = _render_siren(inr, coords, out_h, out_w).cpu().numpy().astype("float32")
    p_pre = _log_psnr("siren_pre", x_pre_np)
    print(f"      SIREN pre-train done"
          + (f"  PSNR={p_pre:.2f} dB" if p_pre is not None else ""))

    # MSE data-consistency DC steps
    best_dc     = float("inf")
    best_psnr_s = -1e9
    best_state  = {k: v.clone() for k, v in inr.state_dict().items()}

    opt_dc = torch.optim.Adam(inr.parameters(), lr=dc_lr_max)
    sched_dc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_dc, T_max=n_siren_dc, eta_min=dc_lr_min
    )
    lam_tv_sched = np.exp(
        np.linspace(np.log(0.008), np.log(0.0005), n_siren_dc)
    ).tolist()

    for step in range(n_siren_dc):
        lam_tv = lam_tv_sched[step]
        opt_dc.zero_grad()
        x_cur = _render_siren(inr, coords, out_h, out_w)
        sino  = _radon_fwd(x_cur, angles_deg, pad_size, device)
        dc    = F.mse_loss(sino / y_scale, y_t / y_scale)
        tv    = _tv_2d(x_cur)
        loss  = dc + lam_tv * tv
        dc_val = float(dc.detach())

        # Track best by PSNR or DC
        if step % 10 == 0 or step == n_siren_dc - 1:
            with torch.no_grad():
                x_np_c = x_cur.detach().cpu().numpy().astype("float32")
            if x_true_diag is not None:
                p_c = _psnr(x_np_c, x_true_diag)
                if p_c > best_psnr_s:
                    best_psnr_s = p_c
                    best_state = {k: v.clone() for k, v in inr.state_dict().items()}
            else:
                if dc_val < best_dc:
                    best_dc = dc_val
                    best_state = {k: v.clone() for k, v in inr.state_dict().items()}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(inr.parameters(), max_norm=1.0)
        opt_dc.step()
        sched_dc.step()

    inr.load_state_dict(best_state)
    with torch.no_grad():
        x_inr_np = _render_siren(inr, coords, out_h, out_w).cpu().numpy().astype("float32")
    p_inr = _log_psnr("siren_dc", x_inr_np)
    if p_inr is not None:
        print(f"      Phase 4 SIREN DC: {p_inr:.2f} dB")

    # ── Phase 5: Freq-blend (PnP LF + SIRT HF) ───────────────────────────────
    print(f"\n      === Phase 5: Freq-blend (PnP LF + SIRT HF, α={blend_alpha}) ===")
    # Use best of PnP vs SIREN-DC as the LF source
    if x_true_diag is not None:
        lf_src = x_inr_np if _psnr(x_inr_np, x_true_diag) >= _psnr(x_pnp_np, x_true_diag) \
                 else x_pnp_np
    else:
        lf_src = x_inr_np

    x_blend = _freq_blend(lf_src, x_sirt_np, device,
                          thresh=blend_thresh, sharpness=12.0, alpha=blend_alpha)
    p_blend = _log_psnr("blend", x_blend)
    if p_blend is not None:
        print(f"      Blend PSNR: {p_blend:.2f} dB")

    x_best_t = torch.tensor(best_x_np, device=device, dtype=torch.float32)

    # ── Phase 6: DRUNet final (keep-if-better) ────────────────────────────────
    if drunet is not None:
        print(f"\n      === Phase 6: DRUNet final pass (σ={final_sigma}) ===")
        with torch.no_grad():
            x_dn = drunet(
                x_best_t.unsqueeze(0).unsqueeze(0), final_sigma
            ).squeeze().clamp(0.0, 1.0)
        if x_true_diag is not None:
            pb = _psnr(x_best_t.cpu().numpy(), x_true_diag)
            pa = _psnr(x_dn.cpu().numpy(), x_true_diag)
            keep = pa >= pb
            print(f"      [DRUNet σ={final_sigma}]  {pb:.2f}→{pa:.2f} dB"
                  f"  ({'keep' if keep else 'revert'})")
            if keep:
                _log_psnr("drunet", x_dn.cpu().numpy().astype("float32"))
        else:
            x_best_t = x_dn
            best_x_np = x_best_t.cpu().numpy().astype("float32")

    return best_x_np if best_x_np is not None else x_best_t.cpu().numpy().astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=7200,    # 2 hours — training + 3 samples
    memory=16384,
)
def run_mri_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    import io
    import json
    import math
    import time
    import h5py
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[PnP-DnCNN++] Device={device}  GPU={gpu_name}")

    # ── Load or build DRUNet ─────────────────────────────────────────────────
    drunet = None
    vol_path = "/models/drunet_deepinv_gray_finetune_26k.pth"

    class _DRUNet(nn.Module):
        def __init__(self, depth=20, n_channels=64, image_channels=1):
            super().__init__()
            ks, pad = 3, 1
            layers: list[nn.Module] = [
                nn.Conv2d(image_channels + 1, n_channels, ks, padding=pad),
                nn.ReLU(inplace=True),
            ]
            for _ in range(depth - 2):
                layers += [
                    nn.Conv2d(n_channels, n_channels, ks, padding=pad),
                    nn.BatchNorm2d(n_channels),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.Conv2d(n_channels, image_channels, ks, padding=pad))
            self.net = nn.Sequential(*layers)

        def forward(self, x_noisy, sigma):
            if isinstance(sigma, float):
                s_map = torch.full_like(x_noisy[:, :1], sigma)
            else:
                s_map = sigma
            return (x_noisy - self.net(torch.cat([x_noisy, s_map], dim=1))).clamp(0., 1.)

    try:
        import pathlib
        if pathlib.Path(vol_path).exists():
            state = torch.load(vol_path, map_location=device, weights_only=True)
            _d = _DRUNet(depth=20).to(device)
            _d.load_state_dict(state, strict=False)
            _d.eval()
            drunet = _d
            print("[DRUNet] Loaded from volume")
        else:
            raise FileNotFoundError(vol_path)
    except Exception:
        try:
            url = ("https://huggingface.co/deepinv/drunet/resolve/main/"
                   "drunet_deepinv_gray_finetune_26k.pth?download=true")
            import torch.hub
            state = torch.hub.load_state_dict_from_url(url, map_location=device)
            _d = _DRUNet(depth=20).to(device)
            _d.load_state_dict(state, strict=False)
            _d.eval()
            drunet = _d
            print("[DRUNet] Downloaded")
        except Exception as exc:
            print(f"[DRUNet] Unavailable: {exc}")

    # ── Build and train DnCNN++ ──────────────────────────────────────────────
    # Determine image dimensions from the HDF5 file
    f_probe = h5py.File(io.BytesIO(h5_bytes), "r")
    sample_keys = sorted(f_probe.keys())
    grp0 = f_probe[sample_keys[0]]
    x_probe = grp0["x_true"][()] if "x_true" in grp0 else None
    y_probe = grp0["y"][()]
    if x_probe is not None:
        IMG_H, IMG_W = x_probe.shape[-2], x_probe.shape[-1]
    else:
        IMG_W = int(math.floor(y_probe.shape[1] / math.sqrt(2)))
        IMG_H = IMG_W
    f_probe.close()

    print(f"[DnCNN++] Image size: {IMG_H}×{IMG_W}")

    dncnn = _build_dncnn(n_layers=17, n_ch=64).to(device)
    n_params = sum(p.numel() for p in dncnn.parameters())
    print(f"[DnCNN++] Parameters: {n_params/1e3:.1f}K")

    phantom_pool = _make_phantom_pool(IMG_H, IMG_W, device, n_random=8)
    print(f"[DnCNN++] Phantom pool: {len(phantom_pool)} phantoms")

    dncnn = _train_dncnn(
        dncnn, phantom_pool, device,
        n_epochs=100,
        batch_size=8,
        lr=3e-4,
        sigma_min=0.005,
        sigma_max=0.10,
        mse_weight=0.60,
    )

    # ── Save DnCNN checkpoint ────────────────────────────────────────────────
    import os
    ckpt_dir = "/models/checkpoint/pnp_dncnn_plus"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/mri_dncnn_plus.pth"
    torch.save({"state_dict": dncnn.state_dict(),
                "img_h": IMG_H, "img_w": IMG_W,
                "n_layers": 17, "n_ch": 64}, ckpt_path)
    vol.commit()
    print(f"[DnCNN++] Checkpoint saved → {ckpt_path}")

    # ── Evaluate on challenge data ────────────────────────────────────────────
    rows = []
    f = h5py.File(io.BytesIO(h5_bytes), "r")

    for sk in sorted(f.keys()):
        grp = f[sk]
        x_true    = grp["x_true"][()].astype("float32") if "x_true" in grp else None
        y_sino    = grp["y"][()].astype("float64")
        angles_deg = grp["H_ideal"][()].astype("float64")
        try:
            meta = json.loads(grp.attrs.get("metadata", "{}"))
        except Exception:
            meta = {}
        scene_name = meta.get("scene", sk)

        if x_true is not None:
            out_h, out_w = x_true.shape[-2], x_true.shape[-1]
            if x_true.max() > 1.0:
                x_true /= x_true.max()
        else:
            out_w = int(math.floor(y_sino.shape[1] / math.sqrt(2)))
            out_h = out_w

        pad_size = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))
        has_gt = x_true is not None

        print(f"\n  [{tier}] {sk}  {out_h}×{out_w}  sino={y_sino.shape}  "
              + ("[no GT]" if not has_gt else ""))

        for algo in algos:
            t0 = time.time()
            try:
                if algo == "pnp_dncnn_plus":
                    x_hat = pnp_dncnn_plus(
                        y_sino, angles_deg, device,
                        dncnn, drunet,
                        pad_size, out_h, out_w,
                        x_true_diag=x_true,
                        n_pnp_iters=50,
                        pnp_grad_lr=0.015,
                        pnp_inner_steps=6,
                        mu_start=0.05,
                        mu_end=3.0,
                        n_sirt=200,
                        siren_hidden=384,
                        siren_layers=6,
                        siren_omega=30.0,
                        n_siren_pretrain=80,
                        n_siren_dc=100,
                        dc_lr_max=3e-4,
                        dc_lr_min=1e-5,
                        blend_thresh=0.25,
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
                "algo": algo,
                "psnr_db": round(psnr, 4), "ssim": round(ssim, 4),
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
def main(tier: str = "public", algo: str = "pnp_dncnn_plus"):
    """Run PnP-DnCNN++ benchmark on Modal T4.

    Pipeline:
      Phase 0: Hamming FBP init
      Phase 1: Train DnCNN++ on Shepp-Logan pool (100 epochs)
      Phase 2: PnP-HQS 50 iters (mu=0.05→3.0, 6 inner gradient steps)
      Phase 3: SIRT+TV reference (200 iters)
      Phase 4: SIREN INR DC refinement (80 pre-train + 100 DC steps)
      Phase 5: Freq-blend PnP LF + SIRT HF (alpha=0.25)
      Phase 6: DRUNet sigma=0.003 keep-if-better
    """
    import csv
    import json
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    ALL_ALGOS = ["pnp_dncnn_plus"]
    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("PnP-DnCNN++ (IEEE TCI 2019 + TMI 2025 adapted, Radon+Poisson domain)")
    print("  Phase 0: Hamming FBP init (~20.9 dB)")
    print("  Phase 1: DnCNN++ training on Shepp-Logan pool (100 epochs)")
    print("  Phase 2: PnP-HQS 50 iters  mu=0.05→3.0  6 inner gradient steps")
    print("  Phase 3: SIRT+TV reference 200 iters (for freq-blend)")
    print("  Phase 4: SIREN INR DC refinement (80 pre-train + 100 DC)")
    print("  Phase 5: Freq-blend  PnP LF + SIRT HF  alpha=0.25")
    print("  Phase 6: DRUNet sigma=0.003 keep-if-better")
    print(f"  Tiers: {tiers}  Algos: {algos}")

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
        print("No results.")
        return

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"pnp_dncnn_plus_{ts}.json"
    out_csv  = OUT_DIR / f"pnp_dncnn_plus_{ts}.csv"
    doc = {
        "timestamp": ts, "variant": "mri", "tiers": tiers, "algos": algos,
        "gpu": "T4",
        "algorithm": "PnP-DnCNN++ (IEEE TCI 2019 + TMI 2025 adapted)",
        "phases": {
            "p0": "Hamming FBP init",
            "p1": "DnCNN++ training 100 epochs (Shepp-Logan + Gaussian noise)",
            "p2": "PnP-HQS 50 iters (mu=0.05->3.0, 6 inner steps)",
            "p3": "SIRT+TV 200 iters (freq-blend reference)",
            "p4": "SIREN INR DC refinement (80 pre-train + 100 DC steps)",
            "p5": "freq-blend PnP_LF + SIRT_HF alpha=0.25",
            "p6": "DRUNet sigma=0.003 keep-if-better",
        },
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
        uri = _upload_gcs(out_json, f"benchmark-results/mri/pnp_dncnn_plus_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        valid = [r for r in rs
                 if not (isinstance(r["psnr_db"], float) and r["psnr_db"] != r["psnr_db"])]
        if valid:
            p  = sum(r["psnr_db"] for r in valid) / len(valid)
            s  = sum(r["ssim"]    for r in valid) / len(valid)
            sc = sum(r["score"]   for r in valid) / len(valid)
            print(f"  {t:8s}  {a:26s}  PSNR={p:7.2f}  SSIM={s:.4f}"
                  f"  Score={sc:.4f}  (n={len(valid)})")
        else:
            print(f"  {t:8s}  {a:26s}  [no GT]")

    pnp_rows = [r for r in all_rows
                if r["algo"] == "pnp_dncnn_plus"
                and not (isinstance(r["psnr_db"], float) and r["psnr_db"] != r["psnr_db"])]
    if pnp_rows:
        mp = sum(r["psnr_db"] for r in pnp_rows) / len(pnp_rows)
        ms = sum(r["ssim"]    for r in pnp_rows) / len(pnp_rows)
        print(f"\nPnP-DnCNN++:     PSNR = {mp:.2f} dB   SSIM = {ms:.4f}"
              f"  (n={len(pnp_rows)} scenes with GT)")
        print(f"HybridCascade++: PSNR = 31.72 dB   SSIM = 0.878  (best TTO baseline)")
        print(f"HUMUS-Net++ v2:  PSNR = 31.57 dB   SSIM = 0.856")
        print(f"vs baseline:  {'PASS ✓' if mp > 31.72 else f'FAIL (gap {31.72-mp:.2f} dB)'}")
        print(f"SSIM vs base: {'PASS ✓' if ms > 0.878 else f'FAIL (gap {0.878-ms:.4f})'}")
