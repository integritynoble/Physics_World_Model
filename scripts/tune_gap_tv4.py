#!/usr/bin/env python3
"""Test different CASSI reconstruction approaches to find what achieves reference PSNR."""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, pwm5_pkg)
sys.path.insert(0, str(ROOT))
for k in list(sys.modules):
    if 'pwm_core' in k or 'algorithm_base' in k:
        del sys.modules[k]

import numpy as np
import h5py

def psnr(x_hat, x_true):
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64)) ** 2)
    if mse < 1e-15: return float('inf')
    return float(10 * np.log10(float(x_true.max()) ** 2 / mse))

f = h5py.File(str(ROOT / "datasets/benchmark/cassi/standard/standard_cassi_00.h5"), 'r')
y = np.array(f['y_ideal'], dtype=np.float32)
x_true = np.array(f['x_true'], dtype=np.float32)
mask = np.array(f['mask'], dtype=np.float32)
n_bands = len(f['wavelength'])
step_val = int(f['H_params'].attrs.get('step', 2))
f.close()

h, w_meas = y.shape
w = w_meas - (n_bands - 1) * step_val

print(f"Data: y={y.shape}, x_true={x_true.shape}, n_bands={n_bands}, step={step_val}")
print(f"Mask: {mask.shape}, fill rate={mask.mean():.2f}")
print()

# ---------- Approach 1: ADMM-TV for CASSI ----------
def cassi_admm_tv(y, mask, n_bands, step, iters=50, rho=1.0, lam_tv=0.1):
    """ADMM-based CASSI reconstruction with TV prior.

    min_x ||Ax - y||^2 + lam_tv * TV(x)

    Using ADMM splitting:
    x-update: solve (A^T A + rho I) x = A^T y + rho (z - u)
    z-update: z = TV_prox(x + u, lam_tv/rho)
    u-update: u = u + x - z
    """
    h_img, w_img = mask.shape
    w_ext = w_img + (n_bands - 1) * step

    # Precompute A^T A diagonal
    Phi_sum = np.zeros((h_img, w_ext), dtype=np.float32)
    for k in range(n_bands):
        Phi_sum[:, k * step:k * step + w_img] += mask ** 2

    # Compute A^T y
    Aty = np.zeros((h_img, w_img, n_bands), dtype=np.float32)
    for k in range(n_bands):
        Aty[:, :, k] = mask * y[:, k * step:k * step + w_img]

    # Per-voxel diagonal of A^T A (for the diagonal CG approximation)
    AtA_diag = np.zeros((h_img, w_img, n_bands), dtype=np.float32)
    for k in range(n_bands):
        AtA_diag[:, :, k] = mask ** 2 * Phi_sum[:, k * step:k * step + w_img]
    AtA_diag = np.maximum(AtA_diag, 1e-8)

    # Initialize
    x = Aty / AtA_diag  # Least-squares initialization
    z = x.copy()
    u = np.zeros_like(x)

    from pwm_core.recon.gap_tv import tv_denoiser_3d

    for it in range(iters):
        # x-update: approximate solve of (A^T A + rho I) x = A^T y + rho(z - u)
        # Use diagonal approximation: x = (A^Ty + rho*(z-u)) / (diag(A^TA) + rho)
        rhs = Aty + rho * (z - u)
        x = rhs / (AtA_diag + rho)

        # z-update: TV proximal
        v = x + u
        z = tv_denoiser_3d(v, lam_tv / rho, iterations=10, device='cpu')
        z = np.maximum(z, 0)

        # u-update
        u = u + x - z

    return np.clip(z, 0, 1).astype(np.float32)


# ---------- Approach 2: Gradient descent with TV ----------
def cassi_gd_tv(y, mask, n_bands, step, iters=200, step_size=0.01, lam_tv=0.01):
    """Simple gradient descent on ||Ax-y||^2 + lam_tv * TV(x)."""
    h_img, w_img = mask.shape
    w_ext = w_img + (n_bands - 1) * step

    # Compute A^T y
    Aty = np.zeros((h_img, w_img, n_bands), dtype=np.float32)
    for k in range(n_bands):
        Aty[:, :, k] = mask * y[:, k * step:k * step + w_img]

    # Diagonal of A^T A for scaling
    Phi_sum = np.zeros((h_img, w_ext), dtype=np.float32)
    for k in range(n_bands):
        Phi_sum[:, k * step:k * step + w_img] += mask ** 2
    AtA_diag = np.zeros((h_img, w_img, n_bands), dtype=np.float32)
    for k in range(n_bands):
        AtA_diag[:, :, k] = mask * Phi_sum[:, k * step:k * step + w_img]

    # Init
    x = Aty / np.maximum(AtA_diag, 1e-6)

    from pwm_core.recon.gap_tv import tv_denoiser_3d

    for it in range(iters):
        # Forward
        y_est = np.zeros((h_img, w_ext), dtype=np.float32)
        for k in range(n_bands):
            y_est[:, k * step:k * step + w_img] += mask * x[:, :, k]

        # Gradient of data term: A^T(Ax - y)
        residual = y_est - y
        grad = np.zeros_like(x)
        for k in range(n_bands):
            grad[:, :, k] = mask * residual[:, k * step:k * step + w_img]

        # Gradient step
        x = x - step_size * grad

        # TV proximal (proximal gradient = ISTA)
        x = tv_denoiser_3d(x, lam_tv * step_size, iterations=5, device='cpu')
        x = np.maximum(x, 0)

    return np.clip(x, 0, 1).astype(np.float32)


# ---------- Approach 3: TwIST-style ----------
def cassi_twist(y, mask, n_bands, step, iters=100, lam_tv=0.005):
    """TwIST-style algorithm for CASSI.
    Two-step IST with TV regularization.
    """
    h_img, w_img = mask.shape
    w_ext = w_img + (n_bands - 1) * step

    # Estimate Lipschitz constant from diagonal
    Phi_sum = np.zeros((h_img, w_ext), dtype=np.float32)
    for k in range(n_bands):
        Phi_sum[:, k * step:k * step + w_img] += mask ** 2
    L = float(Phi_sum.max())  # Upper bound on largest eigenvalue

    alpha = 2.0 / (1 + L)  # IST step size
    beta = alpha  # TwIST parameter

    # Compute A^T y
    def At(r):
        out = np.zeros((h_img, w_img, n_bands), dtype=np.float32)
        for k in range(n_bands):
            out[:, :, k] = mask * r[:, k * step:k * step + w_img]
        return out

    def A(x_cube):
        out = np.zeros((h_img, w_ext), dtype=np.float32)
        for k in range(n_bands):
            out[:, k * step:k * step + w_img] += mask * x_cube[:, :, k]
        return out

    Aty = At(y)

    from pwm_core.recon.gap_tv import tv_denoiser_3d

    x = Aty / L  # Simple initialization
    x_prev = x.copy()

    for it in range(iters):
        # Gradient: A^T(Ax - y)
        grad = At(A(x) - y)

        # IST step
        v = x - (1.0 / L) * grad

        # TV proximal
        x_new = tv_denoiser_3d(v, lam_tv / L, iterations=5, device='cpu')
        x_new = np.maximum(x_new, 0)

        # TwIST momentum (two-step)
        if it > 0:
            x_next = (1 - alpha) * x_prev + (alpha - beta) * x + beta * x_new
        else:
            x_next = x_new

        x_prev = x
        x = x_next

    return np.clip(x, 0, 1).astype(np.float32)

# ---------- Run tests ----------
print(f"{'Config':55s} {'PSNR':>10s} {'Time':>8s}")
print("-" * 77)

configs = [
    # ADMM
    ("ADMM rho=0.1 lam=0.01 50it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=50, rho=0.1, lam_tv=0.01)),
    ("ADMM rho=0.5 lam=0.01 50it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=50, rho=0.5, lam_tv=0.01)),
    ("ADMM rho=1.0 lam=0.01 50it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=50, rho=1.0, lam_tv=0.01)),
    ("ADMM rho=1.0 lam=0.001 100it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=100, rho=1.0, lam_tv=0.001)),
    ("ADMM rho=0.1 lam=0.001 100it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=100, rho=0.1, lam_tv=0.001)),
    ("ADMM rho=0.5 lam=0.005 100it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=100, rho=0.5, lam_tv=0.005)),
    ("ADMM rho=0.1 lam=0.005 200it", lambda: cassi_admm_tv(y, mask, n_bands, step_val, iters=200, rho=0.1, lam_tv=0.005)),
    # GD+TV
    ("GD lr=0.01 lam=0.001 200it", lambda: cassi_gd_tv(y, mask, n_bands, step_val, iters=200, step_size=0.01, lam_tv=0.001)),
    ("GD lr=0.005 lam=0.001 200it", lambda: cassi_gd_tv(y, mask, n_bands, step_val, iters=200, step_size=0.005, lam_tv=0.001)),
    ("GD lr=0.02 lam=0.001 200it", lambda: cassi_gd_tv(y, mask, n_bands, step_val, iters=200, step_size=0.02, lam_tv=0.001)),
    # TwIST
    ("TwIST lam=0.001 100it", lambda: cassi_twist(y, mask, n_bands, step_val, iters=100, lam_tv=0.001)),
    ("TwIST lam=0.005 100it", lambda: cassi_twist(y, mask, n_bands, step_val, iters=100, lam_tv=0.005)),
    ("TwIST lam=0.01 100it", lambda: cassi_twist(y, mask, n_bands, step_val, iters=100, lam_tv=0.01)),
    ("TwIST lam=0.001 200it", lambda: cassi_twist(y, mask, n_bands, step_val, iters=200, lam_tv=0.001)),
]

best_psnr = 0
best_cfg = None
for label, fn in configs:
    t0 = time.time()
    x_hat = fn()
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    mark = " **" if p > best_psnr else ""
    if p > best_psnr:
        best_psnr = p
        best_cfg = label
    print(f"{label:55s} {p:8.2f} dB {dt:6.1f}s{mark}")

print(f"\nBest: {best_cfg} -> {best_psnr:.2f} dB")
print(f"Reference: GAP-TV on KAIST = 24.4 dB")
print(f"Back-projection baseline = 19.64 dB")
