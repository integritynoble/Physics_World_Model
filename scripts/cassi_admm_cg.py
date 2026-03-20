#!/usr/bin/env python3
"""CASSI ADMM with CG inner solver — proper spectral demixing."""
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


def cassi_forward(x, mask, n_bands, step):
    """A(x): CASSI forward model."""
    h, w, _ = x.shape
    w_ext = w + (n_bands - 1) * step
    y = np.zeros((h, w_ext), dtype=np.float32)
    for k in range(n_bands):
        y[:, k * step:k * step + w] += mask * x[:, :, k]
    return y


def cassi_adjoint(y, mask, n_bands, step, w):
    """A^T(y): CASSI adjoint."""
    h = y.shape[0]
    x = np.zeros((h, w, n_bands), dtype=np.float32)
    for k in range(n_bands):
        x[:, :, k] = mask * y[:, k * step:k * step + w]
    return x


def cassi_AtA(x, mask, n_bands, step):
    """A^T A(x): Normal equation operator."""
    y = cassi_forward(x, mask, n_bands, step)
    return cassi_adjoint(y, mask, n_bands, step, x.shape[1])


def cg_solve(b, AtA_fn, x0, cg_iters=20, tol=1e-6, rho=1.0):
    """Solve (A^TA + rho*I)x = b using conjugate gradient.

    Args:
        b: Right-hand side
        AtA_fn: Function computing A^T A x
        x0: Initial guess
        cg_iters: Max CG iterations
        tol: Convergence tolerance
        rho: Regularization (for A^TA + rho*I)

    Returns:
        Solution x
    """
    x = x0.copy()
    r = b - (AtA_fn(x) + rho * x)
    p = r.copy()
    rsold = np.sum(r * r)

    for i in range(cg_iters):
        Ap = AtA_fn(p) + rho * p
        pAp = np.sum(p * Ap)
        if pAp < 1e-12:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def cassi_admm_cg(y, mask, n_bands, step, iters=50, rho=1.0, lam_tv=0.005,
                   cg_iters=20, tv_iters=10):
    """ADMM with CG inner solver for CASSI.

    min_x 0.5*||Ax - y||^2 + lam_tv * TV(x)

    ADMM splitting: x = z
    x-update: (A^TA + rho*I)x = A^Ty + rho*(z - u)  [solved with CG]
    z-update: z = TV_prox(x + u, lam_tv/rho)
    u-update: u += x - z
    """
    from pwm_core.recon.gap_tv import tv_denoiser_3d

    h, w_ext = y.shape
    w = w_ext - (n_bands - 1) * step

    # A^T y
    Aty = cassi_adjoint(y, mask, n_bands, step, w)

    # A^TA operator
    def AtA(x_in):
        return cassi_AtA(x_in, mask, n_bands, step)

    # Initialize with back-projection (diagonal approximation)
    Phi_sum = np.zeros((h, w_ext), dtype=np.float32)
    for k in range(n_bands):
        Phi_sum[:, k * step:k * step + w] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-6)

    x = np.zeros((h, w, n_bands), dtype=np.float32)
    for k in range(n_bands):
        x[:, :, k] = mask * y[:, k * step:k * step + w] / Phi_sum[:, k * step:k * step + w]

    z = x.copy()
    u = np.zeros_like(x)

    for it in range(iters):
        # x-update via CG: (A^TA + rho*I)x = A^Ty + rho*(z - u)
        rhs = Aty + rho * (z - u)
        x = cg_solve(rhs, AtA, x, cg_iters=cg_iters, rho=rho)

        # z-update: TV proximal
        v = x + u
        z = tv_denoiser_3d(v, lam_tv / rho, iterations=tv_iters, device='cpu')
        z = np.maximum(z, 0)

        # u-update
        u = u + x - z

        # Monitor
        if it % 10 == 0 or it == iters - 1:
            y_est = cassi_forward(z, mask, n_bands, step)
            data_fit = np.sum((y - y_est) ** 2)
            print(f"  iter {it:3d}: ||Ax-y||^2 = {data_fit:.4f}")

    return np.clip(z, 0, 1).astype(np.float32)


# ---------- Main ----------
f = h5py.File(str(ROOT / "datasets/benchmark/cassi/standard/standard_cassi_00.h5"), 'r')
y = np.array(f['y_ideal'], dtype=np.float32)
x_true = np.array(f['x_true'], dtype=np.float32)
mask = np.array(f['mask'], dtype=np.float32)
n_bands = len(f['wavelength'])
step_val = int(f['H_params'].attrs.get('step', 2))
f.close()

print(f"Data: y={y.shape}, x_true={x_true.shape}, n_bands={n_bands}, step={step_val}")
print()

# Test different configs
configs = [
    ("ADMM-CG rho=1 lam=0.005 50it cg=20", dict(iters=50, rho=1.0, lam_tv=0.005, cg_iters=20)),
    ("ADMM-CG rho=0.5 lam=0.005 50it cg=20", dict(iters=50, rho=0.5, lam_tv=0.005, cg_iters=20)),
    ("ADMM-CG rho=0.1 lam=0.005 50it cg=30", dict(iters=50, rho=0.1, lam_tv=0.005, cg_iters=30)),
    ("ADMM-CG rho=1 lam=0.001 50it cg=20", dict(iters=50, rho=1.0, lam_tv=0.001, cg_iters=20)),
    ("ADMM-CG rho=1 lam=0.01 50it cg=20", dict(iters=50, rho=1.0, lam_tv=0.01, cg_iters=20)),
]

print(f"{'Config':50s} {'PSNR':>10s} {'Time':>8s}")
print("-" * 72)

best_psnr = 0
best_cfg = None
for label, kwargs in configs:
    print(f"\n--- {label} ---")
    t0 = time.time()
    x_hat = cassi_admm_cg(y, mask, n_bands, step_val, **kwargs)
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    mark = " **" if p > best_psnr else ""
    if p > best_psnr:
        best_psnr = p
        best_cfg = label
    print(f"{label:50s} {p:8.2f} dB {dt:6.1f}s{mark}")

print(f"\nBest: {best_cfg} -> {best_psnr:.2f} dB")
print(f"Reference: GAP-TV on KAIST = 24.4 dB")
