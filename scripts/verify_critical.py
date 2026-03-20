#!/usr/bin/env python3
"""Verify critical solvers — the ones that failed in v2 but may work with correct calling."""
import sys
import os
import json
import time
import traceback
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

rng = np.random.RandomState(42)
results = {}
N = 64


def make_phantom(N=64):
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)
    return gt


def make_blobs(N=64, n=10):
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(n):
        cx, cy = rng.randint(5, N-5, 2)
        r = rng.randint(2, 6)
        yy, xx = np.ogrid[-cx:N-cx, -cy:N-cy]
        gt[xx**2 + yy**2 <= r**2] = rng.uniform(0.3, 1.0)
    return gt


def test(name, fn):
    start = time.time()
    try:
        out = fn()
        t = time.time() - start
        if out is None:
            results[name] = {"status": "NONE", "t": round(t, 3)}
            print(f"  {name:40s} NONE  t={t:.3f}s")
            return
        out = np.real(np.asarray(out)).astype(np.float32)
        results[name] = {"status": "OK", "shape": list(out.shape),
                         "min": float(np.nanmin(out)), "max": float(np.nanmax(out)),
                         "t": round(t, 3)}
        print(f"  {name:40s} OK  shape={out.shape}  range=[{np.nanmin(out):.3f},{np.nanmax(out):.3f}]  t={t:.3f}s")
    except Exception as e:
        t = time.time() - start
        results[name] = {"status": "ERROR", "error": str(e)[:200], "t": round(t, 3)}
        print(f"  {name:40s} ERROR: {str(e)[:80]}")


print("=" * 70)
print("Critical Solver Verification")
print("=" * 70)

# ─── 1. MRI solvers (direct calling) ──────────────────────────────────────
print("\n=== MRI solvers (direct call) ===")
from pwm_core.recon.mri_solvers import zero_filled_reconstruction, cs_mri_wavelet

gt = make_phantom(N)
kspace = np.fft.fftshift(np.fft.fft2(gt))
mask = np.zeros((N, N), dtype=bool)
center = N // 2
mask[:, center-3:center+3] = True
for j in range(N):
    if not mask[0, j] and rng.random() < 0.25:
        mask[:, j] = True
kspace_us = kspace * mask

test("mri_zero_filled_cpu", lambda: zero_filled_reconstruction(kspace_us.astype(np.complex64), mask, device='cpu'))
test("mri_cs_wavelet_cpu", lambda: cs_mri_wavelet(kspace_us.astype(np.complex64), mask.astype(np.float32), device='cpu'))

# ─── 2. MRI via run_gap_tv wrapper (checking interface) ──────────────────
print("\n=== GAP-TV wrapper (run_gap_tv) ===")
from pwm_core.recon.gap_tv import run_gap_tv

# CASSI test via wrapper
n_bands = 4
gt_cube = rng.rand(N, N, n_bands).astype(np.float32) * 0.5 + 0.3
coded_mask = (rng.rand(N, N) > 0.5).astype(np.float32)
step = 1
y_cassi = np.zeros((N, N + (n_bands-1)*step), dtype=np.float32)
for l in range(n_bands):
    y_cassi[:, l*step:l*step+N] += gt_cube[:,:,l] * coded_mask

class CASSIOp:
    def __init__(self):
        self.mask = coded_mask
        self.n_bands = n_bands
        self.step = step
        self.x_shape = (N, N, n_bands)
    def forward(self, x): return x
    def adjoint(self, y): return y
    def info(self): return {"modality": "cassi", "mask": self.mask, "n_bands": self.n_bands, "step": self.step}

test("cassi_run_gap_tv", lambda: run_gap_tv(y_cassi, CASSIOp(), {"iters": 10}))

# CACTI test via wrapper
T = 8
gt_vid = np.stack([make_blobs(N, 5+t) for t in range(T)]).astype(np.float32)
masks = (rng.rand(T, N, N) > 0.5).astype(np.float32)
y_cacti = np.sum(gt_vid * masks, axis=0)

class CACTIOp:
    def __init__(self):
        self.masks = masks
        self.n_frames = T
        self.x_shape = (T, N, N)
    def forward(self, x): return np.sum(x * self.masks, axis=0)
    def adjoint(self, y): return self.masks * y[None, :, :]
    def info(self): return {"modality": "cacti", "masks": self.masks, "n_frames": T}

test("cacti_run_gap_tv", lambda: run_gap_tv(y_cacti, CACTIOp(), {"iters": 10}))

# ─── 3. SPC solvers (direct calling) ──────────────────────────────────────
print("\n=== SPC solvers (direct call) ===")
from pwm_core.recon.spc_solvers import admm_spc, fista_l1

gt_flat = make_phantom(N).ravel()
M = N*N // 4
A = rng.randn(M, N*N).astype(np.float32) / np.sqrt(M)
y_spc = A @ gt_flat + rng.randn(M).astype(np.float32) * 0.01

test("spc_admm_cpu", lambda: admm_spc(y_spc, A, device='cpu'))
test("spc_fista_l1_cpu", lambda: fista_l1(y_spc, A, device='cpu'))

# ─── 4. Matrix (FISTA-L2) ─────────────────────────────────────────────────
print("\n=== Matrix FISTA-L2 (direct call) ===")
from pwm_core.recon.classical import fista_l2

gt_m = rng.randn(32*32).astype(np.float64) * 0.3
gt_m[np.abs(gt_m) < 0.15] = 0
M2 = 32*32 // 4
A2 = rng.randn(M2, 32*32).astype(np.float64) / np.sqrt(M2)
y_m = A2 @ gt_m

test("matrix_fista_l2", lambda: fista_l2(y_m, A2, lam=1e-3, iters=100))

# ─── 5. PnP solver ────────────────────────────────────────────────────────
print("\n=== PnP solver ===")
from pwm_core.recon.pnp import run_pnp

gt_pnp = make_phantom(N)
noisy = gt_pnp + rng.normal(0, 0.1, gt_pnp.shape).astype(np.float32)

class IdentityOp:
    def __init__(self, shape):
        self.x_shape = shape
    def forward(self, x):
        return x.copy()
    def adjoint(self, y):
        return y.copy()

test("pnp_hqs_nlm", lambda: run_pnp(noisy, IdentityOp((N,N)), {"algorithm": "hqs", "denoiser": "nlm", "iters": 10}))
test("pnp_hqs_gaussian", lambda: run_pnp(noisy, IdentityOp((N,N)), {"algorithm": "hqs", "denoiser": "gaussian", "iters": 10}))

# ─── 6. CT solvers (run_fbp, run_sart) ────────────────────────────────────
print("\n=== CT solvers (run_fbp, run_sart) ===")
from pwm_core.recon.ct_solvers import run_fbp, run_sart, fbp_2d, sart_2d

gt_ct = make_phantom(N)
from scipy.ndimage import rotate
n_ang = 60
angs = np.linspace(0, np.pi, n_ang, endpoint=False)
sino = np.zeros((n_ang, N), dtype=np.float32)
for i, a in enumerate(angs):
    sino[i] = rotate(gt_ct, np.degrees(a), reshape=False).sum(axis=0)

class CTOp:
    def __init__(self):
        self.angles = angs
        self.x_shape = (N, N)
    def forward(self, x): return x
    def adjoint(self, y): return y

test("ct_run_fbp", lambda: run_fbp(sino, CTOp(), {"filter": "ramlak", "output_size": N}))
test("ct_run_sart", lambda: run_sart(sino, CTOp(), {"iterations": 5, "output_size": N}))
test("ct_fbp_2d_direct", lambda: fbp_2d(sino, angs, "ramlak", N))
test("ct_sart_2d_direct", lambda: sart_2d(sino, angs, N, iterations=5))

# ─── 7. Lightsheet solver ─────────────────────────────────────────────────
print("\n=== Lightsheet solver ===")
from pwm_core.recon.lightsheet_solver import run_lightsheet

gt_ls = make_blobs(N)
from scipy.signal import fftconvolve
psf_ls = np.zeros((15, 15), dtype=np.float32)
psf_ls[7, :] = 1.0/15

class LSOp:
    def __init__(self):
        self.x_shape = (N, N)
        self.psf = psf_ls
    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)
    def adjoint(self, y):
        return fftconvolve(y, self.psf[::-1,::-1], mode='same').astype(np.float32)
    def info(self): return {"x_shape": (N,N), "stripe_direction": "horizontal"}

y_ls = fftconvolve(gt_ls, psf_ls, mode='same').astype(np.float32) + rng.normal(0, 0.02, (N,N)).astype(np.float32)
test("lightsheet_run", lambda: run_lightsheet(y_ls, LSOp(), {"solver": "fourier_notch", "stripe_direction": "horizontal"}))
test("lightsheet_vsnr", lambda: run_lightsheet(y_ls, LSOp(), {"solver": "vsnr", "iters": 20}))
test("lightsheet_wavelet", lambda: run_lightsheet(y_ls, LSOp(), {"solver": "wavelet"}))

# ─── 8. NeRF solver ───────────────────────────────────────────────────────
print("\n=== NeRF solver ===")
from pwm_core.recon.nerf_solver import run_nerf

n_views = 4
gt_nerf = make_blobs(N)
views = np.stack([gt_nerf + rng.normal(0, 0.05, (N,N)).astype(np.float32) for _ in range(n_views)])

class NeRFOp:
    def __init__(self):
        self.x_shape = (N, N)
        self.n_views = n_views
    def forward(self, x): return x
    def adjoint(self, y): return y
    def info(self): return {"n_views": n_views, "x_shape": (N,N)}

test("nerf_run", lambda: run_nerf(views.astype(np.float32), NeRFOp(), {"n_views": n_views, "iters": 5}))

# ─── 9. SIM solver ────────────────────────────────────────────────────────
print("\n=== SIM solver ===")
from pwm_core.recon.sim_solver import run_sim_reconstruction, wiener_sim_2d

gt_sim = make_blobs(N)
n_patterns = 9
sim_imgs = np.stack([gt_sim + 0.1*rng.randn(N,N).astype(np.float32) for _ in range(n_patterns)])

class SIMOp:
    def __init__(self):
        self.x_shape = (N, N)
        self.n_orientations = 3
        self.n_phases = 3
    def forward(self, x): return x
    def adjoint(self, y): return y
    def info(self): return {"n_orientations": 3, "n_phases": 3, "x_shape": (N,N)}

test("sim_run", lambda: run_sim_reconstruction(sim_imgs, SIMOp(), {"n_orientations": 3, "n_phases": 3}))

# ─── 10. Lensless solver ──────────────────────────────────────────────────
print("\n=== Lensless solver ===")
from pwm_core.recon.lensless_solver import run_lensless

gt_ll = make_blobs(N)
psf_ll = rng.rand(N, N).astype(np.float32); psf_ll /= psf_ll.sum()
y_ll = fftconvolve(gt_ll, psf_ll, mode='same').astype(np.float32)

class LenslessOp:
    def __init__(self):
        self.x_shape = (N, N)
        self.psf = psf_ll
    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)
    def adjoint(self, y):
        return fftconvolve(y, self.psf[::-1,::-1], mode='same').astype(np.float32)
    def info(self): return {"x_shape": (N,N), "psf": self.psf}

test("lensless_wiener", lambda: run_lensless(y_ll, LenslessOp(), {"method": "wiener", "reg": 0.01}))

# ─── 11. Integral solver ──────────────────────────────────────────────────
print("\n=== Integral solver ===")
from pwm_core.recon.integral_solver import run_integral

gt_int = make_phantom(N)
n_v = 9
views_int = np.stack([gt_int + rng.normal(0, 0.02, (N,N)).astype(np.float32) for _ in range(n_v)])

class IntOp:
    def __init__(self):
        self.x_shape = (N, N)
        self.n_views = n_v
    def forward(self, x): return x
    def adjoint(self, y): return y
    def info(self): return {"n_views": n_v, "x_shape": (N,N)}

test("integral_run", lambda: run_integral(views_int.astype(np.float32), IntOp(), {"n_views": n_v}))

# ─── 12. GPU solver tests ─────────────────────────────────────────────────
print("\n=== GPU solver tests (GTX 1660 Ti) ===")
import torch
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # CT FBP on GPU
    test("ct_fbp_gpu", lambda: fbp_2d(sino, angs, "ramlak", N, device='cuda'))

    # MRI on GPU
    test("mri_zero_filled_gpu", lambda: zero_filled_reconstruction(kspace_us.astype(np.complex64), mask, device='cuda'))
    test("mri_cs_wavelet_gpu", lambda: cs_mri_wavelet(kspace_us.astype(np.complex64), mask.astype(np.float32), device='cuda'))

    # SPC on GPU
    test("spc_admm_gpu", lambda: admm_spc(y_spc, A, device='cuda'))
    test("spc_fista_l1_gpu", lambda: fista_l1(y_spc, A, device='cuda'))

    # PnP on GPU
    test("pnp_hqs_gpu", lambda: run_pnp(noisy, IdentityOp((N,N)), {"algorithm": "hqs", "denoiser": "nlm", "iters": 5, "device": "cuda"}))

    # GAP-TV on GPU (CASSI)
    test("cassi_gap_tv_gpu", lambda: run_gap_tv(y_cassi, CASSIOp(), {"iters": 10, "device": "cuda"}))

    # GAP-TV on GPU (CACTI)
    test("cacti_gap_tv_gpu", lambda: run_gap_tv(y_cacti, CACTIOp(), {"iters": 10, "device": "cuda"}))

    # Richardson-Lucy on GPU
    from pwm_core.recon.richardson_lucy import run_richardson_lucy
    gt_rl = make_blobs(N)
    psf_rl = np.zeros((15, 15), dtype=np.float32)
    c = 7
    for yi in range(15):
        for xi in range(15):
            psf_rl[yi, xi] = np.exp(-((xi-c)**2 + (yi-c)**2) / 8)
    psf_rl /= psf_rl.sum()

    class RLOp:
        def __init__(self):
            self.x_shape = (N, N)
            self.psf = psf_rl
        def forward(self, x):
            return fftconvolve(x, self.psf, mode='same').astype(np.float32)
        def adjoint(self, y):
            return fftconvolve(y, self.psf[::-1,::-1], mode='same').astype(np.float32)
        def info(self): return {"psf": self.psf}

    y_rl = fftconvolve(gt_rl, psf_rl, mode='same').astype(np.float32) + rng.normal(0, 0.01, (N,N)).astype(np.float32)
    test("rl_run_gpu", lambda: run_richardson_lucy(np.maximum(y_rl, 1e-6), RLOp(), {"iters": 30, "device": "cuda"}))

    # CARE DL solver (needs to build model on-the-fly)
    from pwm_core.recon.care_unet import care_restore_2d
    test("care_restore_2d_gpu", lambda: care_restore_2d(y_rl, RLOp(), {"device": "cuda"}))

    # RedCNN
    from pwm_core.recon.redcnn import redcnn_denoise
    test("redcnn_gpu", lambda: redcnn_denoise(sino, CTOp(), {"device": "cuda"}))

else:
    print("  No GPU available!")

# ─── 13. Check missing functions ──────────────────────────────────────────
print("\n=== Missing function checks ===")
try:
    from pwm_core.recon.classical import run_fista_l2
    print("  run_fista_l2: EXISTS ✓")
    results["fn_run_fista_l2"] = {"status": "OK"}
except (ImportError, AttributeError) as e:
    print(f"  run_fista_l2: MISSING ✗ ({e})")
    results["fn_run_fista_l2"] = {"status": "MISSING"}

try:
    from pwm_core.recon.noise2void import noise2void_denoise
    print("  noise2void_denoise: EXISTS ✓")
    results["fn_noise2void"] = {"status": "OK"}
except (ImportError, AttributeError) as e:
    print(f"  noise2void_denoise: MISSING ✗ ({e})")
    results["fn_noise2void"] = {"status": "MISSING"}

# ─── Summary ──────────────────────────────────────────────────────────────
ok = sum(1 for r in results.values() if r["status"] == "OK")
err = sum(1 for r in results.values() if r["status"] in ("ERROR", "MISSING"))
print(f"\n{'='*70}")
print(f"SUMMARY: {ok} OK, {err} errors, {len(results)} total")

os.makedirs(ROOT / "benchmark_results", exist_ok=True)
with open(ROOT / "benchmark_results" / "critical_verification.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to benchmark_results/critical_verification.json")
