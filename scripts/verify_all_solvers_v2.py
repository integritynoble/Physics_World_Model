#!/usr/bin/env python3
"""Comprehensive solver verification — tests every registered solver function.

For each of the 168 modalities, reads its YAML config, imports the solver,
creates synthetic test data matching the category_module, and verifies the
solver runs successfully on CPU.

Output: benchmark_results/solver_verification_v2.json
"""
import sys
import os
import json
import time
import importlib
import traceback
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

import yaml

CONFIGS_DIR = ROOT / "benchmarks" / "configs"
RESULTS_PATH = ROOT / "benchmark_results" / "solver_verification_v2.json"

rng = np.random.RandomState(42)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(ref, test):
    ref = ref.astype(np.float64).ravel()
    test = test.astype(np.float64).ravel()
    if ref.size != test.size:
        minlen = min(ref.size, test.size)
        ref, test = ref[:minlen], test[:minlen]
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    dr = ref.max() - ref.min()
    if dr < 1e-15:
        return 0.0
    return float(10 * np.log10(dr ** 2 / mse))


def compute_ssim(ref, test):
    try:
        from skimage.metrics import structural_similarity
        if ref.ndim > 2:
            ref = ref.reshape(ref.shape[0], -1)
            test = test.reshape(test.shape[0], -1)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
            test = test.reshape(1, -1)
        dr = float(ref.max() - ref.min())
        if dr < 1e-15:
            dr = 1.0
        return float(structural_similarity(
            ref.astype(np.float64), test.astype(np.float64), data_range=dr))
    except Exception:
        return -1.0


# ── Simple physics operator ─────────────────────────────────────────────────

class SimpleOperator:
    """Minimal physics operator for testing solvers."""

    def __init__(self, category, x_shape, y_shape, **kwargs):
        self.category = category
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.theta = kwargs.get("theta", {})
        self.psf = kwargs.get("psf", None)
        self.angles = kwargs.get("angles", None)
        self.mask = kwargs.get("mask", None)
        self.A = kwargs.get("A", None)
        self.wavelength = kwargs.get("wavelength", 532e-9)
        self.pixel_size = kwargs.get("pixel_size", 5e-6)
        self.z = kwargs.get("z", 1e-3)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.category == "microscopy_psf" and self.psf is not None:
            from scipy.signal import fftconvolve
            if x.ndim == 2:
                return fftconvolve(x, self.psf, mode="same").astype(np.float32)
            return x
        if self.category == "medical_ct_radon" and self.angles is not None:
            from scipy.ndimage import rotate
            n_angles = len(self.angles)
            N = x.shape[0]
            sino = np.zeros((n_angles, N), dtype=np.float32)
            for i, ang in enumerate(self.angles):
                rotated = rotate(x, np.degrees(ang), reshape=False)
                sino[i] = rotated.sum(axis=0)
            return sino
        if self.category == "compressive_mask" and self.A is not None:
            return (self.A @ x.ravel()).astype(np.float32)
        if self.category == "medical_mri_kspace" and self.mask is not None:
            kspace = np.fft.fftshift(np.fft.fft2(x))
            return (kspace * self.mask).astype(np.complex64)
        return x.copy()

    def adjoint(self, y):
        y = np.asarray(y)
        if self.category == "microscopy_psf" and self.psf is not None:
            from scipy.signal import fftconvolve
            if y.ndim == 2:
                return fftconvolve(y, self.psf[::-1, ::-1], mode="same").astype(np.float32)
            return y
        if self.category == "medical_ct_radon" and self.angles is not None:
            from scipy.ndimage import rotate
            N = self.x_shape[0]
            recon = np.zeros((N, N), dtype=np.float32)
            for i, ang in enumerate(self.angles):
                backproj = np.tile(y[i], (N, 1))
                recon += rotate(backproj, -np.degrees(ang), reshape=False)
            return recon / len(self.angles)
        if self.category == "compressive_mask" and self.A is not None:
            return (self.A.T @ y.ravel()).reshape(self.x_shape).astype(np.float32)
        if self.category == "medical_mri_kspace":
            return np.abs(np.fft.ifft2(np.fft.ifftshift(y))).astype(np.float32)
        return y.copy().astype(np.float32)


# ── Synthetic data generators by category ────────────────────────────────────

def make_phantom(N=64):
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)
    gt += 0.2 * (((x-0.22)/0.11)**2 + (y/0.25)**2 < 1)
    return gt


def make_blobs(N=64, n_blobs=10):
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(n_blobs):
        cx, cy = rng.randint(5, N-5, 2)
        r = rng.randint(2, 6)
        yy, xx = np.ogrid[-cx:N-cx, -cy:N-cy]
        mask = xx**2 + yy**2 <= r**2
        gt[mask] = rng.uniform(0.3, 1.0)
    return gt


def make_psf(N=15, sigma=2.0):
    c = N // 2
    y, x = np.ogrid[:N, :N]
    psf = np.exp(-((x-c)**2 + (y-c)**2) / (2*sigma**2)).astype(np.float32)
    return psf / psf.sum()


def create_test_data(category, x_shape, y_shape):
    """Create synthetic (gt, measurement, operator) for a category."""
    N = x_shape[0] if len(x_shape) > 0 else 64
    if N > 128:
        N = 64  # Cap for speed

    if category == "microscopy_psf":
        gt = make_blobs(N)
        psf = make_psf(15, 2.0)
        op = SimpleOperator(category, (N, N), (N, N), psf=psf)
        y = op.forward(gt) + rng.normal(0, 0.01, (N, N)).astype(np.float32)
        return gt, y, op

    elif category == "medical_ct_radon":
        gt = make_phantom(N)
        n_angles = max(30, N // 2)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        op = SimpleOperator(category, (N, N), (n_angles, N), angles=angles)
        y = op.forward(gt)
        return gt, y, op

    elif category == "compressive_mask":
        gt = make_phantom(N)
        M = N * N // 4
        A = rng.randn(M, N*N).astype(np.float32) / np.sqrt(M)
        op = SimpleOperator(category, (N, N), (M,), A=A)
        y = op.forward(gt) + rng.randn(M).astype(np.float32) * 0.01
        return gt, y, op

    elif category == "medical_mri_kspace":
        gt = make_phantom(N)
        mask = np.zeros((N, N), dtype=bool)
        center = N // 2
        c_width = N // 20
        mask[:, center-c_width:center+c_width] = True
        for j in range(N):
            if not mask[0, j] and rng.random() < 0.25:
                mask[:, j] = True
        op = SimpleOperator(category, (N, N), (N, N), mask=mask)
        y = op.forward(gt)
        return gt, y, op

    elif category == "remote_sensing_sar":
        gt = make_phantom(N)
        psf = make_psf(15, 3.0)
        op = SimpleOperator("microscopy_psf", (N, N), (N, N), psf=psf)
        y = op.forward(gt) + rng.normal(0, 0.02, (N, N)).astype(np.float32)
        return gt, y, op

    elif category == "electron_ctf":
        gt = make_blobs(N)
        psf = make_psf(15, 1.5)
        op = SimpleOperator("microscopy_psf", (N, N), (N, N), psf=psf)
        y = op.forward(gt) + rng.normal(0, 0.01, (N, N)).astype(np.float32)
        return gt, y, op

    elif category == "scanning_probe":
        gt = make_blobs(N)
        psf = make_psf(9, 1.0)
        op = SimpleOperator("microscopy_psf", (N, N), (N, N), psf=psf)
        y = op.forward(gt) + rng.normal(0, 0.01, (N, N)).astype(np.float32)
        return gt, y, op

    else:
        # Generic fallback
        gt = make_phantom(N)
        psf = make_psf(15, 2.0)
        op = SimpleOperator("microscopy_psf", (N, N), (N, N), psf=psf)
        y = op.forward(gt) + rng.normal(0, 0.01, (N, N)).astype(np.float32)
        return gt, y, op


# ── Main verification loop ──────────────────────────────────────────────────

def verify_solver_fn(fn, y, operator, cfg, gt, timeout=120):
    """Call a solver function and measure results."""
    start = time.time()
    try:
        result = fn(y.astype(np.float32), operator, cfg)
        elapsed = time.time() - start

        if isinstance(result, tuple):
            recon = result[0]
        else:
            recon = result

        if recon is None:
            return {"status": "returned_none", "exec_time": round(elapsed, 3)}

        recon = np.real(np.asarray(recon)).astype(np.float32)

        # Compute metrics
        gt_flat = gt.ravel()
        recon_flat = recon.ravel()
        if gt_flat.size != recon_flat.size:
            # Try reshape
            try:
                from scipy.ndimage import zoom
                scale = [gt.shape[i] / recon.shape[i] for i in range(min(len(gt.shape), len(recon.shape)))]
                recon = zoom(recon, scale, order=1).astype(np.float32)
            except Exception:
                return {
                    "status": "shape_mismatch",
                    "gt_shape": list(gt.shape),
                    "recon_shape": list(np.asarray(result[0] if isinstance(result, tuple) else result).shape),
                    "exec_time": round(elapsed, 3),
                }

        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)

        return {
            "status": "verified",
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
            "exec_time": round(elapsed, 3),
            "shape": list(recon.shape),
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "error",
            "error": str(e)[:300],
            "exec_time": round(elapsed, 3),
        }


def main():
    print("=" * 70)
    print("PWM5 Comprehensive Solver Verification v2")
    print("=" * 70)

    # Load all YAML configs
    configs = {}
    for f in sorted(os.listdir(CONFIGS_DIR)):
        if f.endswith(".yaml") and f != "_template.yaml":
            mod_id = f.replace(".yaml", "")
            with open(CONFIGS_DIR / f, encoding="utf-8") as fh:
                configs[mod_id] = yaml.safe_load(fh)

    print(f"\nLoaded {len(configs)} modality configs")

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "local_cpu",
        "modalities": {},
    }

    verified_total = 0
    error_total = 0
    skip_total = 0

    for mod_id, cfg_data in configs.items():
        category = cfg_data.get("category_module", "microscopy_psf")
        x_shape = cfg_data.get("x_shape", [64, 64])
        y_shape = cfg_data.get("y_shape", [64, 64])
        solvers = cfg_data.get("solvers", {}) or {}

        if not solvers:
            continue

        print(f"\n--- {mod_id} ({category}) ---")

        mod_results = {}
        gt, y_data, operator = create_test_data(category, x_shape, y_shape)

        for solver_key, solver_cfg in solvers.items():
            if not solver_cfg or not isinstance(solver_cfg, dict):
                continue

            module_path = solver_cfg.get("module", "")
            func_name = solver_cfg.get("function", "")
            is_gpu = solver_cfg.get("gpu", False)
            params = solver_cfg.get("params", {})

            if not module_path or not func_name:
                continue

            # Parse params — framework always passes {} as cfg (params is informational)
            if isinstance(params, dict):
                pass  # Already a dict, use as-is
            else:
                params = {}  # params: 0, "1.5M", etc. are model size, not config

            solver_label = f"{solver_key}: {module_path}.{func_name}"

            # Skip GPU solvers for now (verify separately)
            if is_gpu:
                print(f"  {solver_key:25s} SKIP (GPU)")
                mod_results[solver_key] = {
                    "status": "skipped_gpu",
                    "module": module_path,
                    "function": func_name,
                }
                skip_total += 1
                continue

            # Try import
            try:
                mod = importlib.import_module(module_path)
                fn = getattr(mod, func_name)
            except (ImportError, AttributeError) as e:
                print(f"  {solver_key:25s} IMPORT_ERR: {e}")
                mod_results[solver_key] = {
                    "status": "import_error",
                    "error": str(e)[:200],
                    "module": module_path,
                    "function": func_name,
                }
                error_total += 1
                continue

            # Run solver
            result = verify_solver_fn(fn, y_data, operator, params, gt)
            mod_results[solver_key] = result
            result["module"] = module_path
            result["function"] = func_name

            status = result.get("status", "unknown")
            if status == "verified":
                verified_total += 1
                p = result.get("psnr_db", 0)
                s = result.get("ssim", 0)
                t = result.get("exec_time", 0)
                print(f"  {solver_key:25s} OK  PSNR={p:7.2f}  SSIM={s:.4f}  t={t:.3f}s")
            elif status == "error":
                error_total += 1
                print(f"  {solver_key:25s} ERR: {result.get('error', '')[:60]}")
            elif status == "shape_mismatch":
                error_total += 1
                print(f"  {solver_key:25s} SHAPE: gt={result.get('gt_shape')} recon={result.get('recon_shape')}")
            else:
                error_total += 1
                print(f"  {solver_key:25s} {status}")

        all_results["modalities"][mod_id] = mod_results

    all_results["summary"] = {
        "verified": verified_total,
        "errors": error_total,
        "skipped_gpu": skip_total,
        "total_tested": verified_total + error_total,
    }

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {verified_total} verified, {error_total} errors, {skip_total} GPU-skipped")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
