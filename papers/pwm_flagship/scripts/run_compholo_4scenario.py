#!/usr/bin/env python3
"""Compressive Holography 4-Scenario Validation for PWM Nature Paper.

Simulated multi-depth holograms with propagation distance mismatch.
Gate 3 mismatch: depth error causes defocus at each plane.

Protocol:
  Scenario I:   Correct propagation distances -> reference reconstruction
  Scenario II:  Mismatched distances (defocus errors of 10, 50, 100, 200 um)
  Scenario III: Calibrated via autofocus sharpness metric (grid search)
  Recovery ratio = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)

Solver: Angular spectrum back-propagation (adjoint) + FISTA-TV inline.
Calibration: grid search over prop_distance_error in [-250, 250] um
maximising a Brenner sharpness metric on the reconstruction.

Usage:
    python run_compholo_4scenario.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Make pwm_core importable
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

from pwm_core.physics.microscopy.compressive_holography_operator import (  # noqa: E402
    CompressiveHolographyOperator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute PSNR between reference and test images."""
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(ref) - np.min(ref)
    if max_val < 1e-15:
        return 0.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    """Simplified SSIM computation."""
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * (ref.max() - ref.min())) ** 2
    C2 = (0.03 * (ref.max() - ref.min())) ** 2

    mu_x = uniform_filter(ref, win_size)
    mu_y = uniform_filter(test, win_size)
    sigma_x2 = uniform_filter(ref ** 2, win_size) - mu_x ** 2
    sigma_y2 = uniform_filter(test ** 2, win_size) - mu_y ** 2
    sigma_xy = uniform_filter(ref * test, win_size) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Multi-depth USAF-chart phantom
# ---------------------------------------------------------------------------
def _draw_bar_group(plane: np.ndarray, cy: int, cx: int,
                    bar_w: int, bar_h: int, n_bars: int,
                    horizontal: bool, value: float) -> None:
    """Draw a group of resolution bars onto *plane* (in-place)."""
    spacing = bar_w * 2
    total = n_bars * spacing
    for i in range(n_bars):
        if horizontal:
            y0 = cy - bar_h // 2
            y1 = cy + bar_h // 2
            x0 = cx - total // 2 + i * spacing
            x1 = x0 + bar_w
        else:
            x0 = cx - bar_h // 2
            x1 = cx + bar_h // 2
            y0 = cy - total // 2 + i * spacing
            y1 = y0 + bar_w
        y0, y1 = max(y0, 0), min(y1, plane.shape[0])
        x0, x1 = max(x0, 0), min(x1, plane.shape[1])
        plane[y0:y1, x0:x1] = value


def _draw_circle_ring(plane: np.ndarray, cy: int, cx: int,
                      r_outer: int, r_inner: int, value: float) -> None:
    """Draw a circle annulus onto *plane* (in-place)."""
    ny, nx = plane.shape
    yy, xx = np.ogrid[:ny, :nx]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (dist2 <= r_outer ** 2) & (dist2 >= r_inner ** 2)
    plane[mask] = value


def _draw_cross(plane: np.ndarray, cy: int, cx: int,
                arm_len: int, arm_w: int, value: float) -> None:
    """Draw a cross target onto *plane* (in-place)."""
    ny, nx = plane.shape
    # Horizontal arm
    y0 = max(cy - arm_w // 2, 0)
    y1 = min(cy + arm_w // 2 + 1, ny)
    x0 = max(cx - arm_len, 0)
    x1 = min(cx + arm_len + 1, nx)
    plane[y0:y1, x0:x1] = value
    # Vertical arm
    y0 = max(cy - arm_len, 0)
    y1 = min(cy + arm_len + 1, ny)
    x0 = max(cx - arm_w // 2, 0)
    x1 = min(cx + arm_w // 2 + 1, nx)
    plane[y0:y1, x0:x1] = value


def _draw_checkerboard(plane: np.ndarray, cy: int, cx: int,
                       size: int, block: int, value: float) -> None:
    """Draw a small checkerboard patch centred at (cy, cx)."""
    ny, nx = plane.shape
    half = size // 2
    for iy in range(size):
        for ix in range(size):
            if ((iy // block) + (ix // block)) % 2 == 0:
                py = cy - half + iy
                px = cx - half + ix
                if 0 <= py < ny and 0 <= px < nx:
                    plane[py, px] = value


def generate_multidepth_phantom(n_depths: int = 4, ny: int = 64,
                                nx: int = 64) -> np.ndarray:
    """Create a simulated multi-depth USAF-style resolution target.

    Each depth plane contains a different class of feature so that
    per-plane PSNR is independently informative:

        Plane 0: Horizontal resolution bars (fine, 2 px)
        Plane 1: Concentric circle rings
        Plane 2: Cross / crosshair targets
        Plane 3: Checkerboard patches

    Returns:
        phantom: (n_depths, ny, nx) float64 in [0, 1]
    """
    phantom = np.zeros((n_depths, ny, nx), dtype=np.float64)

    # --- Plane 0: horizontal resolution bars (fine, medium, coarse) ---
    plane = phantom[0]
    _draw_bar_group(plane, cy=16, cx=32, bar_w=2, bar_h=10, n_bars=5,
                    horizontal=True, value=1.0)
    _draw_bar_group(plane, cy=40, cx=20, bar_w=3, bar_h=12, n_bars=4,
                    horizontal=False, value=0.8)
    _draw_bar_group(plane, cy=48, cx=48, bar_w=4, bar_h=14, n_bars=3,
                    horizontal=True, value=0.6)

    # --- Plane 1: concentric circle rings ---
    plane = phantom[1]
    _draw_circle_ring(plane, cy=32, cx=32, r_outer=20, r_inner=16, value=1.0)
    _draw_circle_ring(plane, cy=32, cx=32, r_outer=14, r_inner=11, value=0.8)
    _draw_circle_ring(plane, cy=32, cx=32, r_outer=9, r_inner=7, value=0.6)
    _draw_circle_ring(plane, cy=32, cx=32, r_outer=5, r_inner=0, value=1.0)

    # --- Plane 2: cross targets at various positions ---
    plane = phantom[2]
    _draw_cross(plane, cy=16, cx=16, arm_len=8, arm_w=2, value=1.0)
    _draw_cross(plane, cy=16, cx=48, arm_len=6, arm_w=3, value=0.7)
    _draw_cross(plane, cy=48, cx=32, arm_len=10, arm_w=2, value=0.9)

    # --- Plane 3: checkerboard patches ---
    plane = phantom[3]
    _draw_checkerboard(plane, cy=20, cx=20, size=16, block=2, value=1.0)
    _draw_checkerboard(plane, cy=20, cx=48, size=12, block=3, value=0.8)
    _draw_checkerboard(plane, cy=48, cx=32, size=14, block=4, value=0.6)

    return phantom


# ---------------------------------------------------------------------------
# FISTA-TV solver (inline, no external dependency)
# ---------------------------------------------------------------------------
def _tv_proximal(x: np.ndarray, lam: float, n_iter: int = 10) -> np.ndarray:
    """Isotropic total-variation proximal operator via Chambolle's projection.

    Operates independently on each depth plane of x (K, ny, nx).
    """
    out = np.empty_like(x)
    for k in range(x.shape[0]):
        out[k] = _tv_proximal_2d(x[k], lam, n_iter)
    return out


def _tv_proximal_2d(img: np.ndarray, lam: float, n_iter: int = 10) -> np.ndarray:
    """2D TV proximal via Chambolle's dual projection algorithm."""
    ny, nx = img.shape
    # Dual variables
    py = np.zeros((ny, nx), dtype=np.float64)
    px = np.zeros((ny, nx), dtype=np.float64)
    tau = 0.25  # step size <= 1/(8) for 2D TV

    for _ in range(n_iter):
        # div(p)
        div = np.zeros((ny, nx), dtype=np.float64)
        # d/dy (py)
        div[1:, :] += py[1:, :] - py[:-1, :]
        div[0, :] += py[0, :]
        # d/dx (px)
        div[:, 1:] += px[:, 1:] - px[:, :-1]
        div[:, 0] += px[:, 0]

        u = img - lam * div

        # Gradient of u
        gy = np.zeros_like(u)
        gx = np.zeros_like(u)
        gy[:-1, :] = u[1:, :] - u[:-1, :]
        gx[:, :-1] = u[:, 1:] - u[:, :-1]

        # Update dual
        py += tau * gy
        px += tau * gx

        # Project onto unit ball
        norm = np.sqrt(py ** 2 + px ** 2)
        norm = np.maximum(norm, 1.0)
        py /= norm
        px /= norm

    # Final reconstruction
    div = np.zeros((ny, nx), dtype=np.float64)
    div[1:, :] += py[1:, :] - py[:-1, :]
    div[0, :] += py[0, :]
    div[:, 1:] += px[:, 1:] - px[:, :-1]
    div[:, 0] += px[:, 0]

    return img - lam * div


def fista_tv(
    operator: CompressiveHolographyOperator,
    hologram: np.ndarray,
    lam_tv: float = 0.005,
    n_iter: int = 80,
    lip: float | None = None,
) -> np.ndarray:
    """FISTA with TV proximal for compressive holography reconstruction.

    Minimises  0.5 * ||A x - y||^2 + lam_tv * TV(x)
    where A is the forward operator (multi-depth -> hologram).

    Args:
        operator: CompressiveHolographyOperator instance (defines A, A^T).
        hologram: Measured hologram (ny, nx).
        lam_tv: TV regularisation weight.
        n_iter: Number of FISTA iterations.
        lip: Lipschitz constant of A^T A.  Estimated via power iteration if None.

    Returns:
        x: Reconstructed multi-depth object (n_depths, ny, nx).
    """
    # Estimate Lipschitz constant via power iteration if not given
    if lip is None:
        lip = _estimate_lipschitz(operator, n_iter=20)
    step = 1.0 / lip

    # Initialise from adjoint (angular spectrum back-propagation)
    x = operator.adjoint(hologram).astype(np.float64)
    x_prev = x.copy()
    t = 1.0

    for it in range(n_iter):
        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / t_new
        z = x + momentum * (x - x_prev)

        # Gradient step:  z - step * A^T(A z - y)
        residual = operator.forward(z.astype(np.float64)) - hologram.astype(np.float64)
        grad = operator.adjoint(residual).astype(np.float64)
        z_grad = z - step * grad

        # TV proximal
        x_prev = x.copy()
        x = _tv_proximal(z_grad, lam=lam_tv * step, n_iter=8)

        # Non-negativity
        np.clip(x, 0.0, None, out=x)

        t = t_new

    return x


def _estimate_lipschitz(operator: CompressiveHolographyOperator,
                        n_iter: int = 20) -> float:
    """Estimate the spectral norm of A^T A via power iteration."""
    rng = np.random.RandomState(0)
    x = rng.randn(*operator.x_shape).astype(np.float64)
    x /= np.linalg.norm(x)

    for _ in range(n_iter):
        y = operator.forward(x)
        x_new = operator.adjoint(y).astype(np.float64)
        norm = np.linalg.norm(x_new)
        if norm < 1e-15:
            return 1.0
        x = x_new / norm

    # One more forward-adjoint to get the eigenvalue
    y = operator.forward(x)
    x_new = operator.adjoint(y).astype(np.float64)
    lip = float(np.linalg.norm(x_new) / (np.linalg.norm(x) + 1e-15))
    return lip * 1.05  # small safety margin


# ---------------------------------------------------------------------------
# Sharpness metric for autofocus calibration
# ---------------------------------------------------------------------------
def brenner_sharpness(vol: np.ndarray) -> float:
    """Brenner gradient sharpness summed over all depth planes.

    S = sum_{k,y,x} (I[k,y,x+2] - I[k,y,x])^2
    Higher is sharper (better focused).
    """
    s = 0.0
    for k in range(vol.shape[0]):
        plane = vol[k].astype(np.float64)
        diff = plane[:, 2:] - plane[:, :-2]
        s += float(np.sum(diff ** 2))
    return s


def autofocus_grid_search(
    hologram: np.ndarray,
    base_operator_kwargs: dict,
    search_range_um: tuple[float, float] = (-250.0, 250.0),
    n_search: int = 51,
    lam_tv: float = 0.005,
    fista_iters: int = 50,
) -> tuple[float, np.ndarray]:
    """Grid-search autofocus: find prop_distance_error that maximises sharpness.

    Sweeps over candidate propagation distance errors, reconstructs with
    FISTA-TV (reduced iterations for speed), and picks the offset that
    produces the sharpest volume.

    Returns:
        best_error_um: Optimal propagation distance error.
        best_recon: Reconstruction at the optimal error.
    """
    candidates = np.linspace(search_range_um[0], search_range_um[1], n_search)
    best_sharpness = -np.inf
    best_error = 0.0
    best_recon = None

    for err in candidates:
        kwargs = dict(base_operator_kwargs)
        kwargs["prop_distance_error_um"] = float(err)
        op = CompressiveHolographyOperator(**kwargs)
        recon = fista_tv(op, hologram, lam_tv=lam_tv, n_iter=fista_iters)
        sharp = brenner_sharpness(recon)
        if sharp > best_sharpness:
            best_sharpness = sharp
            best_error = float(err)
            best_recon = recon.copy()

    # Refine around the best with finer grid and full iterations
    fine_range = (best_error - (search_range_um[1] - search_range_um[0]) / n_search,
                  best_error + (search_range_um[1] - search_range_um[0]) / n_search)
    fine_candidates = np.linspace(fine_range[0], fine_range[1], 11)
    for err in fine_candidates:
        kwargs = dict(base_operator_kwargs)
        kwargs["prop_distance_error_um"] = float(err)
        op = CompressiveHolographyOperator(**kwargs)
        recon = fista_tv(op, hologram, lam_tv=lam_tv, n_iter=fista_iters + 30)
        sharp = brenner_sharpness(recon)
        if sharp > best_sharpness:
            best_sharpness = sharp
            best_error = float(err)
            best_recon = recon.copy()

    return best_error, best_recon


# ---------------------------------------------------------------------------
# Hologram residual: ||y - A x||^2 / ||y||^2
# ---------------------------------------------------------------------------
def hologram_residual(operator: CompressiveHolographyOperator,
                      hologram: np.ndarray, recon: np.ndarray) -> float:
    """Normalised measurement residual."""
    y_hat = operator.forward(recon).astype(np.float64)
    y = hologram.astype(np.float64)
    num = float(np.sum((y - y_hat) ** 2))
    den = float(np.sum(y ** 2))
    return num / max(den, 1e-15)


# ---------------------------------------------------------------------------
# 4-Scenario experiment
# ---------------------------------------------------------------------------
def run_compholo_4scenario() -> dict:
    """Run the full 4-scenario protocol for compressive holography."""
    logger.info("=" * 60)
    logger.info("Compressive Holography: 4-Scenario Protocol")
    logger.info("Gate 3 mismatch: propagation distance error")
    logger.info("=" * 60)

    # ---- Physical parameters ----
    ny, nx = 64, 64
    n_depths = 4
    depth_spacing_um = 100.0
    wavelength_nm = 532.0
    pixel_size_um = 5.0
    carrier_freq = 0.15
    lam_tv = 0.005
    fista_iters = 80

    base_kwargs = dict(
        ny=ny, nx=nx,
        n_depths=n_depths,
        depth_spacing_um=depth_spacing_um,
        wavelength_nm=wavelength_nm,
        pixel_size_um=pixel_size_um,
        carrier_freq=carrier_freq,
        prop_distance_error_um=0.0,
        carrier_freq_error=0.0,
        wavelength_error_nm=0.0,
    )

    # ---- Generate phantom ----
    logger.info("Generating multi-depth USAF-chart phantom (%d depths, %dx%d)...",
                n_depths, ny, nx)
    phantom = generate_multidepth_phantom(n_depths=n_depths, ny=ny, nx=nx)
    logger.info("  Phantom range: [%.3f, %.3f]  non-zero voxels: %d / %d",
                phantom.min(), phantom.max(),
                int(np.count_nonzero(phantom)),
                phantom.size)

    # ---- Simulate hologram with correct operator (no mismatch) ----
    op_true = CompressiveHolographyOperator(**base_kwargs)
    hologram = op_true.forward(phantom)
    logger.info("  Hologram shape: %s  range: [%.4f, %.4f]",
                hologram.shape, hologram.min(), hologram.max())

    # Add a small amount of Gaussian noise (realistic detector noise)
    rng = np.random.RandomState(42)
    noise_std = 0.01 * (hologram.max() - hologram.min())
    hologram_noisy = hologram.astype(np.float64) + rng.randn(*hologram.shape) * noise_std
    hologram_noisy = hologram_noisy.astype(np.float32)
    logger.info("  Added Gaussian noise sigma=%.5f", noise_std)

    # ---- Scenario I: correct operator, FISTA-TV ----
    logger.info("\n--- Scenario I: correct propagation distances ---")
    t0 = time.time()
    recon_I = fista_tv(op_true, hologram_noisy, lam_tv=lam_tv, n_iter=fista_iters)
    t_I = time.time() - t0
    psnr_I = psnr(phantom, recon_I)
    ssim_I = ssim_simple(phantom.ravel(), recon_I.ravel())
    res_I = hologram_residual(op_true, hologram_noisy, recon_I)
    logger.info("  PSNR=%.2f dB   SSIM=%.4f   residual=%.6f   (%.1fs)",
                psnr_I, ssim_I, res_I, t_I)

    # Per-plane PSNR
    plane_psnr_I = [psnr(phantom[k], recon_I[k]) for k in range(n_depths)]
    logger.info("  Per-plane PSNR: %s",
                "  ".join(f"z{k}={p:.2f}" for k, p in enumerate(plane_psnr_I)))

    # ---- Propagation distance errors to test ----
    prop_errors_um = [10.0, 50.0, 100.0, 200.0]
    results = {
        "modality": "compressive_holography",
        "phantom": "multi_depth_USAF_chart",
        "ny": ny,
        "nx": nx,
        "n_depths": n_depths,
        "depth_spacing_um": depth_spacing_um,
        "wavelength_nm": wavelength_nm,
        "pixel_size_um": pixel_size_um,
        "carrier_freq": carrier_freq,
        "noise_std": float(noise_std),
        "lam_tv": lam_tv,
        "fista_iters": fista_iters,
        "scenario_I": {
            "psnr_db": psnr_I,
            "ssim": ssim_I,
            "residual": res_I,
            "plane_psnr_db": plane_psnr_I,
            "time_s": t_I,
        },
        "mismatch_experiments": [],
    }

    for prop_err in prop_errors_um:
        logger.info("\n" + "-" * 50)
        logger.info("Propagation distance error: %.1f um", prop_err)
        logger.info("-" * 50)

        # ---- Scenario II: mismatched operator ----
        logger.info("  Scenario II: mismatched (+%.1f um)...", prop_err)
        kwargs_mis = dict(base_kwargs)
        kwargs_mis["prop_distance_error_um"] = prop_err
        op_mis = CompressiveHolographyOperator(**kwargs_mis)

        t0 = time.time()
        recon_II = fista_tv(op_mis, hologram_noisy, lam_tv=lam_tv, n_iter=fista_iters)
        t_II = time.time() - t0
        psnr_II = psnr(phantom, recon_II)
        ssim_II = ssim_simple(phantom.ravel(), recon_II.ravel())
        res_II_self = hologram_residual(op_mis, hologram_noisy, recon_II)
        res_II_cross = hologram_residual(op_true, hologram_noisy, recon_II)
        delta_II = psnr_I - psnr_II

        plane_psnr_II = [psnr(phantom[k], recon_II[k]) for k in range(n_depths)]
        logger.info("    PSNR=%.2f dB  (Delta=%.2f dB)  SSIM=%.4f  (%.1fs)",
                    psnr_II, delta_II, ssim_II, t_II)
        logger.info("    Self residual=%.6f   Cross residual=%.6f   ratio=%.2fx",
                    res_II_self, res_II_cross,
                    res_II_cross / max(res_I, 1e-15))
        logger.info("    Per-plane PSNR: %s",
                    "  ".join(f"z{k}={p:.2f}" for k, p in enumerate(plane_psnr_II)))

        # ---- Scenario III: autofocus calibration via sharpness grid search ----
        logger.info("  Scenario III: autofocus calibration (grid search)...")
        t0 = time.time()
        calibrated_err, recon_III = autofocus_grid_search(
            hologram_noisy,
            base_kwargs,
            search_range_um=(-250.0, 250.0),
            n_search=51,
            lam_tv=lam_tv,
            fista_iters=50,
        )
        t_III = time.time() - t0
        psnr_III = psnr(phantom, recon_III)
        ssim_III = ssim_simple(phantom.ravel(), recon_III.ravel())

        # Build calibrated operator for residual measurement
        kwargs_cal = dict(base_kwargs)
        kwargs_cal["prop_distance_error_um"] = calibrated_err
        op_cal = CompressiveHolographyOperator(**kwargs_cal)
        res_III = hologram_residual(op_cal, hologram_noisy, recon_III)

        plane_psnr_III = [psnr(phantom[k], recon_III[k]) for k in range(n_depths)]
        logger.info("    Calibrated error: %.2f um  (true error was +%.1f um)",
                    calibrated_err, prop_err)
        logger.info("    PSNR=%.2f dB  SSIM=%.4f  residual=%.6f  (%.1fs)",
                    psnr_III, ssim_III, res_III, t_III)
        logger.info("    Per-plane PSNR: %s",
                    "  ".join(f"z{k}={p:.2f}" for k, p in enumerate(plane_psnr_III)))

        # ---- Recovery ratio ----
        if abs(psnr_I - psnr_II) > 0.01:
            recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II)
        else:
            recovery = float("nan")
        logger.info("    Recovery ratio: %.4f", recovery)

        results["mismatch_experiments"].append({
            "prop_distance_error_um": prop_err,
            "scenario_II": {
                "psnr_db": psnr_II,
                "ssim": ssim_II,
                "delta_psnr_db": delta_II,
                "self_residual": res_II_self,
                "cross_residual": res_II_cross,
                "cross_ratio": float(res_II_cross / max(res_I, 1e-15)),
                "plane_psnr_db": plane_psnr_II,
                "time_s": t_II,
            },
            "scenario_III": {
                "calibrated_error_um": calibrated_err,
                "psnr_db": psnr_III,
                "ssim": ssim_III,
                "residual": res_III,
                "plane_psnr_db": plane_psnr_III,
                "time_s": t_III,
            },
            "recovery_ratio": recovery,
        })

    # ---- Summary table ----
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Compressive Holography 4-Scenario Results")
    logger.info("=" * 60)
    logger.info("  %-12s  %10s  %10s  %10s  %10s",
                "Error (um)", "PSNR I", "PSNR II", "PSNR III", "Recovery")
    logger.info("  " + "-" * 56)
    for exp in results["mismatch_experiments"]:
        logger.info("  %-12.0f  %10.2f  %10.2f  %10.2f  %10.4f",
                    exp["prop_distance_error_um"],
                    psnr_I,
                    exp["scenario_II"]["psnr_db"],
                    exp["scenario_III"]["psnr_db"],
                    exp["recovery_ratio"])

    # Cross-residual diagnostic
    logger.info("\n  Cross-residual diagnostic (mismatch detectable when ratio >> 1):")
    logger.info("  %-12s  %12s  %12s  %10s",
                "Error (um)", "Self-res II", "Cross-res II", "Ratio")
    logger.info("  " + "-" * 50)
    for exp in results["mismatch_experiments"]:
        s2 = exp["scenario_II"]
        logger.info("  %-12.0f  %12.6f  %12.6f  %10.2fx",
                    exp["prop_distance_error_um"],
                    s2["self_residual"],
                    s2["cross_residual"],
                    s2["cross_ratio"])

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    results = run_compholo_4scenario()

    out_path = RESULTS_DIR / "compholo_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
