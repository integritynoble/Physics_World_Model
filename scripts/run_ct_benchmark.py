#!/usr/bin/env python3
"""CT Benchmark Runner — CPU Methods (FBP, TV-ADMM, PnP-ADMM).

Runs all three algorithms on all three tiers (public, dev, hidden) using the
actual challenge HDF5 files and the fan-beam geometry from generate_dataset.py.

Scoring (as on the benchmark page):
    score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
    PSNR_norm = min(1, (PSNR - 10) / 40)
    Consistency = 1 - ||y - H_nom(x_hat)||₂ / ||y||₂

Usage:
    cd /path/to/Physics_World_Model
    python scripts/run_ct_benchmark.py                          # all tiers, all algos
    python scripts/run_ct_benchmark.py --tier public            # public only
    python scripts/run_ct_benchmark.py --algo fbp               # FBP only
    python scripts/run_ct_benchmark.py --tier public --algo fbp

Output:
    results/ct/benchmark_cpu_<timestamp>.json
    results/ct/benchmark_cpu_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
from skimage.metrics import structural_similarity

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "benchmark" / "ct"
RESULTS_DIR = PROJECT_ROOT / "results" / "ct"

# ── CT geometry (must match generate_dataset.py) ──────────────────────────────

IMAGE_SIZE  = 362
D_SO        = 800.0    # source-to-isocenter, px
D_SD        = 568.0    # isocenter-to-detector, px
N_DET       = 736      # detector channels
DET_SPACING = 1.496    # detector pitch, px
MU_SCALE    = 0.058    # pixel-density → nepers
D_SSD       = D_SO + D_SD   # source-to-detector total, px  (1368.0)
# FBP calibration: empirically measured, D_SSD Feldkamp pre-weight + Hann + π/(2N)
FBP_CAL     = 1.655
# TV-ADMM / PnP parameters
N_STEPS_FWD = 300      # ray-sampling steps (0.40 s/scene at 60 views)

# ══════════════════════════════════════════════════════════════════════════════
# Fan-beam operators
# ══════════════════════════════════════════════════════════════════════════════

def fan_beam_project(
    x: np.ndarray,
    angles_rad: np.ndarray,
    center_offset: float = 0.0,
    n_det: int = N_DET,
    D_so: float = D_SO,
    D_sd: float = D_SD,
    det_spacing: float = DET_SPACING,
    n_steps: int = N_STEPS_FWD,
) -> np.ndarray:
    """Fan-beam line-integral projection.  Returns sinogram in pixel-density units."""
    H, W = x.shape
    x64 = x.astype(np.float64)
    cy  = H / 2.0
    cx  = W / 2.0 + center_offset

    det_pos = (np.arange(n_det) - n_det / 2.0) * det_spacing
    diag    = np.sqrt(H ** 2 + W ** 2)
    n_steps = max(n_steps, int(diag * 1.2))
    t_vals  = np.linspace(0.0, 1.0, n_steps)

    sinogram = np.zeros((len(angles_rad), n_det), dtype=np.float32)

    for i, angle in enumerate(angles_rad):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        src_y = -D_so * sin_a + cy
        src_x =  D_so * cos_a + cx
        det_y =  D_sd * sin_a + det_pos * cos_a + cy
        det_x = -D_sd * cos_a + det_pos * sin_a + cx

        ray_y = det_y - src_y
        ray_x = det_x - src_x
        ray_len = np.sqrt(ray_y ** 2 + ray_x ** 2)

        sample_y = src_y + np.outer(ray_y, t_vals)
        sample_x = src_x + np.outer(ray_x, t_vals)
        coords   = np.array([sample_y.ravel(), sample_x.ravel()])
        vals     = map_coordinates(x64, coords, order=1, mode="constant", cval=0.0)
        vals     = vals.reshape(n_det, n_steps)

        step_size     = ray_len / n_steps
        sinogram[i]   = (vals.sum(axis=1) * step_size).astype(np.float32)

    return sinogram


def fan_beam_backproject(
    sino: np.ndarray,
    angles_rad: np.ndarray,
    image_size: int = IMAGE_SIZE,
    D_so: float = D_SO,
    D_sd: float = D_SD,
    n_det: int = N_DET,
    det_spacing: float = DET_SPACING,
) -> np.ndarray:
    """Adjoint of fan_beam_project.  Returns image in pixel-density units.

    Derived from the fan-beam geometry: for each pixel (py, px) and angle θ,
    the detector index is t = (D_so+D_sd) * V / U  (see inline derivation).
    """
    H = W = image_size
    n_views = len(angles_rad)

    # Centered pixel coordinates
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    py = yy - H / 2.0   # (H, W)
    px = xx - W / 2.0   # (H, W)

    img = np.zeros((H, W), dtype=np.float64)

    for i, angle in enumerate(angles_rad):
        cos_a, sin_a = np.cos(float(angle)), np.sin(float(angle))

        # Source in centred coordinates
        sy = -D_so * sin_a
        sx =  D_so * cos_a

        # Source → pixel vectors
        vy = py - sy   # (H, W)
        vx = px - sx   # (H, W)

        # U = projection of (vy, vx) onto source→isocenter direction (sin_a, -cos_a)
        # Equivalently: U = D_so + py*sin_a - px*cos_a
        U = vy * sin_a - vx * cos_a   # (H, W)

        # V = projection onto detector direction (cos_a, sin_a)
        V = vy * cos_a + vx * sin_a   # (H, W)

        # Detector physical offset from centre, then convert to index
        det_j_val = (D_so + D_sd) * V / np.where(U > 1e-6, U, 1e-6)
        det_idx   = det_j_val / det_spacing + n_det / 2.0   # fractional index (H, W)

        # Bilinear interpolation of sino[i] at det_idx
        d0    = np.floor(det_idx).astype(int)
        d1    = d0 + 1
        frac  = det_idx - d0

        valid = (U > 0) & (d0 >= 0) & (d1 < n_det)
        d0c   = np.clip(d0, 0, n_det - 2)
        d1c   = d0c + 1

        row   = sino[i]   # (n_det,)
        val   = (1.0 - frac) * row[d0c] + frac * row[d1c]   # (H, W)
        img  += np.where(valid, val, 0.0)

    return (img / n_views).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 1: Fan-beam Filtered Backprojection (FBP)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_sino(sino_nepers: np.ndarray) -> np.ndarray:
    """Clamp saturated rays and smooth along detector axis.

    Rays with I_expect ≈ 0 photons are clamped by the simulator to I=1,
    giving sino_meas = log(I0) = 9.21 nepers.  These saturated values
    create catastrophic FBP streaks; clamping to 6.5 nepers (≈ 99.5th
    percentile of valid physics) and smoothing with σ=1 px removes them.
    """
    sino_pp = np.clip(sino_nepers.astype(np.float64), -0.1, 6.5)
    return gaussian_filter1d(sino_pp, sigma=1.0, axis=1)


def fbp(sino_nepers: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Fan-beam FBP using Feldkamp pre-weight, Ram-Lak/Hann filter + backprojection.

    Input:  sinogram in nepers  (n_views, n_det)
    Output: reconstructed image in [0, 1] pixel-density units  (H, W)

    Key calibration choices (empirically validated on LoDoPaB-CT):
      - D_SSD Feldkamp pre-weight: cos_β = D_SSD / √(D_SSD²+j²)
      - Hann-windowed ramp filter
      - FBP_CAL = 1.655  (corrects discrete backprojection normalisation)
      - Sinogram preprocessed: clamp [-0.1, 6.5] nepers + σ=1 Gaussian smooth
    """
    # Preprocessing: suppress saturated rays
    sino_pp = preprocess_sino(sino_nepers)   # (n_views, n_det), float64

    # Convert nepers → pixel-density (undo MU_SCALE)
    sino = sino_pp / MU_SCALE   # (n_views, n_det)
    n_views, n_det = sino.shape

    # Detector positions
    det_j = (np.arange(n_det) - n_det / 2.0) * DET_SPACING   # (n_det,)

    # ── Feldkamp cosine pre-weighting (correct form uses D_SSD, not D_SD) ──
    cos_beta = D_SSD / np.sqrt(D_SSD ** 2 + det_j ** 2)   # (n_det,)
    sino_w   = sino * cos_beta[np.newaxis, :]               # (n_views, n_det)

    # ── Ram-Lak (ramp) filter + Hann window ──────────────────────────────
    n_fft   = int(2 ** np.ceil(np.log2(2 * n_det)))
    freq    = np.fft.rfftfreq(n_fft, d=DET_SPACING)        # cycles/px
    ramp    = 2.0 * np.abs(freq)
    hann    = 0.5 + 0.5 * np.cos(np.pi * freq / freq.max())
    kernel  = ramp * hann

    sino_f        = np.fft.rfft(sino_w, n=n_fft, axis=1)
    sino_f       *= kernel[np.newaxis, :]
    sino_filtered = np.fft.irfft(sino_f, n=n_fft, axis=1)[:, :n_det]   # (n_views, n_det)

    # ── Fan-beam backprojection with 1/U² weight ──────────────────────────
    H = W = IMAGE_SIZE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    py = yy - H / 2.0
    px = xx - W / 2.0

    recon = np.zeros((H, W), dtype=np.float64)

    for i, angle in enumerate(angles_rad):
        angle = float(angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        sy = -D_SO * sin_a
        sx =  D_SO * cos_a
        vy = py - sy
        vx = px - sx
        U  = vy * sin_a - vx * cos_a   # (H, W)
        V  = vy * cos_a + vx * sin_a   # (H, W)

        det_j_val = D_SSD * V / np.where(U > 1e-6, U, 1e-6)
        det_idx   = det_j_val / DET_SPACING + n_det / 2.0

        valid = (U > 0) & (det_idx >= 0) & (det_idx < n_det - 1)
        d0    = np.floor(det_idx).astype(int)
        d0    = np.clip(d0, 0, n_det - 2)
        d1    = d0 + 1
        frac  = det_idx - d0

        row = sino_filtered[i]
        val = (1.0 - frac) * row[d0] + frac * row[d1]
        val = np.where(valid, val, 0.0)

        # FBP weight: (D_so / U)^2
        weight = (D_SO / np.where(U > 1e-6, U, 1e-6)) ** 2
        recon += weight * val

    # Normalisation: π / (2 * n_views) + empirical calibration factor
    recon *= np.pi / (2.0 * n_views) * FBP_CAL
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TV proximal (Chambolle's dual algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def tv_prox(v: np.ndarray, lam: float, n_iter: int = 30) -> np.ndarray:
    """Proximal operator for λ·TV: argmin_z λ·TV(z) + 0.5·||z-v||².

    Uses Chambolle's dual gradient ascent (Chambolle 2004).
    """
    tau = 0.25   # dual step size (safe for L=4 of div)
    p   = np.zeros((2, *v.shape), dtype=np.float64)

    for _ in range(n_iter):
        # Compute gradient of current primal estimate
        z  = v - lam * _div(p)
        gz = _grad(z)
        # Update dual
        p_new = p + tau * gz
        # Project dual onto unit ball  |p| ≤ 1  per pixel
        norm = np.maximum(1.0, np.sqrt(p_new[0] ** 2 + p_new[1] ** 2))
        p    = p_new / norm[np.newaxis, ...]

    return v - lam * _div(p)


def _grad(u: np.ndarray) -> np.ndarray:
    """Forward finite difference gradient, zero boundary."""
    gy = np.zeros_like(u)
    gx = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return np.array([gy, gx])


def _div(p: np.ndarray) -> np.ndarray:
    """Backward finite difference divergence (adjoint of _grad)."""
    gy, gx = p[0], p[1]
    dy      = np.zeros_like(gy)
    dx      = np.zeros_like(gx)
    dy[1:, :]  = gy[1:, :]  - gy[:-1, :]
    dy[0,  :]  = gy[0,  :]
    dy[-1, :]  = -gy[-2, :]
    dx[:, 1:]  = gx[:, 1:]  - gx[:, :-1]
    dx[:,  0]  = gx[:,  0]
    dx[:, -1]  = -gx[:, -2]
    return dy + dx


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 2: TV-ADMM (gradient-descent + TV proximal)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_step(angles_rad: np.ndarray) -> float:
    """Estimate Lipschitz constant L = MU_SCALE² · ||AᵀA|| via 4-step power iteration."""
    rng = np.random.default_rng(0)
    _u  = rng.standard_normal((IMAGE_SIZE, IMAGE_SIZE))
    _u /= np.linalg.norm(_u)
    nrm = 1.0
    for _ in range(4):
        Au  = fan_beam_project(_u.astype(np.float32), angles_rad).astype(np.float64) * MU_SCALE
        _u  = fan_beam_backproject(Au.astype(np.float32), angles_rad).astype(np.float64) * MU_SCALE
        nrm = float(np.linalg.norm(_u))
        if nrm > 0:
            _u /= nrm
    return 1.0 / max(nrm, 1e-10)


def tv_admm(
    sino_nepers: np.ndarray,
    angles_rad: np.ndarray,
    n_iter: int = 20,
    lam: float = 0.05,
    step: float | None = None,
) -> np.ndarray:
    """TV-regularised reconstruction via gradient descent + TV proximal (ISTA-TV).

    min_x  0.5·||A(x)·MU - y||² + λ·TV(x)

    Sinogram is preprocessed (clamp + smooth) to suppress saturated rays.
    """
    y = preprocess_sino(sino_nepers)   # (n_views, n_det), float64, nepers

    if step is None:
        step = _estimate_step(angles_rad)

    # Initialise with FBP
    x = fbp(sino_nepers, angles_rad).astype(np.float64)

    for k in range(n_iter):
        # Gradient of data fidelity: MU · Aᵀ(A(x)·MU − y)
        Ax   = fan_beam_project(x.astype(np.float32), angles_rad).astype(np.float64) * MU_SCALE
        res  = Ax - y                                                   # residual (nepers)
        grad = fan_beam_backproject((res * MU_SCALE).astype(np.float32), angles_rad).astype(np.float64)
        x   -= step * grad

        # TV proximal
        x = tv_prox(x, lam * step, n_iter=20)
        x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 3: PnP-ADMM (gradient-descent + Gaussian denoiser)
# ══════════════════════════════════════════════════════════════════════════════

def pnp_admm(
    sino_nepers: np.ndarray,
    angles_rad: np.ndarray,
    n_iter: int = 15,
    sigma_start: float = 0.06,
    sigma_end: float = 0.01,
    step: float | None = None,
) -> np.ndarray:
    """PnP-ADMM with Gaussian denoiser (sigma schedule: sigma_start → sigma_end).

    min_x  0.5·||A(x)·MU − y||²  s.t.  x = D_σ(x)
    (Plug-and-Play: replace TV prox with Gaussian filter denoiser)

    Sinogram is preprocessed (clamp + smooth) to suppress saturated rays.
    """
    y = preprocess_sino(sino_nepers)   # (n_views, n_det), float64, nepers

    if step is None:
        step = _estimate_step(angles_rad)

    # Initialise with FBP
    x = fbp(sino_nepers, angles_rad).astype(np.float64)

    sigma_schedule = np.linspace(sigma_start, sigma_end, n_iter)

    for k, sigma in enumerate(sigma_schedule):
        # Gradient step
        Ax   = fan_beam_project(x.astype(np.float32), angles_rad).astype(np.float64) * MU_SCALE
        res  = Ax - y
        grad = fan_beam_backproject((res * MU_SCALE).astype(np.float32), angles_rad).astype(np.float64)
        x   -= step * grad

        # Denoising step (Gaussian denoiser, sigma in pixel units)
        x = gaussian_filter(x, sigma=sigma * IMAGE_SIZE)
        x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 4: LS-GD (unregularized gradient descent toward LS minimum)
# ══════════════════════════════════════════════════════════════════════════════


def ls_gd(
    sino_nepers: np.ndarray,
    angles_rad: np.ndarray,
    n_iter: int = 30,
    step: float | None = None,
) -> np.ndarray:
    """Unregularized gradient descent on 0.5·||A(x)·MU − y||².

    Initialized from FBP; converges toward the least-squares minimum which,
    for moderate Poisson noise, is closer to x_true than FBP.
    Preprocessed sinogram is used (clamp + smooth) to suppress saturation.
    """
    y = preprocess_sino(sino_nepers)   # (n_views, n_det), float64, nepers

    if step is None:
        step = _estimate_step(angles_rad)

    x = fbp(sino_nepers, angles_rad).astype(np.float64)

    for _ in range(n_iter):
        Ax   = fan_beam_project(x.astype(np.float32), angles_rad).astype(np.float64) * MU_SCALE
        res  = Ax - y
        grad = fan_beam_backproject((res * MU_SCALE).astype(np.float32), angles_rad).astype(np.float64)
        x    = np.clip(x - step * grad, 0.0, 1.0)

    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(structural_similarity(
        x_hat.astype(np.float32),
        x_true.astype(np.float32),
        data_range=1.0,
    ))


def compute_consistency(
    x_hat: np.ndarray,
    sino_measured: np.ndarray,
    angles_rad: np.ndarray,
) -> float:
    """Consistency = 1 - ||y - H_nom(x_hat)||₂ / ||y||₂.

    H_nom uses nominal angles (no mismatch correction) and MU_SCALE.
    """
    y_norm = float(np.linalg.norm(sino_measured))
    if y_norm < 1e-8:
        return 0.0
    y_hat  = fan_beam_project(x_hat, angles_rad).astype(np.float64) * MU_SCALE
    diff   = float(np.linalg.norm(y_hat - sino_measured.astype(np.float64)))
    return float(1.0 - diff / y_norm)


def composite_score(psnr: float, ssim: float, consistency: float) -> float:
    psnr_norm = min(1.0, (psnr - 10.0) / 40.0)
    return 0.4 * psnr_norm + 0.4 * ssim + 0.2 * consistency


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

ALGO_FNS = {
    "fbp":       lambda sino, ang: fbp(sino, ang),
    "ls_gd":     lambda sino, ang: ls_gd(sino, ang),
    "tv_admm":   lambda sino, ang: tv_admm(sino, ang),
    "pnp_admm":  lambda sino, ang: pnp_admm(sino, ang),
}

TIERS = ["public", "dev", "hidden"]


def run_tier(tier: str, algos: list[str]) -> list[dict]:
    h5_path = DATA_ROOT / tier / f"ct_challenge_{tier}.h5"
    if not h5_path.exists():
        print(f"  [SKIP] {h5_path} not found")
        return []

    rows = []
    print(f"\n{'='*70}")
    print(f"  Tier: {tier.upper()}  →  {h5_path.name}")
    print(f"{'='*70}")

    with h5py.File(h5_path, "r") as f:
        scene_keys = sorted(f.keys())
        for sk in scene_keys:
            grp           = f[sk]
            x_true        = grp["x_true"][()].astype(np.float32)
            sino_measured = grp["sinogram_measured"][()].astype(np.float32)
            angles_rad    = grp["angles_nominal"][()].astype(np.float64)
            meta          = json.loads(grp.attrs.get("metadata", "{}"))
            scene_name    = meta.get("scene", sk)
            n_views       = int(len(angles_rad))

            print(f"\n  Scene {sk}  ({scene_name})  n_views={n_views}")
            print(f"    x_true range=[{x_true.min():.3f},{x_true.max():.3f}]  "
                  f"sino_meas range=[{sino_measured.min():.3f},{sino_measured.max():.3f}]")

            for algo in algos:
                t0 = time.time()
                try:
                    x_hat = ALGO_FNS[algo](sino_measured, angles_rad)
                except Exception as exc:
                    print(f"    [{algo}] ERROR: {exc}")
                    continue
                elapsed = time.time() - t0

                psnr  = compute_psnr(x_hat, x_true)
                ssim  = compute_ssim(x_hat, x_true)
                cons  = compute_consistency(x_hat, sino_measured, angles_rad)
                score = composite_score(psnr, ssim, cons)

                print(f"    [{algo:10s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                      f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")

                rows.append({
                    "tier":        tier,
                    "scene":       sk,
                    "scene_name":  scene_name,
                    "algo":        algo,
                    "n_views":     n_views,
                    "psnr_db":     round(psnr, 4),
                    "ssim":        round(ssim, 4),
                    "consistency": round(cons, 4),
                    "score":       round(score, 4),
                    "time_s":      round(elapsed, 2),
                })

    return rows


def print_summary(all_rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY — mean metrics per (tier, algo)")
    print("=" * 70)
    print(f"{'tier':8s}  {'algo':12s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Cons':>6s}  {'Score':>6s}  {'t(s)':>6s}")
    print("-" * 70)
    from collections import defaultdict
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (tier, algo), rows in sorted(acc.items()):
        psnr  = sum(r["psnr_db"]      for r in rows) / len(rows)
        ssim  = sum(r["ssim"]          for r in rows) / len(rows)
        cons  = sum(r["consistency"]   for r in rows) / len(rows)
        score = sum(r["score"]         for r in rows) / len(rows)
        t     = sum(r["time_s"]        for r in rows) / len(rows)
        print(f"{tier:8s}  {algo:12s}  {psnr:7.2f}  {ssim:6.4f}  {cons:6.4f}  {score:6.4f}  {t:6.1f}")
    print("=" * 70)

    # Also print per-algo summary across all tiers
    print("\nSUMMARY — mean across all tiers per algo")
    print("-" * 70)
    algo_acc: dict = defaultdict(list)
    for r in all_rows:
        algo_acc[r["algo"]].append(r)
    for algo, rows in sorted(algo_acc.items()):
        psnr  = sum(r["psnr_db"]      for r in rows) / len(rows)
        ssim  = sum(r["ssim"]          for r in rows) / len(rows)
        score = sum(r["score"]         for r in rows) / len(rows)
        print(f"  {algo:12s}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  Score={score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CT Benchmark — CPU methods")
    parser.add_argument("--tier",  nargs="+", default=TIERS,
                        choices=TIERS + ["all"],
                        help="Tier(s) to evaluate (default: all)")
    parser.add_argument("--algo",  nargs="+", default=list(ALGO_FNS.keys()),
                        choices=list(ALGO_FNS.keys()) + ["all"],
                        help="Algorithm(s) to run (default: all)")
    args = parser.parse_args()

    tiers = TIERS if "all" in args.tier else args.tier
    algos = list(ALGO_FNS.keys()) if "all" in args.algo else args.algo

    print("CT Benchmark — CPU Methods")
    print(f"Tiers: {tiers}  |  Algos: {algos}")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}  D_SO={D_SO}  D_SD={D_SD}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"benchmark_cpu_{ts}.json"
    out_csv  = RESULTS_DIR / f"benchmark_cpu_{ts}.csv"

    all_rows: list[dict] = []
    t_start = time.time()

    for tier in tiers:
        rows = run_tier(tier, algos)
        all_rows.extend(rows)

    elapsed_total = time.time() - t_start

    # Save JSON
    result_doc = {
        "timestamp":     ts,
        "tiers":         tiers,
        "algos":         algos,
        "total_time_s":  round(elapsed_total, 1),
        "scenes":        all_rows,
    }
    with open(out_json, "w") as fj:
        json.dump(result_doc, fj, indent=2)
    print(f"\nResults saved → {out_json}")

    # Save CSV
    if all_rows:
        with open(out_csv, "w", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV saved     → {out_csv}")

    print_summary(all_rows)
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
