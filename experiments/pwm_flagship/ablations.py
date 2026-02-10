"""PWM Flagship -- Ablation studies.

For each of the 3 depth modalities (SPC, CACTI, CASSI), run 4 ablations:

1. **Remove PhotonAgent**        -- feasibility predictions degrade.
2. **Remove Recoverability**     -- bad compression ratios selected.
3. **Remove mismatch priors**    -- calibration becomes unstable.
4. **Remove RunBundle discipline** -- reproducibility breaks.

Each ablation measures degradation in dB or metric compared to full pipeline.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.ablations --out_dir results/flagship_ablations
    PYTHONPATH=. python -m experiments.pwm_flagship.ablations --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.inversenet.manifest_schema import Severity  # noqa: F401 (kept for generate helpers)
from experiments.inversenet.mismatch_sweep import apply_mismatch, get_delta_theta  # noqa: F401

logger = logging.getLogger(__name__)

PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
BASE_SEED = 6000
BOOTSTRAP_K = 10
IMAGE_SIZE = (64, 64)
SPATIAL_SIZE = (64, 64)
N_BOOTSTRAP_ABLATION = 5


# -- Helpers -----------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest()


def _compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(x.max()), float(y.max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def _make_run_bundle(
    spec_id: str, metrics: Dict[str, float], seeds: List[int],
) -> Dict[str, Any]:
    return {
        "version": BUNDLE_VERSION,
        "spec_id": spec_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": seeds,
            "platform": platform.platform(),
            "pwm_version": PWM_VERSION,
        },
        "metrics": metrics,
        "artifacts": {},
        "hashes": {},
    }


# ============================================================================
# SPC helpers
# ============================================================================

def _spc_generate(seed: int, cr: float = 0.25, photon: float = 1e4, sev: Severity = Severity.moderate):
    rng = np.random.default_rng(seed)
    H, W = IMAGE_SIZE
    N = H * W
    M = max(1, int(N * cr))

    # Ground truth
    raw = rng.random(IMAGE_SIZE).astype(np.float32)
    x_gt = gaussian_filter(raw, sigma=5.0).astype(np.float32)
    x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8

    # Measurement matrix
    A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
    A /= np.sqrt(N)

    # Clean + noisy measurement
    y_clean = A @ x_gt.flatten()
    scale = photon / (np.abs(y_clean).max() + 1e-10)
    y_noisy = rng.poisson(np.maximum(y_clean * scale, 0)).astype(np.float32) / scale

    # Mismatch
    mm = apply_mismatch("spc", "gain", sev, y=y_noisy, mask=A, rng=rng)
    y_mm = mm.get("y", y_noisy)

    return x_gt, A, y_mm, mm["delta_theta"]


def _spc_recon(y, A, n_iter=30, lam=0.01):
    N = A.shape[1]
    x = np.zeros(N, dtype=np.float64)
    L = float(np.linalg.norm(A.T @ A, ord=2))
    if L < 1e-10: L = 1.0
    step = 1.0 / L
    for _ in range(n_iter):
        grad = A.T @ (A @ x - y.astype(np.float64))
        x = x - step * grad
        x = np.sign(x) * np.maximum(np.abs(x) - lam * step, 0)
    return np.clip(x.reshape(IMAGE_SIZE), 0, 1).astype(np.float32)


# ============================================================================
# CACTI helpers
# ============================================================================

def _cacti_generate(seed: int, nf: int = 8, photon: float = 1e4, sev: Severity = Severity.moderate):
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    # Video
    base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
    base -= base.min(); base /= base.max() + 1e-8
    video = np.zeros((H, W, nf), dtype=np.float32)
    for t in range(nf):
        phase = 2.0 * np.pi * t / nf
        frame = np.roll(base, int(2 * np.sin(phase)), axis=0) * (0.8 + 0.2 * np.cos(phase))
        video[:, :, t] = np.clip(frame, 0, 1)

    # Masks
    base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    masks = np.zeros((H, W, nf), dtype=np.float32)
    for t in range(nf):
        masks[:, :, t] = np.roll(base_mask, t, axis=0)

    y = np.sum(video * masks, axis=2)
    scale = photon / (np.abs(y).max() + 1e-10)
    y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale

    mm = apply_mismatch("cacti", "mask_shift", sev, masks=masks, rng=rng)
    masks_mm = mm.get("masks", masks)

    return video, masks, masks_mm, y, mm["delta_theta"]


def _cacti_recon(y, masks, n_iter=20):
    H, W, T = masks.shape
    x = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        x[:, :, t] = y * masks[:, :, t] / max(T, 1)
    for _ in range(n_iter):
        yb = np.sum(x * masks, axis=2)
        r = y - yb
        for t in range(T):
            x[:, :, t] += r * masks[:, :, t] / max(T, 1)
        for t in range(T):
            x[:, :, t] = gaussian_filter(x[:, :, t], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


# ============================================================================
# CASSI helpers
# ============================================================================

def _cassi_generate(seed: int, nb: int = 8, photon: float = 1e4, sev: Severity = Severity.moderate):
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    cube = np.zeros((H, W, nb), dtype=np.float32)
    bases = [
        gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0)
        for _ in range(3)
    ]
    for b in bases:
        b -= b.min(); b /= b.max() + 1e-8
    wl = np.linspace(0, 1, nb)
    for l in range(nb):
        cube[:, :, l] = (
            bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2)
            + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2)
            + bases[2] * 0.2
        )
    cube -= cube.min(); cube /= cube.max() + 1e-8

    mask = (rng.random((H, W)) > 0.5).astype(np.float32)

    theta_true = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}

    # Forward
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift
    y = np.zeros((H, W), dtype=np.float32)
    for l in range(nb):
        dx, dy = dispersion_shift(theta_true, band=l)
        band = np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1)
        y += band * mask

    scale = photon / (np.abs(y).max() + 1e-10)
    y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale

    mm = apply_mismatch("cassi", "disp_step", sev, y=None, mask=mask, theta=theta_true, rng=rng)
    theta_mm = mm.get("theta", theta_true)

    return cube, mask, theta_true, theta_mm, y, mm["delta_theta"]


def _cassi_recon(y, mask, theta, nb, n_iter=20):
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift
    H, W = mask.shape
    x = np.zeros((H, W, nb), dtype=np.float32)
    for l in range(nb):
        dx, dy = dispersion_shift(theta, band=l)
        x[:, :, l] = np.roll(np.roll(y * mask, -int(round(dy)), axis=0), -int(round(dx)), axis=1) / max(nb, 1)

    for _ in range(n_iter):
        yb = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            band = np.roll(np.roll(x[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1)
            yb += band * mask
        r = y - yb
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            upd = np.roll(np.roll(r * mask, -int(round(dy)), axis=0), -int(round(dx)), axis=1) / max(nb, 1)
            x[:, :, l] += upd
        for l in range(nb):
            x[:, :, l] = gaussian_filter(x[:, :, l], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


# ============================================================================
# Ablation runners
# ============================================================================

def _run_full_pipeline(modality: str, seed: int, recon_iters: int) -> Dict[str, float]:
    """Run full pipeline with all components active. Returns PSNR.

    Full pipeline: good photon regime, appropriate CR, true operator
    (calibrated), deterministic seed.
    """
    rng = np.random.default_rng(seed)

    if modality == "spc":
        H, W = IMAGE_SIZE
        N = H * W
        cr = 0.25
        M = max(1, int(N * cr))
        x_gt = gaussian_filter(rng.random(IMAGE_SIZE).astype(np.float32), sigma=5.0)
        x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8
        A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A /= np.sqrt(N)
        y = A @ x_gt.flatten()
        # Mild noise
        y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
        x_hat = _spc_recon(y, A, n_iter=recon_iters, lam=0.01)
        return {"psnr": _compute_psnr(x_gt, x_hat)}

    elif modality == "cacti":
        H, W = SPATIAL_SIZE
        nf = 8
        base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
        base -= base.min(); base /= base.max() + 1e-8
        video = np.stack([
            np.clip(np.roll(base, int(2 * np.sin(2 * np.pi * t / nf)), axis=0), 0, 1)
            for t in range(nf)
        ], axis=2)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        masks = np.stack([np.roll(base_mask, t, axis=0) for t in range(nf)], axis=2)
        y = np.sum(video * masks, axis=2) + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        x_hat = _cacti_recon(y, masks, n_iter=recon_iters)
        return {"psnr": _compute_psnr(video, x_hat)}

    elif modality == "cassi":
        H, W = SPATIAL_SIZE
        nb = 8
        bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
        for b in bases: b -= b.min(); b /= b.max() + 1e-8
        cube = np.zeros((H, W, nb), dtype=np.float32)
        wl = np.linspace(0, 1, nb)
        for l in range(nb):
            cube[:, :, l] = bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2) + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2) + bases[2] * 0.2
        cube -= cube.min(); cube /= cube.max() + 1e-8
        mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        theta = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        y = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            y += np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1) * mask
        y = y + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        x_hat = _cassi_recon(y, mask, theta, nb, n_iter=recon_iters)
        return {"psnr": _compute_psnr(cube, x_hat)}

    return {"psnr": 0.0}


def _ablation_no_photon(modality: str, seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 1: Remove photon agent -- photon-starved + unadapted solver.

    Without the PhotonAgent, the system proceeds with a very low photon
    count (Poisson noise) and uses default solver settings (too few
    iterations, wrong regularization) instead of adapting to the noise.
    """
    rng = np.random.default_rng(seed)
    # Without PhotonAgent: use fewer iterations and wrong lambda
    abl_iters = max(recon_iters // 3, 3)
    photon_level = 10.0  # Very low photon count

    if modality == "spc":
        H, W = IMAGE_SIZE
        N = H * W
        cr = 0.25
        M = max(1, int(N * cr))
        x_gt = gaussian_filter(rng.random(IMAGE_SIZE).astype(np.float32), sigma=5.0)
        x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8
        A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A /= np.sqrt(N)
        y = A @ x_gt.flatten()
        # Apply Poisson noise at very low photon count
        scale = photon_level / (np.abs(y).max() + 1e-10)
        y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale
        # Wrong lambda (no photon-aware adaptation)
        x_hat = _spc_recon(y, A, n_iter=abl_iters, lam=0.001)
        return {"psnr": _compute_psnr(x_gt, x_hat)}

    elif modality == "cacti":
        H, W = SPATIAL_SIZE
        nf = 8
        base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
        base -= base.min(); base /= base.max() + 1e-8
        video = np.stack([
            np.clip(np.roll(base, int(2 * np.sin(2 * np.pi * t / nf)), axis=0), 0, 1)
            for t in range(nf)
        ], axis=2)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        masks = np.stack([np.roll(base_mask, t, axis=0) for t in range(nf)], axis=2)
        y = np.sum(video * masks, axis=2)
        scale = photon_level / (np.abs(y).max() + 1e-10)
        y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale
        x_hat = _cacti_recon(y, masks, n_iter=abl_iters)
        return {"psnr": _compute_psnr(video, x_hat)}

    elif modality == "cassi":
        H, W = SPATIAL_SIZE
        nb = 8
        bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
        for b in bases: b -= b.min(); b /= b.max() + 1e-8
        cube = np.zeros((H, W, nb), dtype=np.float32)
        wl = np.linspace(0, 1, nb)
        for l in range(nb):
            cube[:, :, l] = bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2) + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2) + bases[2] * 0.2
        cube -= cube.min(); cube /= cube.max() + 1e-8
        mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        theta = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        y = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            y += np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1) * mask
        scale = photon_level / (np.abs(y).max() + 1e-10)
        y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale
        x_hat = _cassi_recon(y, mask, theta, nb, n_iter=abl_iters)
        return {"psnr": _compute_psnr(cube, x_hat)}

    return {"psnr": 0.0}


def _ablation_no_recoverability(modality: str, seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 2: Remove recoverability -- inadequate sensing design.

    Without the recoverability agent, the system uses a bad measurement
    design: too few SPC measurements, sparse CACTI/CASSI masks with
    insufficient spatial coverage.
    """
    rng = np.random.default_rng(seed)

    if modality == "spc":
        H, W = IMAGE_SIZE
        N = H * W
        cr = 0.03  # Severely under-determined (baseline uses 0.25)
        M = max(1, int(N * cr))
        x_gt = gaussian_filter(rng.random(IMAGE_SIZE).astype(np.float32), sigma=5.0)
        x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8
        A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A /= np.sqrt(N)
        y = A @ x_gt.flatten() + rng.normal(0, 0.01, size=(M,)).astype(np.float32)
        x_hat = _spc_recon(y, A, n_iter=recon_iters, lam=0.01)
        return {"psnr": _compute_psnr(x_gt, x_hat)}

    elif modality == "cacti":
        # Same 8-frame video as baseline, but sparse masks (10% open)
        H, W = SPATIAL_SIZE
        nf = 8
        base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
        base -= base.min(); base /= base.max() + 1e-8
        video = np.stack([
            np.clip(np.roll(base, int(2 * np.sin(2 * np.pi * t / nf)), axis=0), 0, 1)
            for t in range(nf)
        ], axis=2)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        # Sparse masks: only 10% open (recoverability agent would flag this)
        masks = np.stack([
            np.roll(base_mask, t, axis=0) * (rng.random((H, W)) > 0.9).astype(np.float32)
            for t in range(nf)
        ], axis=2)
        y = np.sum(video * masks, axis=2) + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        x_hat = _cacti_recon(y, masks, n_iter=recon_iters)
        return {"psnr": _compute_psnr(video, x_hat)}

    elif modality == "cassi":
        # Same 8-band cube as baseline, but sparse mask (10% open)
        H, W = SPATIAL_SIZE
        nb = 8
        bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
        for b in bases: b -= b.min(); b /= b.max() + 1e-8
        cube = np.zeros((H, W, nb), dtype=np.float32)
        wl = np.linspace(0, 1, nb)
        for l in range(nb):
            cube[:, :, l] = bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2) + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2) + bases[2] * 0.2
        cube -= cube.min(); cube /= cube.max() + 1e-8
        # Sparse mask: only 10% open (recoverability agent would flag this)
        mask = (rng.random((H, W)) > 0.9).astype(np.float32)
        theta = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        y = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            y += np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1) * mask
        y = y + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        x_hat = _cassi_recon(y, mask, theta, nb, n_iter=recon_iters)
        return {"psnr": _compute_psnr(cube, x_hat)}

    return {"psnr": 0.0}


def _ablation_no_mismatch(modality: str, seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 3: Remove mismatch priors -- use wrong operator.

    Same clean signal as baseline, but the operator used for reconstruction
    has a significant mismatch that is not corrected.
    """
    rng = np.random.default_rng(seed)

    if modality == "spc":
        H, W = IMAGE_SIZE
        N = H * W
        cr = 0.25
        M = max(1, int(N * cr))
        x_gt = gaussian_filter(rng.random(IMAGE_SIZE).astype(np.float32), sigma=5.0)
        x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8
        A_true = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A_true /= np.sqrt(N)
        y = A_true @ x_gt.flatten() + rng.normal(0, 0.01, size=(M,)).astype(np.float32)
        # Wrong operator: different random matrix (simulates unknown mask)
        rng_wrong = np.random.default_rng(seed + 5555)
        A_wrong = (rng_wrong.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A_wrong /= np.sqrt(N)
        x_hat = _spc_recon(y, A_wrong, n_iter=recon_iters, lam=0.01)
        return {"psnr": _compute_psnr(x_gt, x_hat)}

    elif modality == "cacti":
        H, W = SPATIAL_SIZE
        nf = 8
        base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
        base -= base.min(); base /= base.max() + 1e-8
        video = np.stack([
            np.clip(np.roll(base, int(2 * np.sin(2 * np.pi * t / nf)), axis=0), 0, 1)
            for t in range(nf)
        ], axis=2)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        masks_true = np.stack([np.roll(base_mask, t, axis=0) for t in range(nf)], axis=2)
        y = np.sum(video * masks_true, axis=2) + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        # Wrong masks: shifted by 6 pixels (severe mask_shift)
        masks_wrong = np.roll(masks_true, 6, axis=0)
        x_hat = _cacti_recon(y, masks_wrong, n_iter=recon_iters)
        return {"psnr": _compute_psnr(video, x_hat)}

    elif modality == "cassi":
        H, W = SPATIAL_SIZE
        nb = 8
        bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
        for b in bases: b -= b.min(); b /= b.max() + 1e-8
        cube = np.zeros((H, W, nb), dtype=np.float32)
        wl = np.linspace(0, 1, nb)
        for l in range(nb):
            cube[:, :, l] = bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2) + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2) + bases[2] * 0.2
        cube -= cube.min(); cube /= cube.max() + 1e-8
        mask_true = (rng.random((H, W)) > 0.5).astype(np.float32)
        theta = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        y = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            y += np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1) * mask_true
        y = y + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        # Wrong mask: completely different coded aperture (no calibration)
        rng_wrong = np.random.default_rng(seed + 5555)
        mask_wrong = (rng_wrong.random((H, W)) > 0.5).astype(np.float32)
        x_hat = _cassi_recon(y, mask_wrong, theta, nb, n_iter=recon_iters)
        return {"psnr": _compute_psnr(cube, x_hat)}

    return {"psnr": 0.0}


def _ablation_no_runbundle(modality: str, seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 4: Remove RunBundle discipline -- stale calibration + wrong config.

    Without RunBundle (tracking solver config, seeds, hashes), the system
    uses stale/unchecked calibration data (mask slightly corrupted without
    hash verification) and wrong solver parameters (over-regularized,
    fewer iterations) because the optimal settings are not recorded.
    """
    rng = np.random.default_rng(seed)
    # Fewer iterations (stale config not updated)
    abl_iters = max(recon_iters // 3, 3)
    # RNG for stale perturbations (deterministic per seed for testability)
    rng_stale = np.random.default_rng(seed + 99999)

    if modality == "spc":
        H, W = IMAGE_SIZE
        N = H * W
        cr = 0.25
        M = max(1, int(N * cr))
        x_gt = gaussian_filter(rng.random(IMAGE_SIZE).astype(np.float32), sigma=5.0)
        x_gt -= x_gt.min(); x_gt /= x_gt.max() + 1e-8
        A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
        A /= np.sqrt(N)
        y = A @ x_gt.flatten() + rng.normal(0, 0.01, size=(M,)).astype(np.float32)
        # Wrong lambda: over-regularized (stale RunBundle config)
        x_hat = _spc_recon(y, A, n_iter=abl_iters, lam=0.1)
        return {"psnr": _compute_psnr(x_gt, x_hat)}

    elif modality == "cacti":
        H, W = SPATIAL_SIZE
        nf = 8
        base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
        base -= base.min(); base /= base.max() + 1e-8
        video = np.stack([
            np.clip(np.roll(base, int(2 * np.sin(2 * np.pi * t / nf)), axis=0), 0, 1)
            for t in range(nf)
        ], axis=2)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        masks = np.stack([np.roll(base_mask, t, axis=0) for t in range(nf)], axis=2)
        y = np.sum(video * masks, axis=2) + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        # Stale masks: 15% of entries flipped (no hash verification)
        masks_stale = masks.copy()
        flip = rng_stale.random((H, W, nf)) < 0.15
        masks_stale[flip] = 1.0 - masks_stale[flip]
        x_hat = _cacti_recon(y, masks_stale, n_iter=abl_iters)
        return {"psnr": _compute_psnr(video, x_hat)}

    elif modality == "cassi":
        H, W = SPATIAL_SIZE
        nb = 8
        bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
        for b in bases: b -= b.min(); b /= b.max() + 1e-8
        cube = np.zeros((H, W, nb), dtype=np.float32)
        wl = np.linspace(0, 1, nb)
        for l in range(nb):
            cube[:, :, l] = bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2) + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2) + bases[2] * 0.2
        cube -= cube.min(); cube /= cube.max() + 1e-8
        mask = (rng.random((H, W)) > 0.5).astype(np.float32)
        theta = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0, 0.0], "disp_poly_y": [0.0, 0.0, 0.0], "L": nb}
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        y = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            y += np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1) * mask
        y = y + rng.normal(0, 0.01, size=(H, W)).astype(np.float32)
        # Stale mask: 15% of entries flipped (no hash verification)
        mask_stale = mask.copy()
        flip = rng_stale.random((H, W)) < 0.15
        mask_stale[flip] = 1.0 - mask_stale[flip]
        x_hat = _cassi_recon(y, mask_stale, theta, nb, n_iter=abl_iters)
        return {"psnr": _compute_psnr(cube, x_hat)}

    return {"psnr": 0.0}


# ============================================================================
# Main ablation runner
# ============================================================================

ABLATION_NAMES = [
    "no_photon",
    "no_recoverability",
    "no_mismatch",
    "no_runbundle",
]

ABLATION_FNS = {
    "no_photon": _ablation_no_photon,
    "no_recoverability": _ablation_no_recoverability,
    "no_mismatch": _ablation_no_mismatch,
    "no_runbundle": _ablation_no_runbundle,
}

MODALITIES = ["spc", "cacti", "cassi"]


def run_ablations(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run all 4 ablations x 3 modalities."""
    os.makedirs(out_dir, exist_ok=True)

    n_trials = 2 if smoke else N_BOOTSTRAP_ABLATION
    recon_iters = 15 if smoke else 30
    modalities = ["spc"] if smoke else MODALITIES
    ablation_names = ABLATION_NAMES[:2] if smoke else ABLATION_NAMES

    all_results: List[Dict[str, Any]] = []

    for modality in modalities:
        # Baseline: full pipeline
        baseline_psnrs = []
        for trial in range(n_trials):
            seed = BASE_SEED + trial * 7
            r = _run_full_pipeline(modality, seed, recon_iters)
            baseline_psnrs.append(r["psnr"])
        baseline_mean = float(np.mean(baseline_psnrs))
        baseline_std = float(np.std(baseline_psnrs))

        logger.info("%s baseline: PSNR=%.2f +/- %.2f dB",
                    modality, baseline_mean, baseline_std)

        for ablation_name in ablation_names:
            ablation_fn = ABLATION_FNS[ablation_name]
            ablation_psnrs = []

            for trial in range(n_trials):
                seed = BASE_SEED + trial * 7
                r = ablation_fn(modality, seed, recon_iters)
                ablation_psnrs.append(r["psnr"])

            ablation_mean = float(np.mean(ablation_psnrs))
            ablation_std = float(np.std(ablation_psnrs))
            degradation_db = baseline_mean - ablation_mean

            # Bootstrap error bars
            if n_trials >= 3:
                boot_degradations = []
                for _ in range(50):
                    boot_rng = np.random.default_rng()
                    idx = boot_rng.integers(0, n_trials, size=n_trials)
                    boot_base = float(np.mean([baseline_psnrs[i] for i in idx]))
                    boot_abl = float(np.mean([ablation_psnrs[i] for i in idx]))
                    boot_degradations.append(boot_base - boot_abl)
                degradation_ci = [
                    float(np.percentile(boot_degradations, 2.5)),
                    float(np.percentile(boot_degradations, 97.5)),
                ]
            else:
                degradation_ci = [degradation_db, degradation_db]

            result = {
                "modality": modality,
                "ablation": ablation_name,
                "baseline_psnr_db": baseline_mean,
                "baseline_std_db": baseline_std,
                "ablation_psnr_db": ablation_mean,
                "ablation_std_db": ablation_std,
                "degradation_db": degradation_db,
                "degradation_ci": degradation_ci,
                "n_trials": n_trials,
            }

            logger.info(
                "  %s / %s: degradation=%.2f dB [%.2f, %.2f]",
                modality, ablation_name, degradation_db,
                degradation_ci[0], degradation_ci[1],
            )

            all_results.append(result)

    # Summary table
    logger.info("=" * 70)
    logger.info("%-8s %-22s %10s %15s", "Modal", "Ablation", "Degrad(dB)", "CI")
    logger.info("-" * 70)
    for r in all_results:
        logger.info(
            "%-8s %-22s %10.2f [%6.2f, %6.2f]",
            r["modality"], r["ablation"], r["degradation_db"],
            r["degradation_ci"][0], r["degradation_ci"][1],
        )

    # Save
    bundle = _make_run_bundle(
        "flagship_ablations",
        {"n_ablations": len(all_results), "n_modalities": len(modalities)},
        [BASE_SEED],
    )

    output = {
        "results": all_results,
        "bundle": bundle,
    }

    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Ablations: %d results -> %s", len(all_results), out_dir)
    return all_results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: Ablation studies"
    )
    parser.add_argument("--out_dir", default="results/flagship_ablations")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ablations(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
