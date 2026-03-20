"""Evaluation utilities for the expert study.

Uses scale-invariant metrics: both x_hat and x_true are normalized to [0, 1]
before computing PSNR/SSIM, matching the platform's evaluation approach.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range for scale-invariant comparison."""
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-12:
        return (arr - lo) / (hi - lo)
    return arr - lo


def compute_metrics(x_hat: np.ndarray, x_true: np.ndarray) -> dict:
    """Compute PSNR and SSIM between reconstructed and ground truth images.

    Both arrays are independently normalized to [0, 1] before comparison
    (scale-invariant, matching the platform's _normalize_01 approach).
    """
    # Handle 3D spectral cubes (CASSI)
    if x_hat.ndim == 3 and x_true.ndim == 3:
        return _compute_metrics_3d(x_hat, x_true)

    # 2D images
    x_hat = np.clip(x_hat.astype(np.float64), 0, None)
    x_true = x_true.astype(np.float64)

    # Check both have non-trivial content
    gt_range = x_true.max() - x_true.min()
    rec_range = x_hat.max() - x_hat.min()
    if gt_range < 1e-10 or rec_range < 1e-10:
        return {"psnr": 0.0, "ssim": 0.0}

    # Ensure same shape
    if x_hat.shape != x_true.shape:
        from scipy.ndimage import zoom
        factors = [t / h for t, h in zip(x_true.shape, x_hat.shape)]
        x_hat = zoom(x_hat, factors, order=1)

    # Normalize both to [0, 1] for scale-invariant comparison
    x_true_n = _normalize_01(x_true)
    x_hat_n = _normalize_01(x_hat)

    psnr_val = float(peak_signal_noise_ratio(x_true_n, x_hat_n, data_range=1.0))
    ssim_val = float(structural_similarity(x_true_n, x_hat_n, data_range=1.0))

    return {"psnr": round(psnr_val, 2), "ssim": round(ssim_val, 4)}


def _compute_metrics_3d(x_hat: np.ndarray, x_true: np.ndarray) -> dict:
    """Compute metrics for 3D spectral cubes (average over bands)."""
    x_hat = np.clip(x_hat.astype(np.float64), 0, None)
    x_true = x_true.astype(np.float64)

    if x_hat.shape != x_true.shape:
        from scipy.ndimage import zoom
        factors = [t / h for t, h in zip(x_true.shape, x_hat.shape)]
        x_hat = zoom(x_hat, factors, order=1)

    # Per-band metrics with scale-invariant normalization
    n_bands = x_true.shape[2]
    psnrs, ssims = [], []
    for b in range(n_bands):
        gt_b = x_true[:, :, b]
        rec_b = x_hat[:, :, b]
        gt_range = gt_b.max() - gt_b.min()
        rec_range = rec_b.max() - rec_b.min()
        if gt_range < 1e-10 or rec_range < 1e-10:
            continue

        gt_n = _normalize_01(gt_b)
        rec_n = _normalize_01(rec_b)

        p = peak_signal_noise_ratio(gt_n, rec_n, data_range=1.0)
        s = structural_similarity(gt_n, rec_n, data_range=1.0)
        psnrs.append(p)
        ssims.append(s)

    return {
        "psnr": round(float(np.mean(psnrs)), 2) if psnrs else 0.0,
        "ssim": round(float(np.mean(ssims)), 4) if ssims else 0.0,
    }


def count_loc(code: str) -> dict:
    """Count lines of code, separating spec from reconstruction."""
    lines = code.strip().split("\n")
    total = 0
    spec_lines = 0
    recon_lines = 0
    in_spec = False
    in_recon = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            # Check if comment is spec/recon marker
            low = stripped.lower()
            if "forward model" in low or "specification" in low:
                in_spec = True
                in_recon = False
            elif "reconstruct" in low or "solver" in low or "algorithm" in low:
                in_spec = False
                in_recon = True
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue

        total += 1
        if in_spec:
            spec_lines += 1
        elif in_recon:
            recon_lines += 1

    return {
        "total_loc": total,
        "spec_loc": spec_lines,
        "recon_loc": recon_lines,
    }
