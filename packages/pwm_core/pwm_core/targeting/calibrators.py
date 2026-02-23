"""pwm_core.targeting.calibrators
================================

Default self-calibrating operators for the LIP-Arena Targeting System.

Each calibrator follows the harness ``calibrator_fn`` interface::

    H_hat, cal_info = calibrator_fn(y, H_nom, budget_s)

where:
    y         — measurements (from H_true, unknown to calibrator)
    H_nom     — nominal (mismatched) physics operator
    budget_s  — wall-clock budget in seconds
    H_hat     — calibrated operator (best estimate of H_true)
    cal_info  — dict of diagnostics

All calibrators are blind: they have access only to y and H_nom.

Calibrators registered in ``DEFAULT_CALIBRATORS`` are auto-loaded by the
Harness when no custom calibrator is provided.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _total_variation(x: np.ndarray) -> float:
    """L1 isotropic total variation of a 2D image."""
    arr = np.abs(x)
    return float(
        np.sum(np.abs(np.diff(arr, axis=-2)))
        + np.sum(np.abs(np.diff(arr, axis=-1)))
    )


def _data_consistency(y: np.ndarray, H_cand: Any, x_hat: np.ndarray) -> float:
    """||y - H_cand(x_hat)||_2^2 / ||y||_2^2.  Lower is better."""
    y_pred = H_cand.forward(x_hat.ravel().reshape(H_cand.x_shape))
    resid = y.ravel() - y_pred.ravel()
    denom = max(float(np.sum(y.ravel() ** 2)), 1e-30)
    return float(np.sum(resid ** 2)) / denom


# ---------------------------------------------------------------------------
# CT: angle-offset calibrator (TV grid-search)
# ---------------------------------------------------------------------------


def _psd_anisotropy(x_hat: np.ndarray, n_angle_bins: int = 8) -> float:
    """Power-spectrum anisotropy of a 2D reconstruction.

    Returns the max/min ratio of angular power at mid-frequencies.
    Value is ~1 for isotropic images; high for streak-corrupted reconstructions.
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(x_hat.astype(np.float64)))) ** 2
    cx, cy = f.shape[0] // 2, f.shape[1] // 2
    yi, xi = np.mgrid[-cx : f.shape[0] - cx, -cy : f.shape[1] - cy]
    radii = np.sqrt(yi ** 2 + xi ** 2)
    r_lo, r_hi = max(cx // 4, 1), max(cx // 2, 2)
    mask = (radii >= r_lo) & (radii < r_hi)
    if not mask.any():
        return 1.0
    angles_map = np.arctan2(yi, xi)
    bins = np.floor(
        (angles_map[mask] + np.pi) / (2 * np.pi / n_angle_bins)
    ).astype(int)
    bins = np.clip(bins, 0, n_angle_bins - 1)
    bin_powers = [
        f[mask][bins == k].mean() if (bins == k).any() else 0.0
        for k in range(n_angle_bins)
    ]
    denom = min(bin_powers) + 1e-30
    return float(max(bin_powers) / denom)


def ct_anisotropy_calibrator(
    y: np.ndarray,
    H_nom: Any,
    budget_s: float = 300.0,
    n_coarse: int = 25,
    n_fine: int = 9,
) -> Tuple[Any, Dict[str, Any]]:
    """CT angle-offset calibrator using PSD anisotropy minimisation.

    Strategy (fast path — no CTRadon matrix rebuilds in the inner loop):
    1. Coarse pass: 25 offsets in [-20°, +20°], run ``fbp_2d`` directly.
    2. Fine pass: 9 offsets ±2 grid-steps around the coarse winner.
    3. Single CTRadon rebuild for the final best offset.

    PSD anisotropy is low for isotropic (correctly reconstructed) images
    and high when angle-offset streaks are present — providing a reliable
    self-calibration signal without reference images.
    """
    from pwm_core.recon.ct_solvers import fbp_2d
    from pwm_core.targeting.mismatch_sampler import inject_mismatch

    t0 = time.perf_counter()

    n_angles = y.shape[0] if y.ndim >= 2 else 180
    scores = []

    def _eval(offset_deg: float) -> float:
        angles = np.linspace(
            np.deg2rad(float(offset_deg)),
            np.deg2rad(180.0 + float(offset_deg)),
            n_angles,
            endpoint=False,
        )
        x_hat = fbp_2d(y, angles, filter_type="ramlak")
        return _psd_anisotropy(x_hat)

    # Coarse sweep
    coarse_offsets = np.linspace(-20.0, 20.0, n_coarse)
    for od in coarse_offsets:
        if time.perf_counter() - t0 > budget_s * 0.7:
            break
        sc = _eval(od)
        scores.append({"offset_deg": float(od), "anisotropy": sc})

    best_coarse = min(scores, key=lambda d: d["anisotropy"])["offset_deg"]
    step = coarse_offsets[1] - coarse_offsets[0] if len(coarse_offsets) > 1 else 1.67

    # Fine sweep around the coarse winner
    fine_offsets = np.linspace(best_coarse - 2 * step, best_coarse + 2 * step, n_fine)
    for od in fine_offsets:
        if time.perf_counter() - t0 > budget_s * 0.9:
            break
        sc = _eval(od)
        scores.append({"offset_deg": float(od), "anisotropy": sc})

    best_entry = min(scores, key=lambda d: d["anisotropy"])
    best_offset = best_entry["offset_deg"]

    # Single rebuild for the winning operator
    best_H = inject_mismatch(H_nom, {"radon.angle_offset": best_offset})

    elapsed = time.perf_counter() - t0
    cal_info = {
        "method": "ct_psd_anisotropy",
        "best_angle_offset_deg": best_offset,
        "best_anisotropy": best_entry["anisotropy"],
        "n_evaluated": len(scores),
        "elapsed_s": elapsed,
        "scores": scores,
    }
    logger.debug(
        f"ct_anisotropy_calibrator: best_offset={best_offset:.2f}°, "
        f"anisotropy={best_entry['anisotropy']:.4f}, n={len(scores)}, "
        f"t={elapsed:.2f}s"
    )
    return best_H, cal_info


# ---------------------------------------------------------------------------
# MRI: k-space seed calibrator (TV over broad seed range)
# ---------------------------------------------------------------------------


def mri_mask_calibrator(
    y: np.ndarray,
    H_nom: Any,
    budget_s: float = 300.0,
) -> Tuple[Any, Dict[str, Any]]:
    """MRI k-space mask calibrator using measurement support inference.

    The true k-space mask M_true is inferred directly from the measurement:
    ``|y[i,j]| > threshold`` iff ``M_true[i,j] == 1``.  This bypasses any
    seed search and exactly recovers the sampling pattern used at acquisition
    (modulo noise floor), giving near-perfect MRI calibration.

    The inferred mask is injected into a cloned H_nom to form H_hat.
    """
    import copy

    t0 = time.perf_counter()

    y_arr = np.asarray(y)

    # Threshold at 1% of max amplitude to infer which k-space lines were sampled
    amp = np.abs(y_arr)
    threshold = 0.01 * amp.max() if amp.max() > 0 else 1e-10
    inferred_mask = (amp > threshold).astype(np.float64)

    # Ensure the always-acquired center k-space is covered
    if inferred_mask.ndim == 2:
        H, W = inferred_mask.shape
        ch, cw = H // 8, W // 8
        inferred_mask[H // 2 - ch : H // 2 + ch, W // 2 - cw : W // 2 + cw] = 1.0

    # Clone H_nom and replace the MRI k-space mask directly
    H_hat = copy.deepcopy(H_nom)
    replaced = False
    if hasattr(H_hat, "node_map"):
        for node_id, prim in list(H_hat.node_map.items()):
            if getattr(prim, "primitive_id", "") == "mri_kspace" and hasattr(prim, "_mask"):
                prim._mask = inferred_mask
                replaced = True
                break

    elapsed = time.perf_counter() - t0
    coverage = float(inferred_mask.mean())
    cal_info = {
        "method": "mri_mask_inference",
        "inferred_coverage": coverage,
        "threshold": threshold,
        "mask_replaced": replaced,
        "elapsed_s": elapsed,
    }
    logger.debug(
        f"mri_mask_calibrator: coverage={coverage:.3f}, "
        f"replaced={replaced}, t={elapsed:.4f}s"
    )
    return H_hat, cal_info


# ---------------------------------------------------------------------------
# Registry and auto-discovery
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATORS: Dict[str, Callable] = {
    "ct": ct_anisotropy_calibrator,
    "mri": mri_mask_calibrator,
}


def get_default_calibrator(modality: str) -> Optional[Callable]:
    """Return the default calibrator callable for *modality*, or None."""
    return DEFAULT_CALIBRATORS.get(modality)
