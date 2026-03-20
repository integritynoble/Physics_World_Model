"""Parameterised mismatch injection for InverseNet.

Defines *severity* levels (mild / moderate / severe) for every mismatch
family used by the three benchmark modalities.  All helpers take a severity
enum and return the corresponding delta-theta dict that is *added* (or
multiplied) to the nominal operator parameters.

Reuses ``pwm_core.mismatch`` primitives wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ── Severity levels ─────────────────────────────────────────────────────

class Severity(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


# ── Sub-pixel imports ──────────────────────────────────────────────────

from pwm_core.mismatch.subpixel import subpixel_shift_2d, subpixel_shift_3d_spatial, subpixel_warp_2d


# ── Per-family severity tables ──────────────────────────────────────────

# SPC mismatch families
SPC_GAIN_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"gain_factor": 1.05, "bias": 0.01},
    Severity.moderate: {"gain_factor": 1.15, "bias": 0.05},
    Severity.severe:   {"gain_factor": 1.40, "bias": 0.12},
}

SPC_MASK_ERROR_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"flip_fraction": 0.02},
    Severity.moderate: {"flip_fraction": 0.08},
    Severity.severe:   {"flip_fraction": 0.20},
}

# CACTI mismatch families
CACTI_MASK_SHIFT_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"shift_px": 1.0},
    Severity.moderate: {"shift_px": 3.0},
    Severity.severe:   {"shift_px": 6.0},
}

CACTI_MASK_ROTATION_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"angle_deg": 0.5},
    Severity.moderate: {"angle_deg": 1.5},
    Severity.severe:   {"angle_deg": 3.0},
}

CACTI_TEMPORAL_JITTER_TABLE: Dict[Severity, Dict[str, int]] = {
    Severity.mild:     {"timing_offset": 1},
    Severity.moderate: {"timing_offset": 2},
    Severity.severe:   {"timing_offset": 4},
}

# CASSI mismatch families
CASSI_DISP_STEP_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"disp_step_delta": 0.3},
    Severity.moderate: {"disp_step_delta": 1.0},
    Severity.severe:   {"disp_step_delta": 2.5},
}

CASSI_MASK_SHIFT_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"mask_dx": 0.5, "mask_dy": 0.5},
    Severity.moderate: {"mask_dx": 1.5, "mask_dy": 1.5},
    Severity.severe:   {"mask_dx": 3.0, "mask_dy": 3.0},
}

CASSI_PSF_BLUR_TABLE: Dict[Severity, Dict[str, float]] = {
    Severity.mild:     {"psf_sigma": 0.5},
    Severity.moderate: {"psf_sigma": 1.5},
    Severity.severe:   {"psf_sigma": 3.0},
}


# ── Lookup helpers ──────────────────────────────────────────────────────

_TABLES = {
    # SPC
    ("spc", "gain"):            SPC_GAIN_TABLE,
    ("spc", "mask_error"):      SPC_MASK_ERROR_TABLE,
    # CACTI
    ("cacti", "mask_shift"):    CACTI_MASK_SHIFT_TABLE,
    ("cacti", "mask_rotation"): CACTI_MASK_ROTATION_TABLE,
    ("cacti", "temporal_jitter"): CACTI_TEMPORAL_JITTER_TABLE,
    # CASSI
    ("cassi", "disp_step"):     CASSI_DISP_STEP_TABLE,
    ("cassi", "mask_shift"):    CASSI_MASK_SHIFT_TABLE,
    ("cassi", "PSF_blur"):      CASSI_PSF_BLUR_TABLE,
}


def get_delta_theta(
    modality: str, mismatch_family: str, severity: Severity
) -> Dict[str, Any]:
    """Return delta-theta dict for a given modality + family + severity.

    Raises ``KeyError`` if the combination is not registered.
    """
    key = (modality, mismatch_family)
    if key not in _TABLES:
        raise KeyError(f"Unknown mismatch family: {key}")
    table = _TABLES[key]
    if severity not in table:
        raise KeyError(f"Unknown severity {severity} for {key}")
    return dict(table[severity])


def list_families(modality: str):
    """Return list of mismatch family names for a modality."""
    return [fam for (mod, fam) in _TABLES if mod == modality]


# ── Apply helpers ───────────────────────────────────────────────────────

def apply_spc_gain_mismatch(
    y: np.ndarray,
    delta: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply gain + bias mismatch to SPC measurements."""
    gain = delta.get("gain_factor", 1.0)
    bias = delta.get("bias", 0.0)
    return y * gain + bias


def apply_spc_mask_error(
    mask: np.ndarray,
    delta: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip a fraction of mask elements."""
    frac = delta.get("flip_fraction", 0.0)
    flip_mask = rng.random(mask.shape) < frac
    out = mask.copy()
    out[flip_mask] = 1.0 - out[flip_mask]
    return out


def apply_cacti_mask_shift(
    masks: np.ndarray,
    delta: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Spatially shift CACTI masks by shift_px pixels vertically (sub-pixel)."""
    shift = delta.get("shift_px", 0.0)
    return subpixel_shift_3d_spatial(masks, 0.0, float(shift))


def apply_cacti_rotation(
    masks: np.ndarray,
    delta: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Rotate CACTI masks by angle_deg degrees (sub-pixel)."""
    angle = delta.get("angle_deg", 0.0)
    out = np.empty_like(masks, dtype=np.float64)
    for t in range(masks.shape[2]):
        out[:, :, t] = subpixel_warp_2d(masks[:, :, t], 0.0, 0.0, angle)
    return out


def apply_cacti_temporal_jitter(
    masks: np.ndarray,
    delta: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Reorder temporal mask indices by timing_offset."""
    offset = delta.get("timing_offset", 0)
    n_frames = masks.shape[2]
    indices = [(t + offset) % n_frames for t in range(n_frames)]
    return masks[:, :, indices]


def apply_cassi_disp_step(
    theta: Dict[str, Any],
    delta: Dict[str, float],
) -> Dict[str, Any]:
    """Perturb the dispersion polynomial coefficients."""
    theta_out = dict(theta)
    disp = list(theta_out.get("disp_poly_x", [0.0, 1.0, 0.0]))
    disp[1] = disp[1] + delta.get("disp_step_delta", 0.0)
    theta_out["disp_poly_x"] = disp
    return theta_out


def apply_cassi_mask_shift(
    mask: np.ndarray,
    delta: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Spatially shift the CASSI coded aperture (sub-pixel)."""
    dx = delta.get("mask_dx", 0.0)
    dy = delta.get("mask_dy", 0.0)
    return subpixel_shift_2d(mask, float(dx), float(dy))


def apply_cassi_psf_blur(
    y: np.ndarray,
    delta: Dict[str, float],
) -> np.ndarray:
    """Apply Gaussian PSF blur to CASSI measurement."""
    from scipy.ndimage import gaussian_filter
    sigma = delta.get("psf_sigma", 0.0)
    if sigma <= 0:
        return y
    return gaussian_filter(y, sigma=sigma).astype(y.dtype)


# ── Dispatcher ──────────────────────────────────────────────────────────

def apply_mismatch(
    modality: str,
    mismatch_family: str,
    severity: Severity,
    *,
    y: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    theta: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Apply mismatch and return dict of modified artifacts.

    Returns a dict that may contain:
      - "y"     : modified measurement
      - "mask"  : modified mask
      - "masks" : modified temporal masks
      - "theta" : modified theta dict
      - "delta_theta": the delta that was applied
    """
    if rng is None:
        rng = np.random.default_rng(0)

    delta = get_delta_theta(modality, mismatch_family, severity)
    result: Dict[str, Any] = {"delta_theta": delta}

    if modality == "spc":
        if mismatch_family == "gain" and y is not None:
            result["y"] = apply_spc_gain_mismatch(y, delta, rng)
        elif mismatch_family == "mask_error" and mask is not None:
            result["mask"] = apply_spc_mask_error(mask, delta, rng)

    elif modality == "cacti":
        if mismatch_family == "mask_shift" and masks is not None:
            result["masks"] = apply_cacti_mask_shift(masks, delta, rng)
        elif mismatch_family == "mask_rotation" and masks is not None:
            result["masks"] = apply_cacti_rotation(masks, delta, rng)
        elif mismatch_family == "temporal_jitter" and masks is not None:
            result["masks"] = apply_cacti_temporal_jitter(masks, delta, rng)

    elif modality == "cassi":
        if mismatch_family == "disp_step" and theta is not None:
            result["theta"] = apply_cassi_disp_step(theta, delta)
        elif mismatch_family == "mask_shift" and mask is not None:
            result["mask"] = apply_cassi_mask_shift(mask, delta, rng)
        elif mismatch_family == "PSF_blur" and y is not None:
            result["y"] = apply_cassi_psf_blur(y, delta)

    return result
