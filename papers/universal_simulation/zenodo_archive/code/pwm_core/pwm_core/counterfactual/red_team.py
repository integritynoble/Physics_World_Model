"""pwm_core.counterfactual.red_team
====================================

Red Team scenario injection functions for counterfactual packs.

Three categories:
- **gate_flip**: Noise-dominant (Gate 2) instead of typical mismatch (Gate 3).
- **oof**: Out-of-family physics not in the declared forward model.
- **compute_trap**: High-dimensional search space tempting brute-force.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Gate-Flip: noise-dominant scenarios
# ---------------------------------------------------------------------------


def cassi_gate_flip() -> Dict[str, Any]:
    """CASSI gate-flip: heavy noise, negligible mismatch."""
    return {
        "mismatch": {
            "mask_dx": 0.05,
            "mask_dy": 0.05,
            "mask_theta": 0.01,
        },
        "noise": {"noise_alpha": 500.0, "noise_sigma": 0.05},
        "metadata": {"red_team": "gate_flip", "gate": 2,
                      "description": "Noise-dominant: alpha=500, sigma=0.05"},
    }


def spc_gate_flip() -> Dict[str, Any]:
    """SPC gate-flip: heavy noise, negligible gain drift."""
    return {
        "mismatch": {"gain_alpha": 0.0001},
        "noise": {"noise_sigma": 0.15},
        "metadata": {"red_team": "gate_flip", "gate": 2,
                      "description": "Noise-dominant: sigma=0.15"},
    }


def cacti_gate_flip() -> Dict[str, Any]:
    """CACTI gate-flip: heavy noise, negligible mismatch."""
    return {
        "mismatch": {"mask_dx": 0.1, "mask_dy": 0.1},
        "noise": {"noise_alpha": 50.0, "noise_sigma": 0.04},
        "metadata": {"red_team": "gate_flip", "gate": 2,
                      "description": "Noise-dominant: alpha=50, sigma=0.04"},
    }


# ---------------------------------------------------------------------------
# Out-of-Family: unmodeled physics
# ---------------------------------------------------------------------------


def cassi_oof_chromatic_aberration(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply chromatic aberration: wavelength-dependent dx shift.

    dx_k = dx_base + 0.02 * k applied as a slight spectral smearing
    of the measurement (simulated as band-dependent horizontal blur).
    """
    H, W = y.shape[:2]
    # Simulate chromatic smear: apply varying horizontal blur
    y_out = y.copy()
    sigma_max = 1.5
    y_out = gaussian_filter(y_out, sigma=[0, sigma_max])
    return y_out.astype(np.float32)


def cassi_oof_nonlinear_detector(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nonlinear detector response: y' = y^gamma with gamma != 1."""
    gamma = rng.choice([0.9, 1.1])
    y_max = float(np.max(y))
    if y_max > 0:
        y_norm = y / y_max
        y_norm = np.power(np.maximum(y_norm, 0), gamma)
        return (y_norm * y_max).astype(np.float32)
    return y


def cassi_oof_config() -> Dict[str, Any]:
    """CASSI out-of-family config."""
    return {
        "mismatch": {},
        "noise": {"noise_alpha": 100_000.0, "noise_sigma": 0.01},
        "post_injection": cassi_oof_combined,
        "metadata": {
            "red_team": "oof",
            "effects": ["chromatic_aberration", "nonlinear_detector"],
            "description": "Chromatic aberration + nonlinear detector gamma",
        },
    }


def cassi_oof_combined(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply both CASSI oof effects."""
    y = cassi_oof_chromatic_aberration(y, x_gt, mask, rng)
    y = cassi_oof_nonlinear_detector(y, x_gt, mask, rng)
    return y


def spc_oof_nonlinear_bucket(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nonlinear bucket response: y' = y + 0.1 * y^2."""
    return (y + 0.1 * y**2).astype(np.float32)


def spc_oof_crosstalk(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pixel crosstalk: 3x3 Gaussian blur on effective measurement."""
    # y is block-based (n_blocks, M); apply slight mixing
    n_mix = min(3, y.shape[0])
    y_out = y.copy()
    for i in range(1, y.shape[0] - 1):
        y_out[i] = 0.8 * y[i] + 0.1 * y[i - 1] + 0.1 * y[i + 1]
    return y_out.astype(np.float32)


def spc_oof_config() -> Dict[str, Any]:
    """SPC out-of-family config."""
    return {
        "mismatch": {},
        "noise": {"noise_sigma": 0.01},
        "post_injection": spc_oof_combined,
        "metadata": {
            "red_team": "oof",
            "effects": ["nonlinear_bucket", "block_crosstalk"],
            "description": "Nonlinear bucket (y+0.1*y^2) + block crosstalk",
        },
    }


def spc_oof_combined(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply both SPC oof effects."""
    y = spc_oof_nonlinear_bucket(y, x_gt, mask, rng)
    y = spc_oof_crosstalk(y, x_gt, mask, rng)
    return y


def cacti_oof_nonlinear_exposure(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nonlinear exposure gamma != 1.0 (from v9 PARAM_CONFIG)."""
    gamma = rng.choice([0.85, 0.9, 1.1, 1.15])
    y_max = float(np.max(y))
    if y_max > 0:
        y_norm = y / y_max
        y_out = np.power(np.maximum(y_norm, 0), gamma) * y_max
        return y_out.astype(np.float32)
    return y


def cacti_oof_temporal_blur(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Temporal motion blur: blend adjacent frames with weight 0.1.

    Applied to x_gt before re-computing measurement (simulated via
    blurring the 2D compressed measurement).
    """
    return gaussian_filter(y, sigma=0.8).astype(np.float32)


def cacti_oof_config() -> Dict[str, Any]:
    """CACTI out-of-family config."""
    return {
        "mismatch": {},
        "noise": {"noise_alpha": 2000.0, "noise_sigma": 0.01},
        "post_injection": cacti_oof_combined,
        "metadata": {
            "red_team": "oof",
            "effects": ["nonlinear_exposure", "temporal_blur"],
            "description": "Nonlinear exposure (gamma) + temporal motion blur",
        },
    }


def cacti_oof_combined(
    y: np.ndarray,
    x_gt: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply both CACTI oof effects."""
    y = cacti_oof_nonlinear_exposure(y, x_gt, mask, rng)
    y = cacti_oof_temporal_blur(y, x_gt, mask, rng)
    return y


# ---------------------------------------------------------------------------
# Compute Trap: high-dimensional search spaces
# ---------------------------------------------------------------------------


def cassi_compute_trap_config() -> Dict[str, Any]:
    """CASSI compute trap: polynomial dispersion (3 free params vs 1).

    Dispersion model: offset_k = a0 + a1*k + a2*k^2 (3 params).
    """
    return {
        "mismatch": {
            "mask_dx": 0.5,
            "mask_dy": 0.3,
            "mask_theta": 0.1,
            "disp_a1": 2.03,
            "disp_alpha": 0.2,
        },
        "noise": {"noise_alpha": 100_000.0, "noise_sigma": 0.01},
        "metadata": {
            "red_team": "compute_trap",
            "trap_type": "polynomial_dispersion",
            "description": "Polynomial dispersion a0+a1*k+a2*k^2 (3 free params)",
            "search_dim": 8,
        },
    }


def spc_compute_trap_config() -> Dict[str, Any]:
    """SPC compute trap: 2D spatially-varying gain map.

    Instead of 1D g(i) = exp(-alpha*i), need g(i,j) = exp(-alpha_h*i - alpha_w*j).
    """
    return {
        "mismatch": {
            "gain_alpha": 0.002,
            "gain_offset": 0.005,
        },
        "noise": {"noise_sigma": 0.02},
        "metadata": {
            "red_team": "compute_trap",
            "trap_type": "2d_gain_map",
            "description": "2D spatially-varying gain g(i,j) instead of 1D g(i)",
            "search_dim": "HxW",
        },
    }


def cacti_compute_trap_config() -> Dict[str, Any]:
    """CACTI compute trap: per-frame mask misalignment.

    Separate dx, dy, theta per frame = 3*T=24 parameters for T=8.
    """
    return {
        "mismatch": {
            "mask_dx": 1.0,
            "mask_dy": 0.5,
            "mask_theta": 0.2,
        },
        "noise": {"noise_alpha": 1500.0, "noise_sigma": 0.01},
        "metadata": {
            "red_team": "compute_trap",
            "trap_type": "per_frame_misalignment",
            "description": "Per-frame mask misalignment (3*T=24 params for T=8)",
            "search_dim": 24,
        },
    }


# ---------------------------------------------------------------------------
# Registry of red-team configs by modality
# ---------------------------------------------------------------------------

RED_TEAM_REGISTRY = {
    "cassi": {
        "gate_flip": cassi_gate_flip,
        "oof": cassi_oof_config,
        "compute_trap": cassi_compute_trap_config,
    },
    "spc": {
        "gate_flip": spc_gate_flip,
        "oof": spc_oof_config,
        "compute_trap": spc_compute_trap_config,
    },
    "cacti": {
        "gate_flip": cacti_gate_flip,
        "oof": cacti_oof_config,
        "compute_trap": cacti_compute_trap_config,
    },
}


def get_red_team_configs(modality: str) -> Dict[str, Dict[str, Any]]:
    """Get all red-team configs for a modality.

    Returns dict: {"gate_flip": {...}, "oof": {...}, "compute_trap": {...}}.
    """
    factories = RED_TEAM_REGISTRY.get(modality, {})
    return {name: factory() for name, factory in factories.items()}
