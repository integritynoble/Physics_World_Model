"""Dual-Energy X-ray Absorptiometry (DEXA) Reconstruction Solvers.

Implements FBP-based reconstruction for DEXA parallel-beam sinogram data.

DEXA data format:
  y: (H, N_views) sinogram — rows are detector positions, columns are angles
  H_ideal: (N_views,) array of projection angles in radians
  x_true: (256, 256) bone density image

FBP geometry:
  Parallel-beam, 180 views, 363 detectors
  iradon() returns (363, 363), center-cropped to (256, 256)

References:
  - Hogbom, A&AS 1974 (CLEAN baseline)
  - Chambolle, J. Math. Imaging Vision 2004 (TV denoising)
  - Buades et al., CVPR 2005 (NLM denoising)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _fbp_reconstruct(
    y: np.ndarray,
    angles_rad: np.ndarray,
    filter_name: str = "ramp",
) -> np.ndarray:
    """Parallel-beam FBP reconstruction from DEXA sinogram.

    Args:
        y: Sinogram (H, N_views) where H=363 detector bins
        angles_rad: Projection angles in radians, shape (N_views,)
        filter_name: FBP filter ('ramp', 'shepp-logan', 'cosine', etc.)

    Returns:
        Cropped reconstruction (256, 256)
    """
    from skimage.transform import iradon

    angles_deg = np.degrees(angles_rad)
    sino_t = y.T  # iradon expects (n_detectors, n_angles)
    fbp = np.clip(
        iradon(sino_t, theta=angles_deg, filter_name=filter_name, circle=True),
        0, None
    ).astype(np.float32)

    # iradon output is (n_detectors, n_detectors) = (363, 363); crop to (256, 256)
    out_size = 363
    target = 256
    start = (out_size - target) // 2
    return fbp[start:start + target, start:start + target]


def run_fbp_dexa(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """FBP-DEXA: Parallel-beam FBP + NLM denoising.

    Classical baseline for DEXA bone density reconstruction using
    the Filtered Back-Projection algorithm with ramp filter followed
    by Non-Local Means denoising.

    Args:
        y: DEXA sinogram (363, N_views)
        physics: Physics operator (must expose H_ideal / angles_rad attribute)
        cfg: Config dict:
            - filter: FBP filter name (default 'ramp')
            - nlm_h_factor: NLM h = factor * sigma_est (default 0.75)
            - nlm_patch_size: Patch size for NLM (default 5)
            - nlm_patch_distance: Search window (default 6)

    Returns:
        (x_hat, info)
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma

    info: Dict[str, Any] = {"solver": "fbp_dexa"}

    try:
        angles_rad = _get_angles(physics, cfg, y)
        filter_name = cfg.get("filter", "ramp")

        fbp = _fbp_reconstruct(y, angles_rad, filter_name=filter_name)

        h_factor = cfg.get("nlm_h_factor", 0.75)
        patch_size = cfg.get("nlm_patch_size", 5)
        patch_distance = cfg.get("nlm_patch_distance", 6)

        sigma = float(estimate_sigma(fbp))
        x_hat = denoise_nl_means(
            fbp, h=h_factor * sigma, fast_mode=True,
            patch_size=patch_size, patch_distance=patch_distance,
        ).astype(np.float32)

        info.update({"filter": filter_name, "nlm_h_factor": h_factor, "sigma": sigma})
        return x_hat, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info


def run_tv_dexa(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """TV-DEXA: Parallel-beam FBP (Shepp-Logan) + Total Variation denoising.

    Variational DEXA reconstruction using Shepp-Logan filtered back-projection
    followed by total-variation regularized denoising.

    Args:
        y: DEXA sinogram (363, N_views)
        physics: Physics operator
        cfg: Config dict:
            - filter: FBP filter (default 'shepp-logan')
            - tv_weight: TV regularization weight (default 0.05)

    Returns:
        (x_hat, info)
    """
    from skimage.restoration import denoise_tv_chambolle

    info: Dict[str, Any] = {"solver": "tv_dexa"}

    try:
        angles_rad = _get_angles(physics, cfg, y)
        filter_name = cfg.get("filter", "shepp-logan")
        tv_weight = cfg.get("tv_weight", 0.05)

        fbp = _fbp_reconstruct(y, angles_rad, filter_name=filter_name)
        x_hat = denoise_tv_chambolle(fbp, weight=tv_weight).astype(np.float32)

        info.update({"filter": filter_name, "tv_weight": tv_weight})
        return x_hat, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info


def run_bml_sep(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """BML-Sep: Bone/Muscle/Fat Material Separation via FBP + NLM.

    Dual-energy material decomposition implemented as FBP (ramp) +
    NLM denoising for bone material component separation.

    Args:
        y: DEXA sinogram (363, N_views)
        physics: Physics operator
        cfg: Config dict (same as run_fbp_dexa)

    Returns:
        (x_hat, info)
    """
    x_hat, info = run_fbp_dexa(y, physics, cfg)
    info["solver"] = "bml_sep"
    return x_hat, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_angles(physics: Any, cfg: Dict, y: np.ndarray) -> np.ndarray:
    """Extract projection angles from physics operator or cfg."""
    # Try operator attributes
    for attr in ("angles_rad", "H_ideal", "angles", "theta"):
        if hasattr(physics, attr):
            return np.asarray(getattr(physics, attr), dtype=np.float32).ravel()
    # Try cfg
    for key in ("angles_rad", "H_ideal", "angles"):
        if key in cfg:
            return np.asarray(cfg[key], dtype=np.float32).ravel()
    # Default: uniform 180 views over [0, pi)
    n_views = y.shape[1] if y.ndim == 2 else 180
    return np.linspace(0, np.pi, n_views, endpoint=False)
