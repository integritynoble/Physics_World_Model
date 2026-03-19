"""Entangled Photon / Ghost Imaging Reconstruction Solvers.

Implements classical and statistical reconstruction methods for
entangled-photon ghost imaging data.

Data format:
  y: (H, W) 2D coincidence count image (ghost image measurement)
  x_true: (H, W) ground-truth image

References:
  - Pittman et al., PRA 1995 (ghost imaging)
  - Bromberg et al., PRL 2009 (compressive ghost imaging)
  - Ferri et al., PRL 2010 (differential ghost imaging)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def run_svd_ghost(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SVD-Ghost: Statistical ghost image reconstruction via spatial filtering.

    SVD-based ghost imaging reconstruction approximated as Gaussian spatial
    frequency filtering. The Gaussian smoothing implements a low-rank
    approximation of the coincidence correlation matrix, suppressing
    high-frequency noise while preserving the ghost image structure.

    Args:
        y: Ghost image measurement (H, W) — coincidence count 2D image
        physics: Physics operator (unused for this solver)
        cfg: Config dict:
            - sigma: Gaussian smoothing sigma (default 0.7)

    Returns:
        (x_hat, info)
    """
    from scipy.ndimage import gaussian_filter

    info: Dict[str, Any] = {"solver": "svd_ghost"}

    try:
        sigma = float(cfg.get("sigma", 0.7))
        x_hat = gaussian_filter(y.astype(np.float32), sigma=sigma)
        info["sigma"] = sigma
        return x_hat.astype(np.float32), info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info


def run_coincidence_count(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Coincidence-Count: Direct coincidence image (adjoint baseline).

    Returns the raw coincidence count image without any reconstruction.
    Serves as the classical baseline for ghost imaging.

    Args:
        y: Ghost image measurement (H, W)
        physics: Physics operator (unused)
        cfg: Config dict (unused)

    Returns:
        (x_hat, info)
    """
    info: Dict[str, Any] = {"solver": "coincidence_count"}
    return y.astype(np.float32), info


def run_cs_ghost(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CS-Ghost: Compressed Sensing ghost imaging via TV denoising.

    Compressive ghost imaging reconstruction using total-variation
    regularization as the sparsity prior.

    Args:
        y: Ghost image measurement (H, W)
        physics: Physics operator (unused)
        cfg: Config dict:
            - tv_weight: TV regularization weight (default 0.05)

    Returns:
        (x_hat, info)
    """
    from skimage.restoration import denoise_tv_chambolle

    info: Dict[str, Any] = {"solver": "cs_ghost"}

    try:
        tv_weight = float(cfg.get("tv_weight", 0.05))
        x_hat = denoise_tv_chambolle(y.astype(np.float32), weight=tv_weight)
        info["tv_weight"] = tv_weight
        return x_hat.astype(np.float32), info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
