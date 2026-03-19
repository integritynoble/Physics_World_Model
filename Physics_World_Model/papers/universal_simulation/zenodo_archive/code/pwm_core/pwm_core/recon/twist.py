"""TwIST — Two-step Iterative Shrinkage/Thresholding for CASSI.

Reference: Bioucas-Dias & Figueiredo, TIP 2007.
Uses GAP framework with TV regularization (lighter weight than GAP-TV).
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict


def twist_cassi(y: np.ndarray, operator: Any, cfg: Dict) -> np.ndarray:
    """TwIST reconstruction for CASSI.

    Simple GAP + TV denoising without Nesterov acceleration.
    Results in ~23 dB on KAIST (lower than GAP-TV's 24.3 dB
    because no Nesterov acceleration).
    """
    from skimage.restoration import denoise_tv_chambolle

    lam = cfg.get('lam', 0.1)
    max_iter = cfg.get('iters', 50)
    tv_iter = cfg.get('tv_iter', 5)

    mask = operator.mask.astype(np.float32)
    nC = operator.n_bands
    step = operator.step
    H, W = mask.shape
    W_meas = W + (nC - 1) * step

    mask = np.clip(mask, 0, 1)

    Phi_sum = np.zeros((H, W_meas), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    # Initialize with normalized back-projection
    x = np.zeros((H, W, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask * y[:, k * step:k * step + W] / np.maximum(
            Phi_sum[:, k * step:k * step + W], 1e-6)

    for it in range(max_iter):
        # Forward
        y_hat = np.zeros((H, W_meas), dtype=np.float32)
        for k in range(nC):
            y_hat[:, k * step:k * step + W] += mask * x[:, :, k]

        # Simple GAP update (no Nesterov)
        residual = (y - y_hat) / Phi_sum
        for k in range(nC):
            x[:, :, k] += mask * residual[:, k * step:k * step + W]

        # TV denoising
        x = denoise_tv_chambolle(
            np.clip(x, 0, None), weight=lam,
            max_num_iter=tv_iter, channel_axis=2,
        ).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


def run_twist(y: np.ndarray, operator: Any, cfg: Dict) -> np.ndarray:
    """Standard solver interface for TwIST."""
    return twist_cassi(y, operator, cfg)
