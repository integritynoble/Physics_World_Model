"""Image quality metrics: PSNR and SSIM."""
from __future__ import annotations
import numpy as np


def psnr(x_hat: np.ndarray, x_true: np.ndarray, data_range: float | None = None) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Args:
        x_hat:      Reconstructed image.
        x_true:     Ground truth image (same shape and scale).
        data_range: Signal range. Defaults to x_true.max() - x_true.min().
    """
    x_hat  = np.asarray(x_hat,  dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    mse = np.mean((x_hat - x_true) ** 2)
    if mse == 0.0:
        return float("inf")
    if data_range is None:
        data_range = float(x_true.max() - x_true.min())
    if data_range <= 0:
        data_range = 1.0
    return float(10.0 * np.log10(data_range**2 / mse))


def ssim(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    data_range: float | None = None,
    win_size: int = 7,
) -> float:
    """Structural Similarity Index (SSIM).

    Falls back to skimage.metrics.structural_similarity if available,
    otherwise uses a lightweight custom implementation.
    """
    try:
        from skimage.metrics import structural_similarity
        dr = data_range or float(x_true.max() - x_true.min()) or 1.0
        return float(
            structural_similarity(
                np.asarray(x_hat,  dtype=np.float64),
                np.asarray(x_true, dtype=np.float64),
                data_range=dr,
                win_size=min(win_size, min(x_true.shape) - 1 | 1),
            )
        )
    except ImportError:
        return _ssim_basic(x_hat, x_true)


def _ssim_basic(x: np.ndarray, y: np.ndarray) -> float:
    """Minimal SSIM implementation (no windowing)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    C1 = (0.01 * (y.max() - y.min() + 1e-8)) ** 2
    C2 = (0.03 * (y.max() - y.min() + 1e-8)) ** 2
    mu_x, mu_y   = x.mean(), y.mean()
    sig_x        = np.sqrt(((x - mu_x) ** 2).mean())
    sig_y        = np.sqrt(((y - mu_y) ** 2).mean())
    sig_xy       = ((x - mu_x) * (y - mu_y)).mean()
    numerator    = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    denominator  = (mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2)
    return float(numerator / denominator)
