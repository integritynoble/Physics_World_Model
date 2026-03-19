"""Unified metrics for all benchmark evaluations.

Consolidates metric computation from existing run_all.py and
run_cassi_benchmark.py into a single module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_psnr(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    max_val: Optional[float] = None,
) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Args:
        x_true: Ground truth array.
        x_hat: Reconstructed array.
        max_val: Peak signal value.  If *None*, uses ``max(x_true.max(), x_hat.max())``.
                 Use 1.0 for data normalised to [0, 1].
    """
    mse = np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    if max_val is None:
        max_val = float(max(x_true.max(), x_hat.max()))
    if max_val < 1e-10:
        max_val = 1.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Structural Similarity Index.

    Uses ``skimage`` if available, otherwise falls back to a simplified
    global SSIM.
    """
    if data_range is None:
        data_range = float(max(x_true.max(), x_hat.max()))
    try:
        from skimage.metrics import structural_similarity
        # For 3-D cubes (spectral, temporal) average over last axis
        if x_true.ndim == 3:
            vals = []
            for i in range(x_true.shape[-1]):
                vals.append(structural_similarity(
                    x_true[..., i], x_hat[..., i], data_range=data_range,
                ))
            return float(np.mean(vals))
        return float(structural_similarity(x_true, x_hat, data_range=data_range))
    except ImportError:
        pass
    # Simplified global SSIM
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = float(x_true.mean()), float(x_hat.mean())
    var_x, var_y = float(x_true.var()), float(x_hat.var())
    cov = float(np.mean((x_true - mu_x) * (x_hat - mu_y)))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim)


def compute_sam(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Spectral Angle Mapper (degrees).

    Operates on 3-D cubes ``(H, W, C)`` — computes mean spectral angle.
    For 2-D inputs returns 0.0.
    """
    if x_true.ndim < 3:
        return 0.0
    # Reshape to (pixels, channels)
    n_pix = x_true.shape[0] * x_true.shape[1]
    a = x_true.reshape(n_pix, -1).astype(np.float64)
    b = x_hat.reshape(n_pix, -1).astype(np.float64)
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1) + 1e-12
    norm_b = np.linalg.norm(b, axis=1) + 1e-12
    cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
    angles = np.arccos(cos_angle) * 180.0 / np.pi
    return float(np.mean(angles))


def compute_nrmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Normalised Root Mean Square Error."""
    rmse = float(np.sqrt(np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2)))
    drange = float(x_true.max() - x_true.min())
    if drange < 1e-10:
        return 0.0
    return rmse / drange


def compute_measurement_residual(
    y_meas: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Relative ℓ₂ measurement residual  ‖y - Ax̂‖ / ‖y‖."""
    diff = y_meas.astype(np.float64) - y_pred.astype(np.float64)
    norm_y = np.linalg.norm(y_meas.astype(np.float64))
    if norm_y < 1e-12:
        return 0.0
    return float(np.linalg.norm(diff) / norm_y)


# ---------------------------------------------------------------------------
# Per-band metrics (spectral modalities)
# ---------------------------------------------------------------------------

def compute_per_band_psnr(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    max_val: float = 1.0,
) -> List[float]:
    """PSNR per spectral band for (H, W, C) cubes."""
    if x_true.ndim < 3:
        return [compute_psnr(x_true, x_hat, max_val)]
    return [compute_psnr(x_true[..., i], x_hat[..., i], max_val)
            for i in range(x_true.shape[-1])]


def compute_per_band_ssim(
    x_true: np.ndarray,
    x_hat: np.ndarray,
) -> List[float]:
    """SSIM per spectral band for (H, W, C) cubes."""
    if x_true.ndim < 3:
        return [compute_ssim(x_true, x_hat)]
    return [compute_ssim(x_true[..., i], x_hat[..., i]) for i in range(x_true.shape[-1])]


# ---------------------------------------------------------------------------
# MetricSet – convenience container
# ---------------------------------------------------------------------------

# Map from metric name to function
METRIC_FUNCTIONS = {
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "sam": compute_sam,
    "nrmse": compute_nrmse,
}


@dataclass
class MetricSet:
    """Named collection of metrics with acceptance thresholds.

    Attributes:
        names: Which metrics to compute (e.g. ``["psnr", "ssim"]``).
        primary: The primary metric used for pass/fail (e.g. ``"psnr"``).
        thresholds: Acceptance thresholds per metric (e.g. ``{"psnr": 25.0}``).
    """
    names: List[str] = field(default_factory=lambda: ["psnr", "ssim"])
    primary: str = "psnr"
    thresholds: Dict[str, float] = field(default_factory=dict)

    def evaluate(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all requested metrics.

        Returns:
            Dict mapping metric name to value.
        """
        results: Dict[str, float] = {}
        for name in self.names:
            fn = METRIC_FUNCTIONS.get(name)
            if fn is not None:
                results[name] = fn(x_true, x_hat)
        return results

    def check_pass(self, results: Dict[str, float]) -> bool:
        """Check whether results meet acceptance thresholds."""
        for metric, threshold in self.thresholds.items():
            val = results.get(metric)
            if val is None:
                continue
            # For PSNR/SSIM higher is better; for NRMSE lower is better
            if metric in ("nrmse",):
                if val > threshold:
                    return False
            else:
                if val < threshold:
                    return False
        return True
