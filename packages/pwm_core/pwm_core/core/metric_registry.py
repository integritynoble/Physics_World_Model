"""MetricRegistry -- per-modality quality metrics beyond PSNR."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import numpy as np


class Metric(ABC):
    name: str = "base"
    higher_is_better: bool = True

    @abstractmethod
    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        ...


class PSNR(Metric):
    name = "psnr"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        mse = np.mean((x_recon.astype(np.float64) - x_true.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        max_val = kwargs.get("max_val", np.max(np.abs(x_true)))
        if max_val == 0:
            return 0.0
        return float(10 * np.log10(max_val ** 2 / mse))


class SSIM(Metric):
    name = "ssim"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        x = x_recon.astype(np.float64).ravel()
        y = x_true.astype(np.float64).ravel()
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.var(x), np.var(y)
        sxy = np.mean((x - mx) * (y - my))
        c1 = (0.01 * (np.max(y) - np.min(y))) ** 2
        c2 = (0.03 * (np.max(y) - np.min(y))) ** 2
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx**2 + my**2 + c1) * (sx + sy + c2)
        return float(num / den) if den != 0 else 0.0


class NLL_Metric(Metric):
    name = "nll"
    higher_is_better = False  # Lower NLL is better

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        residual = x_recon.astype(np.float64) - x_true.astype(np.float64)
        sigma = kwargs.get("sigma", 1.0)
        return float(0.5 * np.sum(residual**2) / sigma**2)


class CRC(Metric):
    """Contrast Recovery Coefficient for PET/SPECT."""
    name = "crc"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        hot_mask = kwargs.get("hot_mask", x_true > np.percentile(x_true, 90))
        bg_mask = kwargs.get("bg_mask", x_true < np.percentile(x_true, 30))
        hot_mask = np.asarray(hot_mask, dtype=bool)
        bg_mask = np.asarray(bg_mask, dtype=bool)
        if not np.any(hot_mask) or not np.any(bg_mask):
            return 0.0
        recon_contrast = np.mean(x_recon[hot_mask]) / max(np.mean(x_recon[bg_mask]), 1e-12) - 1
        true_contrast = np.mean(x_true[hot_mask]) / max(np.mean(x_true[bg_mask]), 1e-12) - 1
        if abs(true_contrast) < 1e-12:
            return 0.0
        return float(recon_contrast / true_contrast)


class CNR(Metric):
    """Contrast-to-Noise Ratio for Ultrasound."""
    name = "cnr"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        hot_mask = kwargs.get("hot_mask", x_true > np.percentile(x_true, 80))
        bg_mask = kwargs.get("bg_mask", x_true < np.percentile(x_true, 20))
        hot_mask = np.asarray(hot_mask, dtype=bool)
        bg_mask = np.asarray(bg_mask, dtype=bool)
        if not np.any(hot_mask) or not np.any(bg_mask):
            return 0.0
        mu_hot = np.mean(x_recon[hot_mask])
        mu_bg = np.mean(x_recon[bg_mask])
        sigma_bg = max(np.std(x_recon[bg_mask]), 1e-12)
        return float(abs(mu_hot - mu_bg) / sigma_bg)


class FRC(Metric):
    """Fourier Ring Correlation proxy for SEM/TEM."""
    name = "frc"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        f1 = np.fft.fft2(x_recon.astype(np.float64).reshape(x_recon.shape[-2:]))
        f2 = np.fft.fft2(x_true.astype(np.float64).reshape(x_true.shape[-2:]))
        num = np.real(np.sum(f1 * np.conj(f2)))
        den = np.sqrt(np.sum(np.abs(f1)**2) * np.sum(np.abs(f2)**2))
        return float(num / den) if den > 0 else 0.0


class SpectralAngle(Metric):
    """Spectral Angle Mapper for hyperspectral (CASSI)."""
    name = "spectral_angle"
    higher_is_better = True

    def __call__(self, x_recon: np.ndarray, x_true: np.ndarray, **kwargs: Any) -> float:
        r = x_recon.astype(np.float64).reshape(-1, x_recon.shape[-1])
        t = x_true.astype(np.float64).reshape(-1, x_true.shape[-1])
        dots = np.sum(r * t, axis=1)
        norm_r = np.linalg.norm(r, axis=1)
        norm_t = np.linalg.norm(t, axis=1)
        denom = norm_r * norm_t
        valid = denom > 1e-12
        if not np.any(valid):
            return 0.0
        cos_angles = np.clip(dots[valid] / denom[valid], -1, 1)
        return float(np.mean(cos_angles))


METRIC_REGISTRY: Dict[str, Type[Metric]] = {
    "psnr": PSNR,
    "ssim": SSIM,
    "nll": NLL_Metric,
    "crc": CRC,
    "cnr": CNR,
    "frc": FRC,
    "spectral_angle": SpectralAngle,
}


def build_metric(name: str) -> Metric:
    if name not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]()
