"""Microscopy PSF category runner.

Adapted from benchmarks/categories/microscopy_psf.py.
Generates cell phantom + PSF-blurred forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/microscopy_psf.py) ──


def _gaussian_psf(size: int, sigma: float = 2.0) -> np.ndarray:
    c = size // 2
    y, x = np.ogrid[:size, :size]
    psf = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
    return (psf / psf.sum()).astype(np.float32)


def _cell_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W), dtype=np.float32)
    n_cells = max(5, int(H * W / 2000))
    for _ in range(n_cells):
        cx = int(rng.integers(int(W * 0.08), int(W * 0.92)))
        cy = int(rng.integers(int(H * 0.08), int(H * 0.92)))
        rx = int(rng.integers(max(2, W // 40), max(4, W // 10)))
        ry = int(rng.integers(max(2, H // 40), max(4, H // 10)))
        intensity = float(rng.random() * 0.7 + 0.3)
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
        x[mask] = np.maximum(x[mask], intensity)
    return x


class MicroscopyPSFRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _cell_phantom(_SIZE, _SIZE, rng)
        return phantom, "Cell phantom", "hot"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        theta = config.get("theta") or {}
        sigma = theta.get("psf_sigma_px", 2.0)
        psf = _gaussian_psf(15, sigma=sigma)
        blurred = fftconvolve(phantom, psf, mode="same").astype(np.float32)
        # Add mild Poisson-like noise for realism
        blurred = np.clip(blurred, 0, None)
        noise = rng.normal(0, 0.02, blurred.shape).astype(np.float32)
        noisy = np.clip(blurred + noise, 0, 1)
        return noisy, "PSF-blurred", "hot"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "microscopy_psf")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"Microscopy simulation &mdash; {gt_ref}" if gt_ref else "Microscopy simulation (synthetic cell phantom)"
        return methods, labels, config["display_name"], attribution
