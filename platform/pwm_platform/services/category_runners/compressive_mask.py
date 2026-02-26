"""Compressive mask category runner.

Adapted from benchmarks/categories/compressive_mask.py.
Generates spectral scene + coded measurement forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/compressive_mask.py) ──


def _binary_random_mask(shape: Tuple[int, ...], density: float = 0.5, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    return (rng.random(shape) < density).astype(np.float32)


def _spectral_phantom(H: int, W: int, C: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W, C), dtype=np.float32)
    n_regions = max(5, int(H * W / 3000))
    spectra = []
    for _ in range(n_regions):
        center = float(rng.random()) * C
        width = float(rng.random()) * C * 0.3 + C * 0.05
        spec = np.exp(-0.5 * ((np.arange(C) - center) / max(width, 0.1)) ** 2)
        spectra.append((spec / (spec.max() + 1e-12)).astype(np.float32))

    for i in range(n_regions):
        cx = int(rng.integers(int(W * 0.1), int(W * 0.9)))
        cy = int(rng.integers(int(H * 0.1), int(H * 0.9)))
        r = int(rng.integers(max(3, min(H, W) // 15), max(5, min(H, W) // 5)))
        intensity = float(rng.random() * 0.7 + 0.3)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        x[mask] += intensity * spectra[i % len(spectra)][None, :]
    return np.clip(x, 0, 1).astype(np.float32)


class CompressiveMaskRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        x_shape = config.get("x_shape", [_SIZE, _SIZE, 28])
        C = x_shape[2] if len(x_shape) >= 3 else 28
        phantom = _spectral_phantom(_SIZE, _SIZE, C, rng)
        # Return mid-band slice for display
        mid = phantom[:, :, C // 2]
        return mid, "Spectral scene (mid-band)", "viridis"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        # Regenerate full spectral cube for coded measurement
        x_shape = config.get("x_shape", [_SIZE, _SIZE, 28])
        C = x_shape[2] if len(x_shape) >= 3 else 28
        cube = _spectral_phantom(_SIZE, _SIZE, C, rng)
        mask = _binary_random_mask((_SIZE, _SIZE), density=0.5, rng=rng)
        # CASSI-like: mask each band then sum
        y = np.zeros((_SIZE, _SIZE), dtype=np.float32)
        for c in range(C):
            y += cube[:, :, c] * mask
        y = y / C  # normalize
        return y, "Coded measurement", "inferno"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "compressive_mask")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"Compressive simulation &mdash; {gt_ref}" if gt_ref else "Compressive simulation (synthetic spectral scene)"
        return methods, labels, config["display_name"], attribution
