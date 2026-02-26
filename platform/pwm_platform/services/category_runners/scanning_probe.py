"""Scanning probe microscopy category runner.

Adapted from benchmarks/categories/scanning_probe.py.
Generates surface topography + tip-convolved forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/scanning_probe.py) ──


def _parabolic_tip(size: int, radius_px: float = 5.0) -> np.ndarray:
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    r2 = (xx - c) ** 2 + (yy - c) ** 2
    tip = r2 / (2 * radius_px)
    tip = np.clip(tip, 0, size)
    tip = (tip.max() - tip) / (tip.max() + 1e-12)
    return tip.astype(np.float32)


def _surface_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W), dtype=np.float32)
    for i in range(4):
        y_start = i * H // 4
        y_end = (i + 1) * H // 4
        x[y_start:y_end, :] = 0.2 * i

    n_bumps = max(5, int(H * W / 3000))
    for _ in range(n_bumps):
        cx = int(rng.integers(3, W - 3))
        cy = int(rng.integers(3, H - 3))
        r = int(rng.integers(2, max(3, min(H, W) // 20)))
        height = float(rng.uniform(0.3, 1.0))
        yy, xx = np.ogrid[:H, :W]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        bump = np.clip(1.0 - r2 / (r ** 2 + 1e-6), 0, 1) * height
        x = np.maximum(x, bump)
    return np.clip(x, 0, 1).astype(np.float32)


class ScanningProbeRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _surface_phantom(_SIZE, _SIZE, rng)
        return phantom, "Surface topography", "copper"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        tip = _parabolic_tip(11)
        convolved = fftconvolve(phantom, tip / tip.sum(), mode="same").astype(np.float32)
        return convolved, "Tip-convolved surface", "copper"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "scanning_probe")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"Scanning probe simulation &mdash; {gt_ref}" if gt_ref else "Scanning probe simulation (synthetic surface)"
        return methods, labels, config["display_name"], attribution
