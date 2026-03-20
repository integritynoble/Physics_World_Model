"""Remote sensing / SAR category runner.

Adapted from benchmarks/categories/remote_sensing_sar.py.
Generates urban/terrain scene + SAR phase history forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/remote_sensing_sar.py) ──


def _sar_phase_history(scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = scene.shape
    n_pulses, n_range_bins = H, W
    from scipy.ndimage import zoom
    if H != n_pulses or W != n_range_bins:
        scene_resized = zoom(scene, (n_pulses / H, n_range_bins / W), order=1)
    else:
        scene_resized = scene.copy()

    raw = np.fft.fft2(scene_resized)
    phase = rng.uniform(-np.pi, np.pi, raw.shape)
    raw = raw * np.exp(1j * phase)
    return np.abs(raw).astype(np.float32)


def _terrain_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W), dtype=np.float32)
    yy = np.linspace(0, 1, H)[:, None]
    xx = np.linspace(0, 1, W)[None, :]
    x += (0.2 * yy + 0.1 * xx).astype(np.float32)

    n_buildings = max(3, int(H * W / 5000))
    for _ in range(n_buildings):
        bx = int(rng.integers(0, W - 10))
        by = int(rng.integers(0, H - 10))
        bw = int(rng.integers(3, max(4, W // 15)))
        bh = int(rng.integers(3, max(4, H // 15)))
        intensity = float(rng.uniform(0.6, 1.0))
        x[by:by + bh, bx:bx + bw] = intensity

    n_points = max(5, int(H * W / 2000))
    for _ in range(n_points):
        py = int(rng.integers(0, H))
        px = int(rng.integers(0, W))
        x[py, px] = float(rng.uniform(0.7, 1.0))

    return np.clip(x, 0, 1).astype(np.float32)


class RemoteSensingRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _terrain_phantom(_SIZE, _SIZE, rng)
        return phantom, "Urban/terrain scene", "terrain"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phase_hist = _sar_phase_history(phantom, rng)
        # Log scale for display
        display = np.log1p(phase_hist)
        display = display / (display.max() + 1e-12)
        return display.astype(np.float32), "SAR phase history", "inferno"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "remote_sensing_sar")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"Remote sensing simulation &mdash; {gt_ref}" if gt_ref else "Remote sensing simulation (synthetic terrain)"
        return methods, labels, config["display_name"], attribution
