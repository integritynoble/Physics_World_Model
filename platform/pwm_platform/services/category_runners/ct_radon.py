"""CT / Radon-transform category runner.

Adapted from benchmarks/categories/medical_ct_radon.py.
Generates Shepp-Logan phantom + sinogram forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import rotate

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128  # Phantom resolution for simulation display


# ── Physics (copied from benchmarks/categories/medical_ct_radon.py) ──


def _shepp_logan(H: int, W: int) -> np.ndarray:
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),
        (0.06, -0.605, 0.023, 0.046, 0, 0.01),
    ]
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    img = np.zeros((H, W), dtype=np.float64)
    for cx, cy, rx, ry, angle, intensity in ellipses:
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Xr = cos_t * (X - cx) + sin_t * (Y - cy)
        Yr = -sin_t * (X - cx) + cos_t * (Y - cy)
        mask = (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1.0
        img[mask] += intensity
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)


def _radon_transform(image: np.ndarray, n_angles: int = 180) -> np.ndarray:
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    H, W = image.shape
    N = max(H, W)
    pad = abs(H - W) // 2
    if H < W:
        padded = np.pad(image, ((pad, pad + (W - H) % 2), (0, 0)), mode="constant")
    elif W < H:
        padded = np.pad(image, ((0, 0), (pad, pad + (H - W) % 2)), mode="constant")
    else:
        padded = image.copy()
    N = padded.shape[0]
    sinogram = np.zeros((n_angles, N), dtype=np.float32)
    for i, angle in enumerate(angles):
        rotated = rotate(padded, angle, reshape=False, order=1)
        sinogram[i] = rotated.sum(axis=0)
    return sinogram


class CTRadonRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _shepp_logan(_SIZE, _SIZE)
        return phantom, "Shepp-Logan phantom", "gray"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        n_angles = (config.get("theta") or {}).get("n_angles", 180)
        sinogram = _radon_transform(phantom, n_angles=min(n_angles, 180))
        return sinogram, "Sinogram", "inferno"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "medical_ct_radon")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"CT simulation &mdash; {gt_ref}" if gt_ref else "CT simulation (synthetic Shepp-Logan)"
        return methods, labels, config["display_name"], attribution
