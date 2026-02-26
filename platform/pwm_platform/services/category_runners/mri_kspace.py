"""MRI k-space category runner.

Adapted from benchmarks/categories/medical_mri_kspace.py.
Generates brain phantom + undersampled k-space forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/medical_mri_kspace.py) ──


def _brain_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W), dtype=np.float32)
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)

    skull = (X / 0.8) ** 2 + (Y / 0.95) ** 2 <= 1.0
    x[skull] = 0.2
    brain = (X / 0.7) ** 2 + (Y / 0.85) ** 2 <= 1.0
    x[brain] = 0.8

    vent_l = ((X + 0.15) / 0.08) ** 2 + (Y / 0.25) ** 2 <= 1.0
    vent_r = ((X - 0.15) / 0.08) ** 2 + (Y / 0.25) ** 2 <= 1.0
    x[vent_l] = 1.0
    x[vent_r] = 1.0

    for _ in range(8):
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.6, 0.6)
        r = rng.uniform(0.05, 0.12)
        patch = ((X - cx) / r) ** 2 + ((Y - cy) / r) ** 2 <= 1.0
        x[brain & patch] = rng.uniform(0.5, 0.9)
    return x


def _image_to_kspace(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))).astype(np.complex64)


def _cartesian_undersampling_mask(
    shape: Tuple[int, int], acceleration: int = 4, acs_lines: int = 16, rng: np.random.Generator = None,
) -> np.ndarray:
    kH, kW = shape
    mask = np.zeros(shape, dtype=np.float32)
    center = kH // 2
    acs_start = center - acs_lines // 2
    mask[acs_start:acs_start + acs_lines, :] = 1.0
    remaining = kH - acs_lines
    n_sample = max(1, remaining // acceleration)
    candidates = list(range(0, acs_start)) + list(range(acs_start + acs_lines, kH))
    if candidates and rng is not None:
        chosen = rng.choice(candidates, size=min(n_sample, len(candidates)), replace=False)
        mask[chosen, :] = 1.0
    return mask


class MRIKspaceRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _brain_phantom(_SIZE, _SIZE, rng)
        return phantom, "Brain phantom", "gray"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        theta = config.get("theta") or {}
        accel = int(theta.get("sampling_rate", 0.25) ** -1) if theta.get("sampling_rate") else 4
        kspace = _image_to_kspace(phantom)
        mask = _cartesian_undersampling_mask(phantom.shape[:2], acceleration=accel, rng=rng)
        undersampled = np.log1p(np.abs(kspace * mask))
        undersampled = undersampled / (undersampled.max() + 1e-12)
        return undersampled.astype(np.float32), "k-space (undersampled)", "magma"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "medical_mri_kspace")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"MRI simulation &mdash; {gt_ref}" if gt_ref else "MRI simulation (synthetic brain phantom)"
        return methods, labels, config["display_name"], attribution
