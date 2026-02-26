"""Electron CTF category runner.

Adapted from benchmarks/categories/electron_ctf.py.
Generates particle phantom + CTF-modulated forward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._base import CategoryRunner, MethodResult
from ._baselines import generate_baselines

_SIZE = 128


# ── Physics (copied from benchmarks/categories/electron_ctf.py) ──


def _compute_ctf(
    shape: Tuple[int, int],
    defocus_nm: float = 1000.0,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    pixel_nm: float = 1.0,
    amplitude_contrast: float = 0.07,
) -> np.ndarray:
    H, W = shape
    e = 1.602e-19
    m0 = 9.109e-31
    c = 3e8
    h = 6.626e-34
    V = voltage_kv * 1e3
    wavelength_m = h / np.sqrt(2 * m0 * e * V * (1 + e * V / (2 * m0 * c ** 2)))
    wavelength_nm = wavelength_m * 1e9
    cs_nm = cs_mm * 1e6

    fy = np.fft.fftfreq(H, d=pixel_nm)
    fx = np.fft.fftfreq(W, d=pixel_nm)
    FX, FY = np.meshgrid(fx, fy)
    s2 = FX ** 2 + FY ** 2
    chi = np.pi * wavelength_nm * defocus_nm * s2 - \
          0.5 * np.pi * cs_nm * wavelength_nm ** 3 * s2 ** 2
    w = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))
    ctf = -np.sin(chi + w)
    return np.fft.fftshift(ctf).astype(np.float32)


def _apply_ctf(image: np.ndarray, defocus_nm: float = 1000.0, **kwargs) -> np.ndarray:
    ctf = _compute_ctf(image.shape[:2], defocus_nm=defocus_nm, **kwargs)
    ft = np.fft.fftshift(np.fft.fft2(image))
    modulated = ft * ctf
    return np.real(np.fft.ifft2(np.fft.ifftshift(modulated))).astype(np.float32)


def _particle_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((H, W), dtype=np.float32)
    n_particles = max(10, int(H * W / 1500))
    for _ in range(n_particles):
        cx = int(rng.integers(5, W - 5))
        cy = int(rng.integers(5, H - 5))
        r = int(rng.integers(2, max(3, min(H, W) // 20)))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        x[mask] = float(rng.uniform(0.4, 1.0))
    return x


class ElectronCTFRunner(CategoryRunner):

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        phantom = _particle_phantom(_SIZE, _SIZE, rng)
        return phantom, "Particle phantom", "gray"

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        theta = config.get("theta") or {}
        defocus = theta.get("defocus_nm", 1000.0)
        voltage = theta.get("accelerating_voltage_kv", 300.0)
        modulated = _apply_ctf(phantom, defocus_nm=defocus, voltage_kv=voltage)
        return modulated, "CTF-modulated", "gray"

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        methods, labels = generate_baselines(config, "electron_ctf")
        sa = config.get("source_attribution") or {}
        gt_ref = sa.get("ground_truth", "")
        attribution = f"Electron microscopy simulation &mdash; {gt_ref}" if gt_ref else "Electron microscopy simulation (synthetic particles)"
        return methods, labels, config["display_name"], attribution
