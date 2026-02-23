"""SyntheticDataGenerator – base class and utilities for generating
test phantoms and calibration patterns across imaging modalities.

Each category module (microscopy_psf, medical_ct_radon, etc.) subclasses
``SyntheticDataGenerator`` to provide modality-appropriate test objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SyntheticDataGenerator(ABC):
    """Base class for category-specific synthetic data generation.

    Subclasses implement ``generate()`` to produce modality-appropriate
    ground truth images (phantoms, test patterns, etc.).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate(
        self,
        dims: Tuple[int, ...],
        **kwargs,
    ) -> np.ndarray:
        """Generate a synthetic ground truth.

        Args:
            dims: Output shape, e.g. ``(256, 256)`` or ``(256, 256, 28)``.
            **kwargs: Modality-specific parameters.

        Returns:
            Ground truth array of the specified shape, values in [0, 1].
        """
        ...

    def generate_batch(
        self,
        dims: Tuple[int, ...],
        n: int = 4,
        **kwargs,
    ) -> List[np.ndarray]:
        """Generate multiple distinct phantoms."""
        results = []
        for i in range(n):
            self.rng = np.random.RandomState(self.seed + i)
            results.append(self.generate(dims, **kwargs))
        return results


# ---------------------------------------------------------------------------
# Standard phantoms (shared across categories)
# ---------------------------------------------------------------------------

class SheppLoganGenerator(SyntheticDataGenerator):
    """Modified Shepp-Logan phantom for 2-D / 3-D."""

    # Standard ellipse parameters (cx, cy, rx, ry, angle_deg, intensity)
    ELLIPSES = [
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

    def generate(self, dims: Tuple[int, ...], **kwargs) -> np.ndarray:
        H = dims[0]
        W = dims[1] if len(dims) >= 2 else H
        x = np.zeros((H, W), dtype=np.float64)

        yy = np.linspace(-1, 1, H)
        xx = np.linspace(-1, 1, W)
        X, Y = np.meshgrid(xx, yy)

        for cx, cy, rx, ry, angle, intensity in self.ELLIPSES:
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            Xr = cos_t * (X - cx) + sin_t * (Y - cy)
            Yr = -sin_t * (X - cx) + cos_t * (Y - cy)
            mask = (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1.0
            x[mask] += intensity

        x = (x - x.min()) / (x.max() - x.min() + 1e-12)

        if len(dims) >= 3:
            C = dims[2]
            x3d = np.zeros((H, W, C), dtype=np.float32)
            for c in range(C):
                scale = 0.7 + 0.6 * c / max(C - 1, 1)
                x3d[..., c] = np.clip(x * scale, 0, 1).astype(np.float32)
            return x3d
        return x.astype(np.float32)


class CellPhantomGenerator(SyntheticDataGenerator):
    """Fluorescence microscopy cell phantom with random ellipsoidal cells."""

    def generate(
        self,
        dims: Tuple[int, ...],
        n_cells: int = 20,
        **kwargs,
    ) -> np.ndarray:
        H, W = dims[0], dims[1] if len(dims) >= 2 else dims[0]
        x = np.zeros((H, W), dtype=np.float32)

        for _ in range(n_cells):
            cx = self.rng.randint(int(W * 0.1), int(W * 0.9))
            cy = self.rng.randint(int(H * 0.1), int(H * 0.9))
            rx = self.rng.randint(max(3, W // 30), max(5, W // 10))
            ry = self.rng.randint(max(3, H // 30), max(5, H // 10))
            intensity = self.rng.rand() * 0.7 + 0.3
            yy, xx = np.ogrid[:H, :W]
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            x[mask] = np.maximum(x[mask], intensity)

        if len(dims) >= 3:
            C = dims[2]
            x3d = np.stack([
                np.clip(x * (0.5 + 0.5 * self.rng.rand()), 0, 1)
                for _ in range(C)
            ], axis=-1)
            return x3d.astype(np.float32)
        return x


class BeadPhantomGenerator(SyntheticDataGenerator):
    """Sub-diffraction bead phantom for PSF calibration."""

    def generate(
        self,
        dims: Tuple[int, ...],
        n_beads: int = 10,
        bead_radius: int = 2,
        **kwargs,
    ) -> np.ndarray:
        H, W = dims[0], dims[1] if len(dims) >= 2 else dims[0]
        x = np.zeros((H, W), dtype=np.float32)

        for _ in range(n_beads):
            cx = self.rng.randint(bead_radius + 5, W - bead_radius - 5)
            cy = self.rng.randint(bead_radius + 5, H - bead_radius - 5)
            yy, xx = np.ogrid[:H, :W]
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            x[r2 <= bead_radius ** 2] = 1.0

        if len(dims) >= 3:
            C = dims[2]
            return np.stack([x] * C, axis=-1)
        return x


class ResolutionTargetGenerator(SyntheticDataGenerator):
    """USAF-style resolution target with bar groups."""

    def generate(self, dims: Tuple[int, ...], **kwargs) -> np.ndarray:
        H, W = dims[0], dims[1] if len(dims) >= 2 else dims[0]
        x = np.zeros((H, W), dtype=np.float32)

        # Create bar groups at different scales
        n_groups = 6
        bar_region_h = H // (n_groups + 1)
        for g in range(n_groups):
            period = max(2, W // (4 * (g + 1)))
            y_start = (g + 1) * bar_region_h - bar_region_h // 4
            y_end = (g + 1) * bar_region_h + bar_region_h // 4
            for col in range(W):
                if (col // max(1, period // 2)) % 2 == 0:
                    x[y_start:y_end, col] = 1.0

        if len(dims) >= 3:
            return np.stack([x] * dims[2], axis=-1)
        return x


class SpectralSceneGenerator(SyntheticDataGenerator):
    """Spectral scene for hyperspectral/compressive modalities.

    Generates ``(H, W, C)`` cube with spatially varying spectra.
    """

    def generate(
        self,
        dims: Tuple[int, ...],
        n_regions: int = 8,
        **kwargs,
    ) -> np.ndarray:
        if len(dims) < 3:
            dims = (*dims, 28)
        H, W, C = dims[0], dims[1], dims[2]
        x = np.zeros((H, W, C), dtype=np.float32)

        # Random spectral basis
        spectra = []
        for _ in range(n_regions):
            center = self.rng.rand() * C
            width = self.rng.rand() * C * 0.3 + C * 0.05
            spec = np.exp(-0.5 * ((np.arange(C) - center) / width) ** 2)
            spec = spec / (spec.max() + 1e-12)
            spectra.append(spec)

        # Place regions
        for i in range(n_regions):
            cx = self.rng.randint(int(W * 0.1), int(W * 0.9))
            cy = self.rng.randint(int(H * 0.1), int(H * 0.9))
            r = self.rng.randint(max(3, min(H, W) // 15), max(5, min(H, W) // 5))
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            intensity = self.rng.rand() * 0.7 + 0.3
            x[mask] += intensity * spectra[i][None, :]

        return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "shepp_logan": SheppLoganGenerator,
    "cell_phantom": CellPhantomGenerator,
    "bead_phantom": BeadPhantomGenerator,
    "resolution_target": ResolutionTargetGenerator,
    "spectral_scene": SpectralSceneGenerator,
}


def get_generator(name: str, seed: int = 42) -> SyntheticDataGenerator:
    """Look up a generator by name."""
    cls = GENERATORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown generator: {name!r}. Available: {list(GENERATORS)}")
    return cls(seed=seed)
