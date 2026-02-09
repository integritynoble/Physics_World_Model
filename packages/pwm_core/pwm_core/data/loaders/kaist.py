"""Synthetic KAIST-style hyperspectral cubes for CASSI fallback benchmarks."""

import numpy as np


class KAISTDataset:
    """Generates synthetic hyperspectral cubes mimicking the KAIST dataset.

    Each cube has smooth spatial patterns with spectrally varying Gaussian
    reflectance profiles, giving solvers meaningful spectral structure to
    reconstruct.
    """

    def __init__(self, resolution: int = 256, num_bands: int = 28):
        self.resolution = resolution
        self.num_bands = num_bands
        self.cubes = self._generate()

    def _generate(self):
        n = self.resolution
        nB = self.num_bands
        rng = np.random.RandomState(2024)
        y, x = np.mgrid[0:n, 0:n].astype(np.float32) / max(n - 1, 1)
        bands = np.linspace(0, 1, nB, dtype=np.float32)
        cubes = []

        # 1. Blobs with distinct spectral peaks
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for _ in range(8):
            cx, cy = rng.uniform(0.15, 0.85, 2)
            sigma_s = rng.uniform(0.08, 0.2)
            spatial = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_s**2))
            peak = rng.uniform(0.1, 0.9)
            sigma_b = rng.uniform(0.08, 0.25)
            spectral = np.exp(-((bands - peak) ** 2) / (2 * sigma_b**2))
            cube += spatial[:, :, None] * spectral[None, None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("blobs_spectral", cube))

        # 2. Striped scene with spectral gradient
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for k in range(5):
            freq = 2 + k
            stripe = 0.5 + 0.5 * np.sin(2 * np.pi * freq * x)
            center = k / 4.0
            spectral = np.exp(-((bands - center) ** 2) / 0.05)
            cube += stripe[:, :, None] * spectral[None, None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("stripes_spectral", cube))

        # 3. Checkerboard with smooth spectral variation
        check = ((np.floor(x * 6) + np.floor(y * 6)) % 2).astype(np.float32)
        base = 0.2 + 0.3 * check
        spectral_slope = bands[None, None, :]
        cube = base[:, :, None] * (0.3 + 0.7 * spectral_slope)
        # Add a couple of spatial blobs for variety
        for cx, cy, pk in [(0.3, 0.3, 0.2), (0.7, 0.7, 0.7)]:
            spatial = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.02)
            spectral = np.exp(-((bands - pk) ** 2) / 0.03)
            cube += 0.4 * spatial[:, :, None] * spectral[None, None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("checker_spectral", cube))

        return cubes

    def __iter__(self):
        for name, cube in self.cubes:
            yield name, cube

    def __len__(self):
        return len(self.cubes)
