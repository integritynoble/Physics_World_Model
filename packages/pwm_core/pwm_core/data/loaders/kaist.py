"""Synthetic KAIST-style hyperspectral cubes for CASSI fallback benchmarks."""

import numpy as np


class KAISTDataset:
    """Generates synthetic hyperspectral cubes mimicking the KAIST dataset.

    Each cube has smooth spatial patterns with spectrally varying Gaussian
    reflectance profiles, giving solvers meaningful spectral structure to
    reconstruct.  Produces 10 distinct scenes.
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

        # 4. Radial star — wedge pattern with spectral modulation
        angle = np.arctan2(y - 0.5, x - 0.5)
        n_arms = 12
        star = 0.5 + 0.5 * np.cos(n_arms * angle)
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        envelope = np.clip(1.0 - r / 0.5, 0, 1)
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for b in range(nB):
            phase = 2 * np.pi * b / nB
            cube[:, :, b] = envelope * (0.3 + 0.7 * (0.5 + 0.5 * np.cos(n_arms * angle + phase)))
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("radial_star", cube))

        # 5. Phantom — Shepp-Logan-like ellipses with spectral peaks
        cube = np.zeros((n, n, nB), dtype=np.float32)
        ellipses = [
            (0.5, 0.5, 0.42, 0.32, 0.5, 0.3),
            (0.5, 0.5, 0.22, 0.16, 0.3, 0.6),
            (0.35, 0.45, 0.09, 0.09, 0.2, 0.15),
            (0.65, 0.45, 0.09, 0.09, 0.2, 0.75),
            (0.5, 0.7, 0.15, 0.06, 0.15, 0.5),
        ]
        for cx_e, cy_e, rx, ry, val, pk in ellipses:
            mask = ((x - cx_e) / rx) ** 2 + ((y - cy_e) / ry) ** 2 <= 1.0
            spectral = np.exp(-((bands - pk) ** 2) / 0.04)
            cube[mask] += val * spectral[None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("phantom", cube))

        # 6. Voronoi mosaic — random cell colors with distinct spectra
        n_cells = 20
        seeds_x = rng.uniform(0, 1, n_cells).astype(np.float32)
        seeds_y = rng.uniform(0, 1, n_cells).astype(np.float32)
        cell_peaks = rng.uniform(0.05, 0.95, n_cells).astype(np.float32)
        cell_vals = rng.uniform(0.3, 1.0, n_cells).astype(np.float32)
        # Assign each pixel to nearest seed
        dist = np.full((n, n), np.inf, dtype=np.float32)
        labels = np.zeros((n, n), dtype=np.int32)
        for ci in range(n_cells):
            d = (x - seeds_x[ci]) ** 2 + (y - seeds_y[ci]) ** 2
            closer = d < dist
            dist[closer] = d[closer]
            labels[closer] = ci
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for ci in range(n_cells):
            mask = labels == ci
            spectral = cell_vals[ci] * np.exp(-((bands - cell_peaks[ci]) ** 2) / 0.06)
            cube[mask] += spectral[None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("voronoi_mosaic", cube))

        # 7. Gabor field — oriented Gabor patches tiling the image
        cube = np.zeros((n, n, nB), dtype=np.float32)
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        for i, theta in enumerate(orientations):
            freq_g = 5.0 + 2 * i
            xr = (x - 0.5) * np.cos(theta) + (y - 0.5) * np.sin(theta)
            envelope_g = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.15 ** 2))
            pattern = envelope_g * (0.5 + 0.5 * np.cos(2 * np.pi * freq_g * xr))
            peak_b = (i + 0.5) / len(orientations)
            spectral = np.exp(-((bands - peak_b) ** 2) / 0.04)
            cube += pattern[:, :, None] * spectral[None, None, :]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("gabor_field", cube))

        # 8. Concentric rings — bullseye with spectral oscillation
        r_ring = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for b in range(nB):
            freq_r = 3 + 0.5 * b / nB
            cube[:, :, b] = 0.5 + 0.5 * np.cos(2 * np.pi * freq_r * r_ring + b * 0.3)
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("concentric_rings", cube))

        # 9. Low-frequency random field — smooth spatial + spectral variation
        from scipy.ndimage import gaussian_filter
        cube = np.zeros((n, n, nB), dtype=np.float32)
        for b in range(nB):
            raw = rng.randn(n, n).astype(np.float32)
            smooth = gaussian_filter(raw, sigma=n / 8)
            smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
            cube[:, :, b] = smooth
        # Add spectral correlation by blending neighboring bands
        for b in range(1, nB):
            cube[:, :, b] = 0.6 * cube[:, :, b] + 0.4 * cube[:, :, b - 1]
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("lowfreq_random", cube))

        # 10. Letter blocks — block letters with distinct spectral signatures
        cube = np.zeros((n, n, nB), dtype=np.float32)
        q = max(n // 8, 1)
        bg_spec = np.exp(-((bands - 0.2) ** 2) / 0.1)
        cube += 0.15 * bg_spec[None, None, :]
        # Letter P
        block_spec_p = np.exp(-((bands - 0.7) ** 2) / 0.04)
        cube[q:7*q, q:2*q, :] += 0.7 * block_spec_p[None, :]       # vertical bar
        cube[q:2*q, q:4*q, :] += 0.7 * block_spec_p[None, :]       # top bar
        cube[3*q:4*q, q:4*q, :] += 0.7 * block_spec_p[None, :]     # mid bar
        cube[q:4*q, 3*q:4*q, :] += 0.7 * block_spec_p[None, :]     # right bar (top half)
        # Letter W
        block_spec_w = np.exp(-((bands - 0.4) ** 2) / 0.04)
        ox = 5 * q
        cube[q:7*q, ox:ox+q, :] += 0.7 * block_spec_w[None, :]            # left bar
        cube[q:7*q, ox+3*q:ox+4*q, :] += 0.7 * block_spec_w[None, :]      # right bar
        cube[5*q:7*q, ox+q:ox+2*q, :] += 0.7 * block_spec_w[None, :]      # left mid
        cube[5*q:7*q, ox+2*q:ox+3*q, :] += 0.7 * block_spec_w[None, :]    # right mid
        cube[6*q:7*q, ox:ox+4*q, :] += 0.7 * block_spec_w[None, :]        # bottom bar
        cube = np.clip(cube / (cube.max() + 1e-8), 0, 1)
        cubes.append(("letter_blocks", cube))

        return cubes

    def __iter__(self):
        for name, cube in self.cubes:
            yield name, cube

    def __len__(self):
        return len(self.cubes)
