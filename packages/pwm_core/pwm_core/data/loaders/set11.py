"""Synthetic Set11-style grayscale test images for SPC benchmarks."""

import numpy as np


class Set11Dataset:
    """Generates 11 synthetic grayscale images mimicking the Set11 CS benchmark.

    Each image uses a different deterministic pattern so solvers have
    meaningful structure to reconstruct (not random noise).
    """

    def __init__(self, resolution: int = 33):
        self.resolution = resolution
        self.images = self._generate()

    def _generate(self):
        n = self.resolution
        rng = np.random.RandomState(2024)
        y, x = np.mgrid[0:n, 0:n].astype(np.float32) / max(n - 1, 1)
        images = []

        # 1. Gaussian blob
        img = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.04)
        images.append(("gaussian_blob", img.astype(np.float32)))

        # 2. Concentric circles
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        img = 0.5 + 0.5 * np.cos(2 * np.pi * r * 6)
        images.append(("circles", img.astype(np.float32)))

        # 3. Horizontal stripes
        img = 0.5 + 0.5 * np.sin(2 * np.pi * y * 5)
        images.append(("stripes_h", img.astype(np.float32)))

        # 4. Checkerboard
        img = ((np.floor(x * 6) + np.floor(y * 6)) % 2).astype(np.float32)
        images.append(("checkerboard", img.astype(np.float32)))

        # 5. Shepp-Logan-like ellipses
        img = np.zeros((n, n), dtype=np.float32)
        for cx, cy, rx, ry, val in [
            (0.5, 0.5, 0.4, 0.3, 0.6),
            (0.5, 0.5, 0.2, 0.15, 0.3),
            (0.35, 0.5, 0.08, 0.08, 0.2),
            (0.65, 0.5, 0.08, 0.08, 0.2),
        ]:
            mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
            img[mask] += val
        img = np.clip(img, 0, 1)
        images.append(("phantom", img))

        # 6. Diagonal gradient
        img = ((x + y) / 2.0).astype(np.float32)
        images.append(("gradient", img))

        # 7. Sinusoidal texture
        img = 0.5 + 0.25 * np.sin(2 * np.pi * x * 4) + 0.25 * np.cos(2 * np.pi * y * 3)
        images.append(("sinusoidal", img.astype(np.float32)))

        # 8. Binary letter-like block pattern
        img = np.zeros((n, n), dtype=np.float32)
        q = max(n // 5, 1)
        img[q : 4 * q, q : 2 * q] = 1.0
        img[2 * q : 3 * q, q : 4 * q] = 1.0
        images.append(("block_letter", img))

        # 9. Crosshatch
        img = np.maximum(
            0.5 + 0.5 * np.sin(2 * np.pi * x * 7),
            0.5 + 0.5 * np.sin(2 * np.pi * y * 7),
        ).astype(np.float32)
        img = img / img.max()
        images.append(("crosshatch", img))

        # 10. Gabor-like pattern
        sigma = 0.15
        freq = 6.0
        envelope = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma**2))
        img = envelope * (0.5 + 0.5 * np.cos(2 * np.pi * freq * x))
        images.append(("gabor", img.astype(np.float32)))

        # 11. Smooth speckle (low-freq random)
        raw = rng.randn(max(n // 4, 2), max(n // 4, 2)).astype(np.float32)
        from scipy.ndimage import zoom as _zoom

        img = _zoom(raw, n / max(n // 4, 2), order=3)[:n, :n]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images.append(("speckle", img.astype(np.float32)))

        return images

    def __iter__(self):
        for name, img in self.images:
            yield name, img

    def __len__(self):
        return len(self.images)
