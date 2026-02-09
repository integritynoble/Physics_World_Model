"""Synthetic CACTI benchmark video sequences for snapshot compressive imaging."""

import numpy as np


class CACTIBenchmark:
    """Generates 6 synthetic grayscale video clips for CACTI benchmarks.

    Each clip has shape (H, W, nF) with different motion patterns so
    temporal-compressive solvers have meaningful structure to recover.
    """

    _REF_PSNR = {
        "moving_disk": 26.0,
        "expanding_circle": 25.0,
        "rotating_bar": 24.5,
        "scrolling_stripes": 27.0,
        "bouncing_ball": 25.5,
        "pulsing_gaussian": 28.0,
    }

    def __init__(self, height: int = 256, width: int = 256, num_frames: int = 8):
        self.h = height
        self.w = width
        self.nF = num_frames
        self.videos = self._generate()

    def _generate(self):
        h, w, nF = self.h, self.w, self.nF
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        yc, xc = h / 2.0, w / 2.0
        videos = []

        # 1. Moving disk — translates right across frames
        vid = np.zeros((h, w, nF), dtype=np.float32)
        radius = h * 0.12
        for f in range(nF):
            cx = w * 0.2 + (w * 0.6) * f / max(nF - 1, 1)
            dist = np.sqrt((x - cx) ** 2 + (y - yc) ** 2)
            vid[:, :, f] = np.clip(1.0 - dist / radius, 0, 1)
        videos.append(("moving_disk", vid))

        # 2. Expanding circle — ring expanding outward
        vid = np.zeros((h, w, nF), dtype=np.float32)
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        for f in range(nF):
            ring_r = h * 0.08 + h * 0.3 * f / max(nF - 1, 1)
            vid[:, :, f] = np.exp(-((r - ring_r) ** 2) / (2 * (h * 0.03) ** 2))
        videos.append(("expanding_circle", vid))

        # 3. Rotating bar
        vid = np.zeros((h, w, nF), dtype=np.float32)
        for f in range(nF):
            angle = np.pi * f / max(nF - 1, 1)
            dx = x - xc
            dy = y - yc
            proj = dx * np.cos(angle) + dy * np.sin(angle)
            perp = -dx * np.sin(angle) + dy * np.cos(angle)
            bar = (np.abs(perp) < h * 0.04) & (np.abs(proj) < h * 0.35)
            vid[:, :, f] = bar.astype(np.float32)
        videos.append(("rotating_bar", vid))

        # 4. Scrolling stripes — vertical stripes moving horizontally
        vid = np.zeros((h, w, nF), dtype=np.float32)
        for f in range(nF):
            phase = 2 * np.pi * f / nF
            vid[:, :, f] = 0.5 + 0.5 * np.sin(2 * np.pi * x / (w / 4.0) + phase)
        videos.append(("scrolling_stripes", vid))

        # 5. Bouncing ball — ball moves diagonally and bounces
        vid = np.zeros((h, w, nF), dtype=np.float32)
        radius = h * 0.08
        for f in range(nF):
            t = f / max(nF - 1, 1)
            bx = w * (0.2 + 0.6 * t)
            by_raw = h * (0.3 + 0.4 * t)
            by = h * 0.3 + abs(by_raw - h * 0.5) * 0.8  # simple bounce
            dist = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
            vid[:, :, f] = np.exp(-(dist**2) / (2 * radius**2))
        videos.append(("bouncing_ball", vid))

        # 6. Pulsing Gaussian — intensity modulation
        vid = np.zeros((h, w, nF), dtype=np.float32)
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        sigma2 = (h * 0.2) ** 2
        for f in range(nF):
            amp = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * f / nF))
            vid[:, :, f] = amp * np.exp(-r2 / (2 * sigma2))
        videos.append(("pulsing_gaussian", vid))

        return videos

    def __iter__(self):
        for name, vid in self.videos:
            yield name, vid

    def __len__(self):
        return len(self.videos)

    def get_reference_psnr(self, name: str) -> float:
        """Return conservative reference PSNR for a given video name."""
        return self._REF_PSNR.get(name, 25.0)
