"""Real CACTI benchmark video sequences for snapshot compressive imaging.

Loads standard SCI benchmark videos (Kobe, Traffic, Runner, Drop, Crash, Aerial)
from .mat files used in the PnP-SCI / GAP-TV literature.
"""

import os

import numpy as np

_DEFAULT_DATA_DIR = (
    "/home/spiritai/PnP-SCI_python-master/dataset/cacti/grayscale_benchmark"
)

# Published GAP-TV reference PSNRs (approximate, from PnP-SCI literature)
_REF_PSNR = {
    "kobe": 26.5,
    "traffic": 20.5,
    "runner": 28.5,
    "drop": 34.3,
    "crash": 24.2,
    "aerial": 24.8,
}

# Video file mapping: name -> filename
_VIDEOS = [
    ("kobe", "kobe32_cacti.mat"),
    ("traffic", "traffic48_cacti.mat"),
    ("runner", "runner40_cacti.mat"),
    ("drop", "drop40_cacti.mat"),
    ("crash", "crash32_cacti.mat"),
    ("aerial", "aerial32_cacti.mat"),
]


class CACTIBenchmark:
    """Loads 6 standard CACTI benchmark videos from .mat files.

    Each .mat file contains:
      - orig: ground truth video (256x256xT), values in [0, 255]
      - mask: binary coded aperture (256x256x8)
      - meas: pre-computed measurements (256x256xT/8)

    Iterator yields (name, group_gt, mask, meas) for each 8-frame measurement
    group, where group_gt is (H,W,8) in [0,1], mask is (H,W,8) binary, and
    meas is (H,W) — a single compressed snapshot.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.environ.get(
            "CACTI_BENCHMARK_DIR", _DEFAULT_DATA_DIR
        )
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"CACTI benchmark data directory not found: {self.data_dir}\n"
                "Set CACTI_BENCHMARK_DIR environment variable to the correct path."
            )
        self._groups = self._load_all()

    def _load_all(self):
        from scipy.io import loadmat

        groups = []
        for name, filename in _VIDEOS:
            path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing benchmark file: {path}")

            mat = loadmat(path)
            orig = mat["orig"].astype(np.float64) / 255.0  # normalize to [0, 1]
            mask = mat["mask"].astype(np.float32)
            meas = mat["meas"].astype(np.float64) / 255.0  # normalize to match

            n_groups = orig.shape[2] // 8
            for g in range(n_groups):
                group_gt = orig[:, :, 8 * g : 8 * (g + 1)].astype(np.float32)
                group_meas = meas[:, :, g].astype(np.float32)
                groups.append((name, group_gt, mask, group_meas))

        return groups

    @property
    def video_names(self):
        """Return ordered list of unique video names."""
        seen = set()
        names = []
        for name, _, _, _ in self._groups:
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def __iter__(self):
        for item in self._groups:
            yield item

    def __len__(self):
        return len(self._groups)

    def get_reference_psnr(self, name: str) -> float:
        """Return published GAP-TV reference PSNR for a given video name."""
        return _REF_PSNR.get(name, 25.0)


class SyntheticCACTIDataset:
    """Generates 6 synthetic 256x256x8 grayscale video sequences for CACTI benchmarks.

    Used as a fallback when real .mat benchmark files are unavailable.
    Each video simulates a distinct motion pattern with smooth temporal
    transitions so GAP-TV solvers have meaningful structure to reconstruct.

    Iterator yields (name, video, mask) where:
      - video: (H, W, 8) float32 in [0, 1]
      - mask:  (H, W, 8) float32 binary coded aperture
    """

    def __init__(self, resolution: int = 256, num_frames: int = 8):
        self.resolution = resolution
        self.num_frames = num_frames
        self._items = self._generate()

    def _generate(self):
        n = self.resolution
        nF = self.num_frames
        rng = np.random.RandomState(2025)
        y_coord, x_coord = np.mgrid[0:n, 0:n].astype(np.float32) / max(n - 1, 1)
        items = []

        # Generate a shared random binary mask (H, W, nF)
        mask = (rng.rand(n, n, nF) > 0.5).astype(np.float32)

        # 1. Sliding blocks — rectangles translating horizontally
        video = np.zeros((n, n, nF), dtype=np.float32)
        for f in range(nF):
            shift = f / nF * 0.4
            for bx, by, bw, bh, val in [(0.1, 0.2, 0.2, 0.15, 0.8),
                                          (0.3, 0.5, 0.15, 0.2, 0.6),
                                          (0.05, 0.7, 0.25, 0.1, 0.9)]:
                cx = (bx + shift) % 1.0
                in_box = (x_coord >= cx) & (x_coord < cx + bw) & (y_coord >= by) & (y_coord < by + bh)
                video[:, :, f][in_box] = val
            video[:, :, f] += 0.1  # background
        video = np.clip(video, 0, 1)
        items.append(("sliding_blocks", video, mask.copy()))

        # 2. Diagonal flow — texture moving diagonally
        video = np.zeros((n, n, nF), dtype=np.float32)
        for f in range(nF):
            shift = f / nF * 0.3
            phase = x_coord + y_coord + shift
            video[:, :, f] = 0.5 + 0.4 * np.sin(2 * np.pi * 4 * phase)
        video = np.clip(video, 0, 1)
        items.append(("diagonal_flow", video, mask.copy()))

        # 3. Running silhouette — ellipse moving across frame
        video = np.zeros((n, n, nF), dtype=np.float32)
        video += 0.1  # background
        for f in range(nF):
            cx_f = 0.15 + 0.7 * f / (nF - 1)
            cy_f = 0.5 + 0.05 * np.sin(2 * np.pi * f / nF)
            r = ((x_coord - cx_f) / 0.06) ** 2 + ((y_coord - cy_f) / 0.15) ** 2
            silhouette = np.clip(1.0 - r, 0, 1)
            video[:, :, f] += 0.8 * silhouette
        video = np.clip(video, 0, 1)
        items.append(("running_silhouette", video, mask.copy()))

        # 4. Falling drops — circular blobs falling downward
        video = np.zeros((n, n, nF), dtype=np.float32)
        video += 0.05
        drop_centers = [(0.3, 0.1), (0.6, 0.2), (0.45, 0.05), (0.8, 0.15)]
        for f in range(nF):
            for dx, dy0 in drop_centers:
                dy = dy0 + 0.1 * f
                if dy < 1.0:
                    r2 = (x_coord - dx) ** 2 + (y_coord - dy) ** 2
                    drop = np.exp(-r2 / (2 * 0.03 ** 2))
                    video[:, :, f] += 0.7 * drop
        video = np.clip(video, 0, 1)
        items.append(("falling_drops", video, mask.copy()))

        # 5. Expanding wave — circular wave expanding from center
        video = np.zeros((n, n, nF), dtype=np.float32)
        r_dist = np.sqrt((x_coord - 0.5) ** 2 + (y_coord - 0.5) ** 2)
        for f in range(nF):
            radius = 0.05 + 0.4 * f / (nF - 1)
            wave = np.exp(-((r_dist - radius) ** 2) / (2 * 0.03 ** 2))
            video[:, :, f] = 0.1 + 0.8 * wave
        video = np.clip(video, 0, 1)
        items.append(("expanding_wave", video, mask.copy()))

        # 6. Smooth pan — textured image with horizontal camera pan
        from scipy.ndimage import gaussian_filter
        texture = rng.rand(n, n * 2).astype(np.float32)
        texture = gaussian_filter(texture, sigma=n / 16)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        video = np.zeros((n, n, nF), dtype=np.float32)
        for f in range(nF):
            offset = int(n * f / nF * 0.5)
            video[:, :, f] = texture[:, offset:offset + n]
        video = np.clip(video, 0, 1)
        items.append(("smooth_pan", video, mask.copy()))

        return items

    def __iter__(self):
        for name, video, m in self._items:
            yield name, video, m

    def __len__(self):
        return len(self._items)
