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
