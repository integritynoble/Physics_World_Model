"""pwm_core.counterfactual.spc_generator
========================================

Single-Pixel Camera (SPC) counterfactual pack generator.

Source: 11 Set11 images (grayscale), 33x33 block-based with Phi matrix.
Forward model: 5-param mismatch (gain_alpha, gain_offset, dark_current,
  noise_sigma, mask_perturbation) with Gaussian noise.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from pwm_core.counterfactual.base_generator import (
    BaseCounterfactualGenerator,
    add_gaussian_noise,
)
from pwm_core.counterfactual.red_team import get_red_team_configs
from pwm_core.counterfactual.schema import SplitKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SET11_DIR = Path("/home/spiritai/ISTA-Net-PyTorch-master/data/Set11")
PHI_PATH = Path(
    "/home/spiritai/ISTA-Net-PyTorch-master/sampling_matrix/phi_0_25_1089.mat"
)

BLOCK_SIZE = 33
N_PIX = BLOCK_SIZE * BLOCK_SIZE  # 1089
M_MEAS = 272  # 25% of 1089

# ---------------------------------------------------------------------------
# PARAM_CONFIG
# ---------------------------------------------------------------------------

PARAM_CONFIG = {
    SplitKind.probe: {
        "gain_alpha":       {"min": 0.0005, "max": 0.003,  "unit": "1/row"},
        "gain_offset":      {"min": -0.005, "max": 0.005,  "unit": "norm"},
        "dark_current":     {"min": 0.0,    "max": 0.01,   "unit": "norm"},
        "noise_sigma":      {"min": 0.01,   "max": 0.03,   "unit": "norm"},
        "mask_perturbation": {"min": 0.0,   "max": 0.005,  "unit": "norm"},
    },
    SplitKind.hidden: {
        "gain_alpha":       {"min": 0.003,  "max": 0.01,   "unit": "1/row"},
        "gain_offset":      {"min": -0.02,  "max": 0.02,   "unit": "norm"},
        "dark_current":     {"min": 0.0,    "max": 0.04,   "unit": "norm"},
        "noise_sigma":      {"min": 0.03,   "max": 0.10,   "unit": "norm"},
        "mask_perturbation": {"min": 0.0,   "max": 0.02,   "unit": "norm"},
    },
}

NOISE_CONFIG = {
    SplitKind.probe:  {"noise_sigma": 0.02},
    SplitKind.hidden: {"noise_sigma": 0.05},
}


# ---------------------------------------------------------------------------
# Forward model helpers
# ---------------------------------------------------------------------------


def make_gain_vector_exp(m: int, alpha: float) -> np.ndarray:
    """Exponential gain drift: g_i = exp(-alpha * i)."""
    i = np.arange(m, dtype=np.float32)
    return np.exp(-alpha * i)


def imread_cs(img_y: np.ndarray) -> Tuple[np.ndarray, int, int, np.ndarray, int, int]:
    """Pad image to multiple of BLOCK_SIZE."""
    row, col = img_y.shape
    row_pad = (BLOCK_SIZE - row % BLOCK_SIZE) % BLOCK_SIZE
    col_pad = (BLOCK_SIZE - col % BLOCK_SIZE) % BLOCK_SIZE
    ipad = np.zeros((row + row_pad, col + col_pad), dtype=img_y.dtype)
    ipad[:row, :col] = img_y
    row_new, col_new = ipad.shape
    return img_y, row, col, ipad, row_new, col_new


def img2col(ipad: np.ndarray) -> np.ndarray:
    """Extract BLOCK_SIZE x BLOCK_SIZE column blocks. Returns (N_PIX, n_blocks)."""
    row, col = ipad.shape
    row_block = row // BLOCK_SIZE
    col_block = col // BLOCK_SIZE
    n_blocks = row_block * col_block
    img_col = np.zeros((N_PIX, n_blocks), dtype=np.float32)
    count = 0
    for x in range(0, row - BLOCK_SIZE + 1, BLOCK_SIZE):
        for y in range(0, col - BLOCK_SIZE + 1, BLOCK_SIZE):
            img_col[:, count] = ipad[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE].reshape(-1)
            count += 1
    return img_col


def _load_image(path: Path) -> Optional[np.ndarray]:
    """Load grayscale image, return as float32 [0, 1]."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float32) / 255.0
    except ImportError:
        pass
    # Fallback: try PIL
    try:
        from PIL import Image
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        return img
    except Exception as e:
        logger.warning("Failed to load image %s: %s", path, e)
    return None


def _load_phi(path: Path) -> Optional[np.ndarray]:
    """Load Phi sampling matrix from .mat file. Returns (M, N)."""
    try:
        data = loadmat(str(path))
        if "phi" in data:
            return data["phi"].astype(np.float32)
    except Exception as e:
        logger.warning("Failed to load Phi from %s: %s", path, e)
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SpcCounterfactualGenerator(BaseCounterfactualGenerator):
    """SPC counterfactual pack generator.

    11 Set11 images, 5-param gain/noise model, 22 scenarios per image per split.
    Total: 484 scenarios (242 probe + 242 hidden).
    """

    def __init__(
        self,
        seed_public: int = 2026_02_18,
        seed_hidden: int = 9999_02_18,
    ):
        super().__init__(
            modality="spc",
            pack_id="spc_cfpack_v1",
            seed_public=seed_public,
            seed_hidden=seed_hidden,
        )

    def load_scenes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load 11 Set11 images and Phi matrix.

        Returns list of (image_id, x_blocks, phi) where:
          - x_blocks: (n_blocks, N_PIX) column blocks, values in [0, 1]
          - phi: (M, N) measurement matrix
        """
        phi = _load_phi(PHI_PATH)
        if phi is None:
            raise FileNotFoundError(f"Cannot load Phi matrix from {PHI_PATH}")

        # Find all images
        patterns = ["*.tif", "*.png", "*.bmp", "*.jpg"]
        image_paths: List[Path] = []
        for pat in patterns:
            image_paths.extend(sorted(SET11_DIR.glob(pat)))
        image_paths = sorted(set(image_paths))

        scenes = []
        for img_path in image_paths:
            img = _load_image(img_path)
            if img is None:
                continue
            _, row, col, ipad, row_new, col_new = imread_cs(img)
            x_col = img2col(ipad)  # (N_PIX, n_blocks)
            x_blocks = x_col.T  # (n_blocks, N_PIX)
            image_id = img_path.stem
            scenes.append((image_id, x_blocks, phi))

        if not scenes:
            raise FileNotFoundError(f"No images found in {SET11_DIR}")
        return scenes

    def forward_model(
        self,
        x_gt: np.ndarray,
        mask: np.ndarray,
        mismatch_params: Dict[str, float],
        noise_config: Dict[str, float],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SPC forward model with gain drift and noise.

        Args:
            x_gt: (n_blocks, N_PIX) column blocks.
            mask: (M, N) measurement matrix Phi.
            mismatch_params: gain_alpha, gain_offset, dark_current, etc.
            noise_config: noise_sigma.
            rng: Generator for reproducibility.

        Returns:
            (y_meas, phi_corrupted) — measurement and corrupted operator.
        """
        gain_alpha = mismatch_params.get("gain_alpha", 0.0)
        gain_offset = mismatch_params.get("gain_offset", 0.0)
        dark_current = mismatch_params.get("dark_current", 0.0)
        noise_sigma = noise_config.get("noise_sigma", 0.01)
        mask_perturbation = mismatch_params.get("mask_perturbation", 0.0)

        phi = mask.copy()
        M, N = phi.shape

        # Apply mask perturbation (additive Gaussian noise to Phi entries)
        if mask_perturbation > 0:
            phi = phi + mask_perturbation * rng.standard_normal(phi.shape).astype(
                np.float32
            )

        # Apply gain drift: g = exp(-alpha * i) + offset
        g = make_gain_vector_exp(M, gain_alpha) + gain_offset
        phi_real = g[:, np.newaxis] * phi  # (M, N)

        # Compute clean measurement
        y_clean = x_gt @ phi_real.T  # (n_blocks, M)

        # Add dark current
        y_clean = y_clean + dark_current

        # Add Gaussian noise
        y_meas = add_gaussian_noise(y_clean, sigma=noise_sigma, rng=rng)

        return y_meas, phi_real

    def get_param_config(self, split: SplitKind) -> Dict[str, Dict[str, float]]:
        return PARAM_CONFIG[split]

    def get_red_team_configs(self, split: SplitKind) -> Dict[str, Dict[str, Any]]:
        return get_red_team_configs("spc")

    def get_nominal_noise(self, split: SplitKind) -> Dict[str, float]:
        return NOISE_CONFIG[split]
