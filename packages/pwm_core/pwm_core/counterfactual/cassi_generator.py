"""pwm_core.counterfactual.cassi_generator
==========================================

CASSI counterfactual pack generator.

Source: 10 KAIST scenes (256x256x28) with coded aperture mask.
Forward model: 5-param mismatch (mask_dx, mask_dy, mask_theta, disp_a1, disp_alpha)
  + Poisson-Gaussian noise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import affine_transform

from pwm_core.counterfactual.base_generator import (
    BaseCounterfactualGenerator,
    add_poisson_gaussian_noise,
)
from pwm_core.counterfactual.red_team import get_red_team_configs
from pwm_core.counterfactual.schema import SplitKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_DIR = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
MASK_PATH = DATASET_DIR / "mask.mat"
TRUTH_DIR = DATASET_DIR / "Truth"

SCENE_NAMES = [
    "scene01", "scene02", "scene03", "scene04", "scene05",
    "scene06", "scene07", "scene08", "scene09", "scene10",
]

STEP = 2
N_BANDS = 28

# ---------------------------------------------------------------------------
# PARAM_CONFIG: probe vs hidden tiers
# ---------------------------------------------------------------------------

PARAM_CONFIG = {
    SplitKind.probe: {
        "mask_dx":     {"min": -1.5,  "max": 1.5,    "unit": "px"},
        "mask_dy":     {"min": -1.0,  "max": 1.0,    "unit": "px"},
        "mask_theta":  {"min": -0.3,  "max": 0.3,    "unit": "deg"},
        "disp_a1":     {"min": 1.95,  "max": 2.05,   "unit": "px/band"},
        "disp_alpha":  {"min": -0.5,  "max": 0.5,    "unit": "deg"},
    },
    SplitKind.hidden: {
        "mask_dx":     {"min": -3.0,  "max": 3.0,    "unit": "px"},
        "mask_dy":     {"min": -3.0,  "max": 3.0,    "unit": "px"},
        "mask_theta":  {"min": -0.6,  "max": 0.6,    "unit": "deg"},
        "disp_a1":     {"min": 1.85,  "max": 2.15,   "unit": "px/band"},
        "disp_alpha":  {"min": -1.0,  "max": 1.0,    "unit": "deg"},
    },
}

NOISE_CONFIG = {
    SplitKind.probe:  {"noise_alpha": 100_000.0, "noise_sigma": 0.01},
    SplitKind.hidden: {"noise_alpha": 30_000.0,  "noise_sigma": 0.03},
}


# ---------------------------------------------------------------------------
# Forward-model helpers (self-contained, copied from validate_cassi_pwmi.py)
# ---------------------------------------------------------------------------


def warp_affine_2d(
    mask: np.ndarray, dx: float, dy: float, theta: float
) -> np.ndarray:
    """Apply 2D affine transformation to mask (translation + rotation).

    Args:
        mask: (H, W) input mask.
        dx: x-translation in pixels.
        dy: y-translation in pixels.
        theta: rotation in degrees.

    Returns:
        Warped mask (H, W), clipped to [0, 1].
    """
    H, W = mask.shape
    cx, cy = W / 2.0, H / 2.0

    th = np.radians(theta)
    cos_t, sin_t = np.cos(th), np.sin(th)

    mat = np.array([
        [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
        [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
    ])

    inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
    warped = affine_transform(
        mask, inv[:2, :2], offset=inv[:2, 2], cval=0, order=1
    )
    return np.clip(warped, 0, 1).astype(np.float32)


def _compute_band_offsets(
    nC: int, a1: float = 2.0, alpha: float = 0.0
) -> List[Tuple[int, int]]:
    """Compute per-band integer dispersion offsets (dx_k, dy_k)."""
    cos_a = np.cos(np.radians(alpha))
    sin_a = np.sin(np.radians(alpha))
    return [
        (int(round(a1 * k * cos_a)), int(round(a1 * k * sin_a)))
        for k in range(nC)
    ]


def cassi_forward_parametric(
    scene: np.ndarray,
    mask: np.ndarray,
    a1: float = 2.0,
    alpha: float = 0.0,
) -> np.ndarray:
    """Native-resolution parametric CASSI forward model.

    y[:, dx_k : dx_k+W] += mask * scene[:,:,k]
    where dx_k = round(a1 * k * cos(alpha)).
    """
    H, W, nC = scene.shape
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    W_ext = W + max(max_dx, 0)
    y = np.zeros((H, W_ext), dtype=np.float32)

    for k in range(nC):
        dx_k = offsets[k][0]
        if 0 <= dx_k and dx_k + W <= W_ext:
            y[:, dx_k:dx_k + W] += mask * scene[:, :, k]
    return y


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Load mask from MATLAB .mat file."""
    try:
        data = loadmat(str(path))
        for key in ["mask", "Mask", "mask_data"]:
            if key in data:
                m = data[key]
                if isinstance(m, np.ndarray):
                    return m.astype(np.float32)
    except Exception as e:
        logger.warning("Failed to load mask from %s: %s", path, e)
    return None


def _load_scene(scene_name: str) -> Optional[np.ndarray]:
    """Load scene from MATLAB .mat file (256x256x28)."""
    for subdir in [TRUTH_DIR, DATASET_DIR]:
        path = subdir / f"{scene_name}.mat"
        if not path.exists():
            continue
        try:
            data = loadmat(str(path))
            for key in ["img", "Img", "scene", "Scene", "data"]:
                if key in data:
                    s = data[key].astype(np.float32)
                    if s.ndim == 3 and s.shape[2] == N_BANDS:
                        return s
        except Exception as e:
            logger.warning("Failed to load scene %s: %s", scene_name, e)
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CassiCounterfactualGenerator(BaseCounterfactualGenerator):
    """CASSI counterfactual pack generator.

    10 KAIST scenes, 5-param mismatch model, 22 scenarios per scene per split.
    Total: 440 scenarios (220 probe + 220 hidden).
    """

    def __init__(
        self,
        seed_public: int = 2026_02_18,
        seed_hidden: int = 9999_02_18,
    ):
        super().__init__(
            modality="cassi",
            pack_id="cassi_cfpack_v1",
            seed_public=seed_public,
            seed_hidden=seed_hidden,
        )

    def load_scenes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load 10 KAIST scenes and mask."""
        mask = _load_mask(MASK_PATH)
        if mask is None:
            raise FileNotFoundError(f"Cannot load CASSI mask from {MASK_PATH}")

        scenes = []
        for name in SCENE_NAMES:
            s = _load_scene(name)
            if s is not None:
                scenes.append((name, s, mask))
            else:
                logger.warning("Skipping scene %s (not found)", name)
        if not scenes:
            raise FileNotFoundError(
                f"No CASSI scenes found in {TRUTH_DIR}"
            )
        return scenes

    def forward_model(
        self,
        x_gt: np.ndarray,
        mask: np.ndarray,
        mismatch_params: Dict[str, float],
        noise_config: Dict[str, float],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CASSI forward model with mismatch and noise."""
        dx = mismatch_params.get("mask_dx", 0.0)
        dy = mismatch_params.get("mask_dy", 0.0)
        theta = mismatch_params.get("mask_theta", 0.0)
        a1 = mismatch_params.get("disp_a1", 2.0)
        alpha = mismatch_params.get("disp_alpha", 0.0)

        # Warp mask
        if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(theta) > 1e-6:
            mask_warped = warp_affine_2d(mask, dx, dy, theta)
        else:
            mask_warped = mask.copy()

        # Forward model
        y_clean = cassi_forward_parametric(x_gt, mask_warped, a1=a1, alpha=alpha)

        # Add noise
        noise_alpha = noise_config.get("noise_alpha", 100_000.0)
        noise_sigma = noise_config.get("noise_sigma", 0.01)
        y_noisy = add_poisson_gaussian_noise(
            y_clean, peak=noise_alpha, sigma=noise_sigma, rng=rng
        )

        return y_noisy, mask_warped

    def get_param_config(self, split: SplitKind) -> Dict[str, Dict[str, float]]:
        return PARAM_CONFIG[split]

    def get_red_team_configs(self, split: SplitKind) -> Dict[str, Dict[str, Any]]:
        return get_red_team_configs("cassi")

    def get_nominal_noise(self, split: SplitKind) -> Dict[str, float]:
        return NOISE_CONFIG[split]
