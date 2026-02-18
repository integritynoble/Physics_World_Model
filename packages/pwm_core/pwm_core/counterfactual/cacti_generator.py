"""pwm_core.counterfactual.cacti_generator
==========================================

CACTI (Coded Aperture Compressive Temporal Imaging) counterfactual pack generator.

Source: 6 grayscale benchmark videos (256x256x8) from PnP-SCI dataset.
Forward model: 8-param mismatch (mask_dx, mask_dy, mask_theta, mask_blur_sigma,
  clock_offset, duty_cycle, gain, offset) + Poisson-Gaussian noise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import affine_transform, gaussian_filter

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

BENCHMARK_DIR = Path(
    "/home/spiritai/PnP-SCI_python-master/dataset/cacti/grayscale_benchmark"
)

# ---------------------------------------------------------------------------
# PARAM_CONFIG
# ---------------------------------------------------------------------------

PARAM_CONFIG = {
    SplitKind.probe: {
        "mask_dx":         {"min": -2.0,    "max": 2.0,    "unit": "px"},
        "mask_dy":         {"min": -2.0,    "max": 2.0,    "unit": "px"},
        "mask_theta":      {"min": -0.25,   "max": 0.25,   "unit": "deg"},
        "mask_blur_sigma": {"min": 0.0,     "max": 0.5,    "unit": "px"},
        "clock_offset":    {"min": -0.1,    "max": 0.1,    "unit": "frames"},
        "duty_cycle":      {"min": 0.90,    "max": 1.0,    "unit": "ratio"},
        "gain":            {"min": 0.95,    "max": 1.05,   "unit": "mult"},
        "offset":          {"min": -0.005,  "max": 0.005,  "unit": "norm"},
    },
    SplitKind.hidden: {
        "mask_dx":         {"min": -3.0,    "max": 3.0,    "unit": "px"},
        "mask_dy":         {"min": -3.0,    "max": 3.0,    "unit": "px"},
        "mask_theta":      {"min": -1.0,    "max": 1.0,    "unit": "deg"},
        "mask_blur_sigma": {"min": 0.0,     "max": 2.0,    "unit": "px"},
        "clock_offset":    {"min": -0.5,    "max": 0.5,    "unit": "frames"},
        "duty_cycle":      {"min": 0.7,     "max": 1.0,    "unit": "ratio"},
        "gain":            {"min": 0.5,     "max": 1.5,    "unit": "mult"},
        "offset":          {"min": -0.1,    "max": 0.1,    "unit": "norm"},
    },
}

NOISE_CONFIG = {
    SplitKind.probe:  {"noise_alpha": 1500.0, "noise_sigma": 0.01},
    SplitKind.hidden: {"noise_alpha": 500.0,  "noise_sigma": 0.02},
}


# ---------------------------------------------------------------------------
# Forward model helpers
# ---------------------------------------------------------------------------


def warp_mask_3d(
    mask: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Affine-warp each temporal frame of mask (H, W, T).

    Args:
        mask: (H, W, T) coded aperture masks.
        dx: x-translation in pixels.
        dy: y-translation in pixels.
        theta_deg: rotation in degrees.
        blur_sigma: Gaussian blur sigma (simulates mask defocus).

    Returns:
        Warped mask (H, W, T), clipped to [0, 1].
    """
    H, W, T = mask.shape
    out = np.zeros_like(mask)
    cx, cy = W / 2.0, H / 2.0
    th = np.radians(theta_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)

    for t in range(T):
        mat = np.array([
            [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        frame = affine_transform(
            mask[:, :, t], inv[:2, :2], offset=inv[:2, 2], cval=0, order=1
        )
        if blur_sigma > 0:
            frame = gaussian_filter(frame, sigma=blur_sigma)
        out[:, :, t] = np.clip(frame, 0, 1)
    return out.astype(np.float32)


def cacti_forward(
    x_gt: np.ndarray,
    mask: np.ndarray,
    gain: float = 1.0,
    offset: float = 0.0,
    duty_cycle: float = 1.0,
    clock_offset: float = 0.0,
) -> np.ndarray:
    """CACTI forward model: y = gain * sum(x_t * mask_t * duty) + offset.

    Args:
        x_gt: (H, W, T) ground truth video.
        mask: (H, W, T) coded aperture masks (possibly warped).
        gain: radiometric gain multiplier.
        offset: additive offset.
        duty_cycle: effective exposure fraction per frame.
        clock_offset: temporal misalignment (fractional frames, not used in
            simple model but affects mask selection in full model).

    Returns:
        y: (H, W) compressed measurement.
    """
    H, W, T = x_gt.shape

    # Apply duty cycle: scale each frame's contribution
    y = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        # Clock offset: shift which mask frame is used (linear interpolation)
        t_eff = t + clock_offset
        t_lo = int(np.floor(t_eff)) % T
        t_hi = (t_lo + 1) % T
        frac = t_eff - np.floor(t_eff)

        mask_t = (1.0 - frac) * mask[:, :, t_lo] + frac * mask[:, :, t_hi]
        y += x_gt[:, :, t] * mask_t * duty_cycle

    y = gain * y + offset
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def _load_cacti_mat(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load CACTI benchmark .mat file.

    Returns (orig, mask) where:
      - orig: (H, W, T) ground truth video, float32 [0, 1]
      - mask: (H, W, T) coded aperture mask, float32 [0, 1]
    """
    try:
        data = loadmat(str(path))
        orig = None
        mask = None

        for key in ["orig", "Orig", "gt", "GT"]:
            if key in data:
                orig = data[key].astype(np.float32)
                break

        for key in ["mask", "Mask"]:
            if key in data:
                mask = data[key].astype(np.float32)
                break

        if orig is None or mask is None:
            logger.warning("Missing orig/mask in %s", path)
            return None

        # Normalize to [0, 1]
        if float(orig.max()) > 1.5:
            orig = orig / 255.0
        orig = np.clip(orig, 0, 1)

        if float(mask.max()) > 1.5:
            mask = mask / 255.0
        mask = np.clip(mask, 0, 1)

        # Ensure mask has same T as orig
        if mask.ndim == 3 and orig.ndim == 3:
            T = orig.shape[2]
            H, W, M = mask.shape
            if M != T:
                mask_exp = np.empty((H, W, T), dtype=np.float32)
                for t in range(T):
                    mask_exp[:, :, t] = mask[:, :, t % M]
                mask = mask_exp

        return orig, mask
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CactiCounterfactualGenerator(BaseCounterfactualGenerator):
    """CACTI counterfactual pack generator.

    6 benchmark video clips, 8-param mismatch model,
    31 scenarios per clip per split.
    Total: 372 scenarios (186 probe + 186 hidden).
    """

    def __init__(
        self,
        seed_public: int = 2026_02_18,
        seed_hidden: int = 9999_02_18,
    ):
        super().__init__(
            modality="cacti",
            pack_id="cacti_cfpack_v1",
            seed_public=seed_public,
            seed_hidden=seed_hidden,
        )

    def load_scenes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load CACTI benchmark videos and masks."""
        scenes = []
        mat_files = sorted(BENCHMARK_DIR.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found in {BENCHMARK_DIR}"
            )

        for mat_path in mat_files:
            result = _load_cacti_mat(mat_path)
            if result is not None:
                orig, mask = result
                clip_id = mat_path.stem
                scenes.append((clip_id, orig, mask))

        if not scenes:
            raise FileNotFoundError(
                f"No valid CACTI scenes in {BENCHMARK_DIR}"
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
        """Apply CACTI forward model with mismatch and noise."""
        dx = mismatch_params.get("mask_dx", 0.0)
        dy = mismatch_params.get("mask_dy", 0.0)
        theta = mismatch_params.get("mask_theta", 0.0)
        blur = mismatch_params.get("mask_blur_sigma", 0.0)
        clock_off = mismatch_params.get("clock_offset", 0.0)
        duty = mismatch_params.get("duty_cycle", 1.0)
        gain = mismatch_params.get("gain", 1.0)
        offset = mismatch_params.get("offset", 0.0)

        # Warp mask
        has_warp = abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(theta) > 1e-6 or blur > 0
        if has_warp:
            mask_warped = warp_mask_3d(mask, dx, dy, theta, blur)
        else:
            mask_warped = mask.copy()

        # Binarize for forward model
        mask_bin = (mask_warped > 0.5).astype(np.float32)

        # Forward model
        y_clean = cacti_forward(
            x_gt, mask_bin,
            gain=gain, offset=offset,
            duty_cycle=duty, clock_offset=clock_off,
        )

        # Add noise
        noise_alpha = noise_config.get("noise_alpha", 1500.0)
        noise_sigma = noise_config.get("noise_sigma", 0.01)
        y_noisy = add_poisson_gaussian_noise(
            y_clean, peak=noise_alpha, sigma=noise_sigma, rng=rng
        )

        return y_noisy, mask_warped

    def get_param_config(self, split: SplitKind) -> Dict[str, Dict[str, float]]:
        return PARAM_CONFIG[split]

    def get_red_team_configs(self, split: SplitKind) -> Dict[str, Dict[str, Any]]:
        return get_red_team_configs("cacti")

    def get_nominal_noise(self, split: SplitKind) -> Dict[str, float]:
        return NOISE_CONFIG[split]
