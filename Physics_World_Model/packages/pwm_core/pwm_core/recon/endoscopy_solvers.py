"""Endoscopy deep learning reconstruction solvers.

Delegates to existing reconstruction infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.recon.care_unet import run_care


def endomapper_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run endomapper_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "endomapper_recon"
    return result, info


def af_sfm_learner_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run af_sfm_learner_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "af_sfm_learner_recon"
    return result, info
