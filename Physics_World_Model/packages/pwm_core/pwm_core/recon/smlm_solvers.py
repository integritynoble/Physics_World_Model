"""Single-Molecule Localization Microscopy (SMLM) deep learning solvers.

Delegates to existing reconstruction infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.recon.care_unet import run_care


def decode_smlm_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run decode_smlm_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "decode_smlm_recon"
    return result, info


def deep_storm_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run deep_storm_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "deep_storm_recon"
    return result, info
