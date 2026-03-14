"""STED (Stimulated Emission Depletion) microscopy deep learning solvers.

Delegates to existing reconstruction infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.recon.care_unet import run_care


def sted_care_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run sted_care_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "sted_care_recon"
    return result, info


def rcan_sted_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run rcan_sted_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "rcan_sted_recon"
    return result, info
