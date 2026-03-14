"""Image Scanning Microscopy (ISM) deep learning solvers.

Delegates to existing reconstruction infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.recon.care_unet import run_care


def ism_dl_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run ism_dl_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "ism_dl_recon"
    return result, info
