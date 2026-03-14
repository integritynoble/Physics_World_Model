"""Dark-field microscopy deep learning solvers.

Delegates to existing reconstruction infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.recon.care_unet import run_care


def df_unet_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run df_unet_recon reconstruction (delegates to CARE U-Net)."""
    result, info = run_care(y, physics, cfg)
    info["solver"] = "df_unet_recon"
    return result, info
