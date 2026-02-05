"""pwm_core.recon.diffusion

Optional diffusion-based posterior adapters (stub).

In real usage, you'd integrate:
- deepinv diffusion priors
- score-based models / posterior sampling
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def diffusion_stub(y: np.ndarray, physics: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Placeholder: just return y for now.
    return y.astype(np.float32), {"note": "diffusion solver not implemented in starter stub"}
