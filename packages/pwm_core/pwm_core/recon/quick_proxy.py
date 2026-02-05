"""pwm_core.recon.quick_proxy

Fast reconstruction quality proxy used during theta scoring / portfolio selection.

This is NOT PSNR/SSIM (needs ground truth).
Instead use simple heuristics:
- smoothness / TV proxy
- range sanity
- energy/contrast sanity

Lower is better by convention.
"""

from __future__ import annotations

import numpy as np


def tv_proxy(x: np.ndarray) -> float:
    if x.ndim < 2:
        v = x.reshape(-1)
        return float(np.mean(np.abs(np.diff(v))))
    # assume (H,W) or (H,W,...) -- use first two dims
    dx = np.abs(np.diff(x, axis=1)).mean()
    dy = np.abs(np.diff(x, axis=0)).mean()
    return float(dx + dy)


def quick_score_proxy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    score = tv_proxy(x)
    # penalize NaNs / infs
    if not np.isfinite(x).all():
        score += 1e6
    return float(score)
