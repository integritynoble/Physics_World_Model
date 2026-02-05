"""pwm_core.analysis.metrics

Metrics for recon quality when ground truth is available.
Also provides no-reference proxies for measured-data mode.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(x, y)
    if m <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))


def no_reference_energy(x: np.ndarray) -> float:
    """Cheap proxy: energy + TV proxy (lower often smoother)."""
    v = x.reshape(-1).astype(np.float32)
    return float(np.mean(v * v))
