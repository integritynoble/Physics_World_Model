"""pwm_core.analysis.residual_tests

Residual structure tests:
- whiteness proxy
- frequency-domain structure
- simple autocorrelation metrics

Used for diagnosing mismatch and for operator-fit scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ResidualDiagnostics:
    rms: float
    autocorr1: float
    hf_energy_ratio: float


def residual_diagnostics(r: np.ndarray) -> ResidualDiagnostics:
    rr = r.reshape(-1).astype(np.float32)
    rms = float(np.sqrt(np.mean(rr * rr)))
    if rr.size < 4:
        return ResidualDiagnostics(rms=rms, autocorr1=0.0, hf_energy_ratio=0.0)

    # lag-1 autocorrelation
    a = rr[:-1]
    b = rr[1:]
    denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)) + 1e-12)
    autoc1 = float(np.mean(a * b) / denom)

    # high-frequency energy ratio in FFT (1D proxy)
    f = np.fft.rfft(rr)
    mag2 = (np.abs(f) ** 2).astype(np.float64)
    n = mag2.size
    hf = float(np.sum(mag2[int(0.7 * n) :]) / (np.sum(mag2) + 1e-12))

    return ResidualDiagnostics(rms=rms, autocorr1=autoc1, hf_energy_ratio=hf)


def as_dict(d: ResidualDiagnostics) -> Dict[str, Any]:
    return {"rms": d.rms, "autocorr1": d.autocorr1, "hf_energy_ratio": d.hf_energy_ratio}
