"""pwm_core.core.policies

Bounded auto-refine + safety budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BoundedSearchPolicy:
    max_candidates: int = 10
    max_refine_steps: int = 8
    max_seconds: float = 60.0
    stop_on_score: Optional[float] = None


@dataclass
class SafeRanges:
    max_photons_min: float = 1.0
    max_photons_max: float = 1e7
    sampling_rate_min: float = 0.01
    sampling_rate_max: float = 1.0
    drift_px_max: float = 10.0
    psf_sigma_px_min: float = 0.5
    psf_sigma_px_max: float = 8.0


DEFAULT_SEARCH_POLICY = BoundedSearchPolicy()
DEFAULT_SAFE_RANGES = SafeRanges()


def bounded_sweep_grid() -> Dict[str, List[float]]:
    return {
        "states.budget.photon_budget.max_photons": [200.0, 800.0, 2500.0],
        "states.budget.measurement_budget.sampling_rate": [0.1, 0.25, 0.5],
        "states.sample.motion.drift_amp_px": [0.0, 0.5, 1.5],
        "states.calibration.psf.sigma_px": [1.0, 1.5, 2.5],
    }
