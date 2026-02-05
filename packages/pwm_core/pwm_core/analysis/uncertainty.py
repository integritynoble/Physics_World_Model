"""pwm_core.analysis.uncertainty

Uncertainty estimation utilities:
- ensembles (multiple recon runs with different seeds / solvers)
- bootstrap over measurements (if available)

Starter implementation: ensemble variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class UncertaintyResult:
    mean: np.ndarray
    var: np.ndarray
    meta: Dict[str, Any]


def ensemble_mean_var(xs: List[np.ndarray]) -> UncertaintyResult:
    stack = np.stack([np.asarray(x, dtype=np.float32) for x in xs], axis=0)
    mean = stack.mean(axis=0)
    var = stack.var(axis=0)
    return UncertaintyResult(mean=mean, var=var, meta={"n": int(stack.shape[0])})
