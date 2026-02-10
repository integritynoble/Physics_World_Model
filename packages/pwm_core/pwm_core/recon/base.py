"""pwm_core.recon.base

Reconstruction solver interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from pwm_core.recon.protocols import LinearLikeOperator


@dataclass
class ReconResult:
    x_hat_path: str
    metrics: Dict[str, float]
    solver_id: str
    solver_params: Dict[str, Any]


class ReconSolver(Protocol):
    solver_id: str
    def run(self, y: np.ndarray, physics: LinearLikeOperator, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]: ...
