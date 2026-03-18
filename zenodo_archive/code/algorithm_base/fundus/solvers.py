"""Solvers for Fundus Camera (fundus).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "fundus"
DISPLAY_NAME = "Fundus Camera"


# Solver registry for fundus
SOLVERS = {
    "traditional_cpu": {
        "name": "Richardson-Lucy",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "RETFound",
        "module": "pwm_core.recon.fundus_solvers",
        "function": "retfound_recon",
        "gpu": True,
        "reference": "Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156",
    },
    "famous_dl": {
        "name": "DR-Grade-Net",
        "module": "pwm_core.recon.fundus_solvers",
        "function": "dr_grade_net_recon",
        "gpu": True,
        "reference": "Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22)",
    },
}


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl']
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for fundus."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RETFound. GPU required.
    Reference: Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DR-Grade-Net. GPU required.
    Reference: Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22)
    """
    return run_solver("famous_dl", y, operator, cfg)
