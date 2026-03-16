"""Solvers for Susceptibility-Weighted Imaging (SWI) (swi).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "swi"
DISPLAY_NAME = "Susceptibility-Weighted Imaging (SWI)"


# Solver registry for swi
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "best_quality": {
        "name": "DL-Recon [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "swi_dl": {
        "name": "SWI-Net [proxy]",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_zero_filled",
        "gpu": False,
        "reference": "",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'swi_dl']
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
    """List all available solvers for swi."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DL-Recon [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_swi_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SWI-Net [proxy]. CPU only.
    """
    return run_solver("swi_dl", y, operator, cfg)
