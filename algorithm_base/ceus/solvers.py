"""Solvers for Contrast-Enhanced Ultrasound (CEUS) (ceus).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "ceus"
DISPLAY_NAME = "Contrast-Enhanced Ultrasound (CEUS)"


# Solver registry for ceus
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
    "us_dl_enhance": {
        "name": "US-DeepSight [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'us_dl_enhance']
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
    """List all available solvers for ceus."""
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

def run_us_dl_enhance(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """US-DeepSight [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("us_dl_enhance", y, operator, cfg)
