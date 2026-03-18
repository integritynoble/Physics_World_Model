"""Solvers for Fourier Ptychographic Microscopy (FPM) (fpm).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "fpm"
DISPLAY_NAME = "Fourier Ptychographic Microscopy (FPM)"


# Solver registry for fpm
SOLVERS = {
    "traditional_cpu": {
        "name": "Sequential Phase Retrieval",
        "module": "pwm_core.recon.fpm_solver",
        "function": "run_fpm",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "Gradient Descent FPM [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "famous_dl": {
        "name": "Fourier Ptychnet",
        "module": "pwm_core.recon.fpm_solver",
        "function": "run_fpm",
        "gpu": False,
        "reference": "Jiang et al. 2018, Biomed. Optics Express",
    },
    "small_gpu": {
        "name": "Fourier Ptychnet",
        "module": "pwm_core.recon.fpm_solver",
        "function": "run_fpm",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu']
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
    """List all available solvers for fpm."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Sequential Phase Retrieval. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Gradient Descent FPM [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Fourier Ptychnet. CPU only.
    Reference: Jiang et al. 2018, Biomed. Optics Express
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Fourier Ptychnet. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
