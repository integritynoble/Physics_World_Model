"""Solvers for Fluorescence Lifetime Imaging (FLIM) (flim).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "flim"
DISPLAY_NAME = "Fluorescence Lifetime Imaging (FLIM)"


# Solver registry for flim
SOLVERS = {
    "traditional_cpu": {
        "name": "Phasor Analysis",
        "module": "pwm_core.recon.flim_solver",
        "function": "run_flim",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "MLE Fit",
        "module": "pwm_core.recon.flim_solver",
        "function": "run_flim",
        "gpu": False,
        "reference": "Becker 2012, J. Microscopy",
    },
    "famous_dl": {
        "name": "MLE Fit (iterative)",
        "module": "pwm_core.recon.flim_solver",
        "function": "run_flim",
        "gpu": False,
        "reference": "Becker 2012, J. Microscopy",
    },
    "small_gpu": {
        "name": "Phasor Analysis",
        "module": "pwm_core.recon.flim_solver",
        "function": "run_flim",
        "gpu": False,
        "reference": "Digman et al. 2008, Biophysical Journal",
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
        result = np.asarray(result[0], dtype=np.float32)
    else:
        result = np.asarray(result, dtype=np.float32)
    # run_flim returns (H, W, 2) stacking [lifetime, amplitude]. Keep amplitude channel.
    if result.ndim == 3 and result.shape[2] <= 4:
        result = result[..., 1] if result.shape[2] == 2 else result[..., 0]
    return result


def list_solvers():
    """List all available solvers for flim."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Phasor Analysis. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MLE Fit. CPU only.
    Reference: Becker 2012, J. Microscopy
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MLE Fit (iterative). CPU only.
    Reference: Becker 2012, J. Microscopy
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Phasor Analysis. CPU only.
    Reference: Digman et al. 2008, Biophysical Journal
    """
    return run_solver("small_gpu", y, operator, cfg)
