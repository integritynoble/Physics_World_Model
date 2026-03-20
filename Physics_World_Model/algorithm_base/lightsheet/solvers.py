"""Solvers for Light-Sheet Fluorescence Microscopy (LSFM) (lightsheet).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "lightsheet"
DISPLAY_NAME = "Light-Sheet Fluorescence Microscopy (LSFM)"


# Solver registry for lightsheet
SOLVERS = {
    "traditional_cpu": {
        "name": "Fourier Notch Filter",
        "module": "pwm_core.recon.lightsheet_solver",
        "function": "run_lightsheet",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "VSNR",
        "module": "pwm_core.recon.lightsheet_solver",
        "function": "run_lightsheet",
        "gpu": False,
        "reference": "",
    },
    "famous_dl": {
        "name": "DeStripe",
        "module": "pwm_core.recon.destripe_net",
        "function": "run_destripe",
        "gpu": False,
        "reference": "Liang et al. 2022",
    },
    "small_gpu": {
        "name": "DeStripe",
        "module": "pwm_core.recon.destripe_net",
        "function": "run_destripe",
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
    """List all available solvers for lightsheet."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Fourier Notch Filter. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """VSNR. CPU only.
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DeStripe. CPU only.
    Reference: Liang et al. 2022
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DeStripe. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
