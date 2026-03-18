"""Solvers for Lensless (Diffuser Camera) Imaging (lensless).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "lensless"
DISPLAY_NAME = "Lensless (Diffuser Camera) Imaging"


# Solver registry for lensless
SOLVERS = {
    "traditional_cpu": {
        "name": "ADMM-TV",
        "module": "pwm_core.recon.lensless_solver",
        "function": "run_lensless",
        "gpu": False,
        "reference": "Antipa et al. 2018",
    },
    "best_quality": {
        "name": "FlatNet",
        "module": "pwm_core.recon.flatnet",
        "function": "run_flatnet",
        "gpu": False,
        "reference": "Khan et al. TPAMI 2020",
    },
    "famous_dl": {
        "name": "FlatNet",
        "module": "pwm_core.recon.flatnet",
        "function": "run_flatnet",
        "gpu": False,
        "reference": "",
    },
    "small_gpu": {
        "name": "FlatNet-Lite",
        "module": "pwm_core.recon.flatnet",
        "function": "run_flatnet",
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
    """List all available solvers for lensless."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ADMM-TV. CPU only.
    Reference: Antipa et al. 2018
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FlatNet. CPU only.
    Reference: Khan et al. TPAMI 2020
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FlatNet. CPU only.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FlatNet-Lite. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
