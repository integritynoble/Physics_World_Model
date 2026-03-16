"""Solvers for Panorama Multi-Focus Fusion (panorama).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "panorama"
DISPLAY_NAME = "Panorama Multi-Focus Fusion"


# Solver registry for panorama
SOLVERS = {
    "traditional_cpu": {
        "name": "Laplacian Pyramid Fusion",
        "module": "pwm_core.recon.panorama_solver",
        "function": "run_panorama_fusion",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "Guided Filter Fusion",
        "module": "pwm_core.recon.panorama_solver",
        "function": "run_panorama_fusion",
        "gpu": False,
        "reference": "",
    },
    "famous_dl": {
        "name": "IFCNN",
        "module": "pwm_core.recon.ifcnn",
        "function": "run_ifcnn",
        "gpu": False,
        "reference": "Zhang et al. 2020",
    },
    "small_gpu": {
        "name": "IFCNN",
        "module": "pwm_core.recon.ifcnn",
        "function": "run_ifcnn",
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
    """List all available solvers for panorama."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Laplacian Pyramid Fusion. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Guided Filter Fusion. CPU only.
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """IFCNN. CPU only.
    Reference: Zhang et al. 2020
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """IFCNN. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
