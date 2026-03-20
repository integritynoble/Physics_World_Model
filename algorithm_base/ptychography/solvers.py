"""Solvers for Ptychographic Imaging (ptychography).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "ptychography"
DISPLAY_NAME = "Ptychographic Imaging"


# Solver registry for ptychography
SOLVERS = {
    "traditional_cpu": {
        "name": "ePIE",
        "module": "pwm_core.recon.ptychography_solver",
        "function": "run_epie",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "PtychoNN",
        "module": "pwm_core.recon.ptychonn",
        "function": "run_ptychonn",
        "gpu": False,
        "reference": "Cherukara et al. 2020",
    },
    "famous_dl": {
        "name": "PtychoNN",
        "module": "pwm_core.recon.ptychonn",
        "function": "run_ptychonn",
        "gpu": False,
        "reference": "",
    },
    "small_gpu": {
        "name": "PtychoNN 2.0",
        "module": "pwm_core.recon.ptychonn",
        "function": "run_ptychonn",
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
        result = np.asarray(result[0], dtype=np.float32)
    else:
        result = np.asarray(result, dtype=np.float32)
    # Resize small outputs to 256x256
    if result.ndim == 2 and result.shape != (256, 256):
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for ptychography."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ePIE. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PtychoNN. CPU only.
    Reference: Cherukara et al. 2020
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PtychoNN. CPU only.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PtychoNN 2.0. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
