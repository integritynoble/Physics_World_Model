"""Solvers for 4D-STEM Electron Diffraction (electron_diffraction).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "electron_diffraction"
DISPLAY_NAME = "4D-STEM Electron Diffraction"


# Solver registry for electron_diffraction
SOLVERS = {
    "traditional_cpu": {
        "name": "ePIE (electron ptychography)",
        "module": "pwm_core.recon.ptychography_solver",
        "function": "run_epie",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "ED-Net [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "famous_dl": {
        "name": "CRISP-ED [proxy]",
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
        result = np.asarray(result[0], dtype=np.float32)
    else:
        result = np.asarray(result, dtype=np.float32)
    # run_epie returns a 2D complex object (2x input size). Resize to 256x256.
    if result.ndim == 2 and result.shape != (256, 256):
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for electron_diffraction."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ePIE (electron ptychography). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ED-Net [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CRISP-ED [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("famous_dl", y, operator, cfg)
