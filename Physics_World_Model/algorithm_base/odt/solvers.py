"""Solvers for Optical Diffraction Tomography (ODT) (odt).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "odt"
DISPLAY_NAME = "Optical Diffraction Tomography (ODT)"


# Solver registry for odt
SOLVERS = {
    "traditional_cpu": {
        "name": "Adjoint [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "best_quality": {
        "name": "PnP-ADMM [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "odt_dl": {
        "name": "ODT-Net (PhaseNet) [proxy]",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'odt_dl']
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})
    # For multi-angle complex ODT input (n_angles, H, W, 2), collapse to 2D
    y_2d = y.astype(np.float32)
    if y_2d.ndim == 4 and y_2d.shape[-1] == 2:
        # Take magnitude average over angles
        mag = np.sqrt(y_2d[..., 0]**2 + y_2d[..., 1]**2)  # (n_angles, H, W)
        y_2d = mag.mean(axis=0)  # (H, W)
    elif y_2d.ndim == 3:
        y_2d = y_2d.mean(axis=0)
    fn = _load_fn(solver_key)
    result = fn(y_2d, operator, cfg)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for odt."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Adjoint [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PnP-ADMM [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_odt_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ODT-Net (PhaseNet) [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("odt_dl", y, operator, cfg)
