"""Solvers for Digital Holographic Microscopy (holography).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "holography"
DISPLAY_NAME = "Digital Holographic Microscopy"


# Solver registry for holography
SOLVERS = {
    "traditional_cpu": {
        "name": "Angular Spectrum",
        "module": "pwm_core.recon.holography_solver",
        "function": "run_holography_reconstruction",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "PhaseNet",
        "module": "pwm_core.recon.phasenet",
        "function": "run_phasenet",
        "gpu": False,
        "reference": "Rivenson et al. 2018, Light: S&A",
    },
    "famous_dl": {
        "name": "PhaseNet",
        "module": "pwm_core.recon.phasenet",
        "function": "run_phasenet",
        "gpu": False,
        "reference": "",
    },
    "small_gpu": {
        "name": "PhaseNet",
        "module": "pwm_core.recon.phasenet",
        "function": "run_phasenet",
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
    # run_holography_reconstruction can return (H, W, 2) [amplitude, phase].
    # Take amplitude (index 0) to produce the standard 2D output.
    if result.ndim == 3 and result.shape[2] <= 4:
        result = result[..., 0]
    return result


def list_solvers():
    """List all available solvers for holography."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Angular Spectrum. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PhaseNet. CPU only.
    Reference: Rivenson et al. 2018, Light: S&A
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PhaseNet. CPU only.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PhaseNet. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
