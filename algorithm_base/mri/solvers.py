"""Solvers for Magnetic Resonance Imaging (MRI) (mri).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "mri"
DISPLAY_NAME = "Magnetic Resonance Imaging (MRI)"


# Solver registry for mri
SOLVERS = {
    "traditional_cpu": {
        "name": "Zero-Filled IFFT",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_zero_filled",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "CS-MRI (Wavelet)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_cs_mri",
        "gpu": False,
        "reference": "Lustig et al. 2007, MRM",
    },
    "famous_dl": {
        "name": "MoDL",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": False,
        "reference": "Aggarwal et al. 2019, IEEE TMI",
    },
    "small_gpu": {
        "name": "MoDL (5 unrolls)",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": False,
        "reference": "",
    },
    "sense": {
        "name": "SENSE",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_sense",
        "gpu": False,
        "reference": "Pruessmann et al., MRM 1999",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu', 'sense']
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
    """List all available solvers for mri."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Zero-Filled IFFT. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CS-MRI (Wavelet). CPU only.
    Reference: Lustig et al. 2007, MRM
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MoDL. CPU only.
    Reference: Aggarwal et al. 2019, IEEE TMI
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MoDL (5 unrolls). CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_sense(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SENSE. CPU only.
    Reference: Pruessmann et al., MRM 1999
    """
    return run_solver("sense", y, operator, cfg)
