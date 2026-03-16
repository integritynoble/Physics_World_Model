"""Solvers for Expansion Microscopy (ExM) (expansion).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "expansion"
DISPLAY_NAME = "Expansion Microscopy (ExM)"


# Solver registry for expansion
SOLVERS = {
    "traditional_cpu": {
        "name": "Richardson-Lucy",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "CARE",
        "module": "pwm_core.recon.care_unet",
        "function": "run_care",
        "gpu": True,
        "reference": "Weigert et al. 2018",
    },
    "exm_dl": {
        "name": "EXpansionNet",
        "module": "pwm_core.recon.expansion_solvers",
        "function": "expansion_dl_recon",
        "gpu": True,
        "reference": "Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'exm_dl']
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
    """List all available solvers for expansion."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CARE. GPU required.
    Reference: Weigert et al. 2018
    """
    return run_solver("best_quality", y, operator, cfg)

def run_exm_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EXpansionNet. GPU required.
    Reference: Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090
    """
    return run_solver("exm_dl", y, operator, cfg)
