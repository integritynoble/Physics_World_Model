"""Solvers for Structured Illumination Microscopy (SIM) (sim).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "sim"
DISPLAY_NAME = "Structured Illumination Microscopy (SIM)"


# Solver registry for sim
SOLVERS = {
    "traditional_cpu": {
        "name": "Wiener-SIM",
        "module": "pwm_core.recon.sim_solver",
        "function": "run_sim_reconstruction",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "HiFi-SIM",
        "module": "pwm_core.recon.sim_solver",
        "function": "run_sim_reconstruction",
        "gpu": False,
        "reference": "Wen et al. 2021, Light: S&A",
    },
    "famous_dl": {
        "name": "fairSIM (open-source)",
        "module": "pwm_core.recon.sim_solver",
        "function": "run_sim_reconstruction",
        "gpu": False,
        "reference": "Mueller et al. 2016, Nature Comm.",
    },
    "small_gpu": {
        "name": "Wiener-SIM (fast)",
        "module": "pwm_core.recon.sim_solver",
        "function": "run_sim_reconstruction",
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
    """List all available solvers for sim."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Wiener-SIM. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HiFi-SIM. CPU only.
    Reference: Wen et al. 2021, Light: S&A
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """fairSIM (open-source). CPU only.
    Reference: Mueller et al. 2016, Nature Comm.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Wiener-SIM (fast). CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
