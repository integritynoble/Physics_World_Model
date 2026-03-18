"""Solvers for PALM/STORM Single-Molecule Localization (palm_storm).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "palm_storm"
DISPLAY_NAME = "PALM/STORM Single-Molecule Localization"


# Solver registry for palm_storm
SOLVERS = {
    "traditional_cpu": {
        "name": "Richardson-Lucy (STORM/PALM)",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "DECODE-SMLM",
        "module": "pwm_core.recon.smlm_solvers",
        "function": "decode_smlm_recon",
        "gpu": True,
        "reference": "Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090",
    },
    "famous_dl": {
        "name": "DeepSTORM",
        "module": "pwm_core.recon.smlm_solvers",
        "function": "deep_storm_recon",
        "gpu": True,
        "reference": "Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4)",
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
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for palm_storm."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy (STORM/PALM). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DECODE-SMLM. GPU required.
    Reference: Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DeepSTORM. GPU required.
    Reference: Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4)
    """
    return run_solver("famous_dl", y, operator, cfg)
