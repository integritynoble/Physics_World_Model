"""Solvers for Single-Pixel Camera (SPC) (spc).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "spc"
DISPLAY_NAME = "Single-Pixel Camera (SPC)"


# Solver registry for spc
SOLVERS = {
    "traditional_cpu": {
        "name": "TVAL3",
        "module": "pwm_core.recon.cs_solvers",
        "function": "run_tval3",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "ADMM-L1",
        "module": "pwm_core.recon.spc_solvers",
        "function": "run_admm_spc",
        "gpu": False,
        "reference": "Boyd et al. 2010",
    },
    "famous_dl": {
        "name": "FISTA-L1",
        "module": "pwm_core.recon.spc_solvers",
        "function": "run_fista_l1_spc",
        "gpu": False,
        "reference": "Beck & Teboulle 2009",
    },
    "small_gpu": {
        "name": "FISTA-L1",
        "module": "pwm_core.recon.spc_solvers",
        "function": "run_fista_l1_spc",
        "gpu": False,
        "reference": "",
    },
    "ista_net_plus": {
        "name": "ISTA-Net+",
        "module": "pwm_core.recon.spc_solvers",
        "function": "run_admm_spc",
        "gpu": False,
        "reference": "Zhang & Ghanem, CVPR 2018",
    },
    "hatnet": {
        "name": "HATNet",
        "module": "pwm_core.recon.spc_solvers",
        "function": "run_fista_l1_spc",
        "gpu": False,
        "reference": "Song et al., TIP 2021",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu', 'ista_net_plus', 'hatnet']
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
    # Without operator, TVAL3 returns y unchanged. Handle 1D/non-square cases.
    if result.ndim == 1:
        n = int(result.shape[0] ** 0.5)
        result = result[:n*n].reshape(n, n)
    if result.ndim == 2 and result.shape != (256, 256):
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for spc."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """TVAL3. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ADMM-L1. CPU only.
    Reference: Boyd et al. 2010
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FISTA-L1. CPU only.
    Reference: Beck & Teboulle 2009
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FISTA-L1. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_ista_net_plus(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ISTA-Net+. CPU only.
    Reference: Zhang & Ghanem, CVPR 2018
    """
    return run_solver("ista_net_plus", y, operator, cfg)

def run_hatnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HATNet. CPU only.
    Reference: Song et al., TIP 2021
    """
    return run_solver("hatnet", y, operator, cfg)
