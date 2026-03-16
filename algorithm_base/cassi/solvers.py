"""Solvers for Coded Aperture Snapshot Spectral Imaging (CASSI) (cassi).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cassi"
DISPLAY_NAME = "Coded Aperture Snapshot Spectral Imaging (CASSI)"


# Solver registry for cassi
SOLVERS = {
    "traditional_cpu": {
        "name": "GAP-TV",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016",
    },
    "best_quality": {
        "name": "GAP-TV (guided)",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016",
    },
    "famous_dl": {
        "name": "GAP-TV (fast)",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "",
    },
    "small_gpu": {
        "name": "GAP-TV (small)",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "",
    },
    "mst_l": {
        "name": "MST-L",
        "module": "pwm_core.recon.mst",
        "function": "mst_recon_cassi",
        "gpu": False,
        "reference": "Cai et al., CVPR 2022",
    },
    "hdnet": {
        "name": "HDNet",
        "module": "pwm_core.recon.hdnet",
        "function": "run_hdnet",
        "gpu": False,
        "reference": "Hu et al., CVPR 2022",
    },
    "hsi_sdecnn": {
        "name": "HSI-SDeCNN",
        "module": "pwm_core.recon.hsi_sdecnn",
        "function": "run_hsi_sdecnn",
        "gpu": False,
        "reference": "Maffei et al., TGRS 2020",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu', 'mst_l', 'hdnet', 'hsi_sdecnn']
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
    """List all available solvers for cassi."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV. CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (guided). CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (fast). CPU only.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (small). CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_mst_l(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MST-L. CPU only.
    Reference: Cai et al., CVPR 2022
    """
    return run_solver("mst_l", y, operator, cfg)

def run_hdnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HDNet. CPU only.
    Reference: Hu et al., CVPR 2022
    """
    return run_solver("hdnet", y, operator, cfg)

def run_hsi_sdecnn(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HSI-SDeCNN. CPU only.
    Reference: Maffei et al., TGRS 2020
    """
    return run_solver("hsi_sdecnn", y, operator, cfg)
