"""Solvers for Neural Radiance Fields (NeRF) (nerf).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "nerf"
DISPLAY_NAME = "Neural Radiance Fields (NeRF)"


# Solver registry for nerf
SOLVERS = {
    "traditional_cpu": {
        "name": "SfM + MVS",
        "module": "pwm_core.recon.nerf_solver",
        "function": "run_nerf",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "Mip-NeRF 360",
        "module": "pwm_core.recon.nerf_solver",
        "function": "run_nerf",
        "gpu": True,
        "reference": "Barron et al. CVPR 2022",
    },
    "famous_dl": {
        "name": "NeRF (original MLP)",
        "module": "pwm_core.recon.nerf_solver",
        "function": "run_nerf",
        "gpu": False,
        "reference": "Mildenhall et al. 2020",
    },
    "small_gpu": {
        "name": "Instant-NGP",
        "module": "pwm_core.recon.nerf_solver",
        "function": "run_nerf",
        "gpu": False,
        "reference": "Muller et al. 2022",
    },
    "rl_proxy": {
        "name": "Richardson-Lucy (proxy baseline)",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "fista_proxy": {
        "name": "FISTA-TV (proxy baseline)",
        "module": "pwm_core.recon.cs_solvers",
        "function": "run_ista",
        "gpu": False,
        "reference": "Beck & Teboulle 2009, SIAM",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu', 'rl_proxy', 'fista_proxy']
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
    # Convert RGB to grayscale if needed
    if result.ndim == 3 and result.shape[-1] == 3:
        result = (0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for nerf."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SfM + MVS. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Mip-NeRF 360. GPU required.
    Reference: Barron et al. CVPR 2022
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """NeRF (original MLP). CPU only.
    Reference: Mildenhall et al. 2020
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Instant-NGP. CPU only.
    Reference: Muller et al. 2022
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_rl_proxy(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy (proxy baseline). CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("rl_proxy", y, operator, cfg)

def run_fista_proxy(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FISTA-TV (proxy baseline). CPU only.
    Reference: Beck & Teboulle 2009, SIAM
    """
    return run_solver("fista_proxy", y, operator, cfg)
