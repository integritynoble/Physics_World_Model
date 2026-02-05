"""pwm_core.recon.portfolio

Solver portfolio + auto-tuning.

Strategy:
- pick solver candidates based on PhysicsState + TaskState
- run quick proxies if needed
- return best result (plus optionally all results)

Supports:
- Matrix-based operators (lsq, fista)
- Operator-based using forward/adjoint (cg, gd, adjoint)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.recon.classical import (
    least_squares,
    fista_l2,
    gradient_descent_operator,
    conjugate_gradient_operator,
)
from pwm_core.recon.quick_proxy import quick_score_proxy
from pwm_core.recon.pnp import run_pnp

# Import specialized solvers
try:
    from pwm_core.recon.richardson_lucy import run_richardson_lucy
except ImportError:
    run_richardson_lucy = None

try:
    from pwm_core.recon.ct_solvers import run_fbp, run_sart
except ImportError:
    run_fbp = run_sart = None

try:
    from pwm_core.recon.cs_solvers import run_tval3, run_ista
except ImportError:
    run_tval3 = run_ista = None

try:
    from pwm_core.recon.gap_tv import run_gap_tv, run_gap_denoise
except ImportError:
    run_gap_tv = run_gap_denoise = None

try:
    from pwm_core.recon.mri_solvers import run_espirit_recon, run_cs_mri
except ImportError:
    run_espirit_recon = run_cs_mri = None

try:
    from pwm_core.recon.sim_solver import run_sim_reconstruction
except ImportError:
    run_sim_reconstruction = None

try:
    from pwm_core.recon.ptychography_solver import run_epie
except ImportError:
    run_epie = None

try:
    from pwm_core.recon.holography_solver import run_holography_reconstruction
except ImportError:
    run_holography_reconstruction = None


@dataclass
class PortfolioConfig:
    candidates: List[str]
    max_candidates: int = 3


def _get_operator_x_shape(physics: Any) -> Optional[Tuple[int, ...]]:
    """Try to determine x_shape from operator info or attributes."""
    # Try x_shape attribute (WidefieldOperator has this)
    if hasattr(physics, 'x_shape'):
        return tuple(physics.x_shape)

    # Try info() method
    if hasattr(physics, 'info'):
        info = physics.info()
        if 'x_shape' in info:
            return tuple(info['x_shape'])

    # Try theta for CASSI-style operators
    if hasattr(physics, 'theta'):
        theta = physics.theta if isinstance(physics.theta, dict) else {}
        if 'L' in theta and hasattr(physics, 'mask') and physics.mask is not None:
            H, W = physics.mask.shape
            L = int(theta['L'])
            return (H, W, L)

    return None


def _run_operator_solver(
    y: np.ndarray,
    physics: Any,
    solver_id: str,
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run a solver using forward/adjoint operators."""
    if not (hasattr(physics, 'forward') and hasattr(physics, 'adjoint')):
        return None, {"error": "operator has no forward/adjoint"}

    x_shape = _get_operator_x_shape(physics)
    if x_shape is None:
        # Fall back to y shape (works for self-adjoint operators like widefield)
        x_shape = y.shape

    iters = cfg.get("iters", 50)
    reg = cfg.get("reg", 1e-4)

    try:
        if solver_id in ("cg", "conjugate_gradient"):
            x = conjugate_gradient_operator(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                iters=iters,
                reg=reg,
            )
            return x, {"solver": "cg", "iters": iters, "reg": reg}

        elif solver_id in ("gd", "gradient_descent"):
            step = cfg.get("step", 0.01)
            x = gradient_descent_operator(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                iters=iters,
                step=step,
                reg=reg,
            )
            return x, {"solver": "gd", "iters": iters, "step": step, "reg": reg}

        elif solver_id == "adjoint":
            # Simple adjoint (matched filter / backprojection)
            x = physics.adjoint(y)
            if hasattr(x, 'reshape'):
                x = x.reshape(x_shape)
            return x.astype(np.float32), {"solver": "adjoint"}

        elif solver_id in ("tv_fista", "fista"):
            # FISTA with operator (gradient-based)
            lam = cfg.get("lambda", 1e-3)
            x = gradient_descent_operator(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                iters=iters * 2,  # More iters for FISTA-style
                step=0.005,
                reg=lam,
            )
            return x, {"solver": "tv_fista", "iters": iters * 2, "lambda": lam}

        elif solver_id in ("pnp", "pnp_hqs", "pnp_admm", "pnp_fista"):
            # Plug-and-Play with denoiser
            algorithm = solver_id.replace("pnp_", "") if "_" in solver_id else "hqs"
            pnp_cfg = {
                "algorithm": algorithm,
                "denoiser": cfg.get("denoiser", "auto"),
                "iters": cfg.get("iters", 30),
                "sigma": cfg.get("sigma", 0.1),
                "rho": cfg.get("rho", 1.0),
            }
            return run_pnp(y, physics, pnp_cfg)

        # Richardson-Lucy deconvolution
        elif solver_id in ("rl", "rl_deconv", "richardson_lucy"):
            if run_richardson_lucy is not None:
                return run_richardson_lucy(y, physics, cfg)
            return None, {"error": "richardson_lucy not available"}

        # CT solvers
        elif solver_id == "fbp":
            if run_fbp is not None:
                return run_fbp(y, physics, cfg)
            return None, {"error": "fbp not available"}

        elif solver_id == "sart":
            if run_sart is not None:
                return run_sart(y, physics, cfg)
            return None, {"error": "sart not available"}

        # CS solvers
        elif solver_id == "tval3":
            if run_tval3 is not None:
                return run_tval3(y, physics, cfg)
            return None, {"error": "tval3 not available"}

        elif solver_id in ("ista", "ista_tv"):
            if run_ista is not None:
                return run_ista(y, physics, cfg)
            return None, {"error": "ista not available"}

        # GAP-TV for CASSI/CACTI
        elif solver_id == "gap_tv":
            if run_gap_tv is not None:
                return run_gap_tv(y, physics, cfg)
            return None, {"error": "gap_tv not available"}

        elif solver_id == "gap_pnp":
            if run_gap_denoise is not None:
                return run_gap_denoise(y, physics, cfg)
            return None, {"error": "gap_pnp not available"}

        # MRI solvers
        elif solver_id in ("espirit", "sense"):
            if run_espirit_recon is not None:
                return run_espirit_recon(y, physics, cfg)
            return None, {"error": "espirit not available"}

        elif solver_id == "cs_mri":
            if run_cs_mri is not None:
                return run_cs_mri(y, physics, cfg)
            return None, {"error": "cs_mri not available"}

        # SIM solver
        elif solver_id in ("sim", "wiener_sim"):
            if run_sim_reconstruction is not None:
                return run_sim_reconstruction(y, physics, cfg)
            return None, {"error": "sim_solver not available"}

        # Ptychography
        elif solver_id == "epie":
            if run_epie is not None:
                return run_epie(y, physics, cfg)
            return None, {"error": "epie not available"}

        # Holography
        elif solver_id in ("holography", "angular_spectrum"):
            if run_holography_reconstruction is not None:
                return run_holography_reconstruction(y, physics, cfg)
            return None, {"error": "holography_solver not available"}

    except Exception as e:
        return None, {"error": str(e)}

    return None, {"error": f"unknown solver: {solver_id}"}


def run_portfolio(y: np.ndarray, physics: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run a small set of solvers; select best by proxy score.

    Supports both matrix-based and operator-based physics models.

    Args:
        y: Measurements.
        physics: Physics operator (with forward/adjoint or A matrix).
        cfg: Configuration dict with 'candidates', 'max_candidates', etc.

    Returns:
        Tuple of (best_x, best_info).
    """
    cand = cfg.get("candidates", ["cg", "adjoint"])
    best_x = None
    best_score = float("inf")
    best_info: Dict[str, Any] = {}

    for sid in cand[: int(cfg.get("max_candidates", 3))]:
        x: Optional[np.ndarray] = None
        info: Dict[str, Any] = {}

        # Try matrix-based solvers first
        if sid == "lsq" and hasattr(physics, "A") and physics.A is not None:
            A = physics.A
            x = least_squares(A, y.reshape(-1))
            info = {"solver": "lsq"}

        elif sid == "fista" and hasattr(physics, "A") and physics.A is not None:
            A = physics.A
            lam = cfg.get("lambda", 1e-3)
            iters = cfg.get("iters", 50)
            x = fista_l2(y.reshape(-1), A, lam=lam, iters=iters)
            info = {"solver": "fista", "lambda": lam, "iters": iters}

        # Try operator-based solvers
        elif hasattr(physics, 'forward') and hasattr(physics, 'adjoint'):
            x, info = _run_operator_solver(y, physics, sid, cfg)

        # Fallback
        if x is None:
            x = y.astype(np.float32)
            info = {"solver": sid, "note": "fallback identity"}

        score = quick_score_proxy(x, y)
        info["proxy_score"] = float(score)

        if score < best_score:
            best_score = float(score)
            best_x = x
            best_info = info

    if best_x is None:
        # Ultimate fallback
        best_x = y.astype(np.float32)
        best_info = {"solver": "identity", "note": "no solver succeeded"}

    return best_x, best_info
