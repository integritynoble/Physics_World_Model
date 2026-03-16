"""Central algorithm registry for PWM Algorithm Base.

Provides unified access to all 168 modalities and their solvers.
"""
from __future__ import annotations

import importlib
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple


def list_modalities() -> List[str]:
    """List all available modality IDs."""
    from pathlib import Path
    base = Path(__file__).parent
    return sorted([
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "solvers.py").exists()
    ])


def list_solvers(modality_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    """List all solvers for a modality.

    Returns:
        List of (solver_key, solver_info) tuples.
    """
    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")
    return mod.list_solvers()


def get_solver(modality_id: str, solver_key: str) -> Callable:
    """Get a solver function by modality and key.

    Returns:
        Callable with signature fn(y, operator, cfg) -> x_hat
    """
    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")
    return mod._load_fn(solver_key)


def run_solver(
    modality_id: str,
    solver_key: str,
    y: np.ndarray,
    operator: Any = None,
    cfg: Optional[Dict] = None,
) -> np.ndarray:
    """Run a solver directly.

    Args:
        modality_id: e.g. "cassi", "mri", "ct"
        solver_key: e.g. "traditional_cpu", "best_quality"
        y: Measurement data
        operator: Forward operator
        cfg: Hyperparameters

    Returns:
        x_hat: Reconstructed signal

    Example:
        >>> from algorithm_base import run_solver
        >>> x_hat = run_solver("cassi", "traditional_cpu", y, op)
    """
    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")
    return mod.run_solver(solver_key, y, operator, cfg)
