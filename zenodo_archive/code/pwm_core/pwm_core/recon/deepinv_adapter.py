"""pwm_core.recon.deepinv_adapter

Adapter layer that maps PWM PhysicsOperator -> deepinv Physics + solver recipes.

This file is intentionally lightweight; concrete recipes live in portfolio.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import deepinv as dinv
except Exception:  # pragma: no cover
    dinv = None


@dataclass
class DeepInvRecipe:
    method: str  # e.g., "pnp_pgd", "unrolled", "diffusion", "tv"
    params: Dict[str, Any]


def has_deepinv() -> bool:
    return dinv is not None


def build_deepinv_physics(pwm_physics: Any) -> Any:
    """Bridge PWM PhysicsOperator to deepinv Physics.

    In a full implementation, you'd wrap forward/adjoint.
    Here we simply return pwm_physics if it already *is* a deepinv Physics.
    """
    if dinv is None:
        raise RuntimeError("deepinv is not installed")
    # If already a deepinv physics, pass-through.
    if hasattr(pwm_physics, "A") or pwm_physics.__class__.__name__.lower().endswith("physics"):
        return pwm_physics
    # Otherwise, create a custom physics wrapper (left as future work).
    raise NotImplementedError("Custom PWM->deepinv physics wrapping not implemented in this starter stub.")
