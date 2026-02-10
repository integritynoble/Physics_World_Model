"""Shared benchmark helpers for graph-first operator construction.

Provides ``build_benchmark_operator()`` which routes through
``physics_factory.build_operator()`` → ``GraphOperatorAdapter`` (graph-first)
with a legacy-operator fallback, so every benchmark uses the same
OperatorGraph infrastructure as the Mode 1 / Mode 2 pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pwm_core.api.types import (
    ExperimentInput,
    ExperimentSpec,
    ExperimentStates,
    InputMode,
    PhysicsState,
    TaskKind,
    TaskState,
)
from pwm_core.core.physics_factory import build_operator
from pwm_core.physics.base import BaseOperator


def quick_spec(modality: str, dims: Tuple[int, ...]) -> ExperimentSpec:
    """Build a minimal ``ExperimentSpec`` suitable for ``build_operator()``.

    Args:
        modality: Modality string (e.g. ``"widefield"``, ``"cassi"``).
        dims: Spatial (and optional spectral/temporal) dimensions,
              e.g. ``(256, 256)`` or ``(256, 256, 28)``.

    Returns:
        A valid ``ExperimentSpec`` with the physics state populated.
    """
    dims_dict: Dict[str, Any] = {}
    if len(dims) >= 2:
        dims_dict["H"] = dims[0]
        dims_dict["W"] = dims[1]
    if len(dims) >= 3:
        dims_dict["D"] = dims[2]

    return ExperimentSpec(
        id=f"benchmark_{modality}",
        input=ExperimentInput(mode=InputMode.simulate),
        states=ExperimentStates(
            physics=PhysicsState(modality=modality, dims=dims_dict),
            task=TaskState(kind=TaskKind.simulate_recon_analyze),
        ),
    )


def build_benchmark_operator(
    modality: str,
    dims: Tuple[int, ...],
    theta: Optional[Dict[str, Any]] = None,
    assets: Optional[Dict[str, Any]] = None,
) -> BaseOperator:
    """Build an operator via the physics-factory graph-first path.

    Creates a minimal ``ExperimentSpec``, calls ``build_operator()`` which:

    1. Tries the graph template first → ``GraphOperatorAdapter``
    2. Falls back to the legacy modality-specific operator

    Then applies *theta* via ``set_theta()`` if provided.

    Args:
        modality: Modality string (``"widefield"``, ``"cassi"``, etc.).
        dims: Spatial / spectral dimensions.
        theta: Optional parameter overrides applied after construction.
        assets: Optional asset dict (e.g. mask array for CASSI).

    Returns:
        A ``BaseOperator`` (usually ``GraphOperatorAdapter``) ready for
        ``forward()`` / ``adjoint()`` / ``set_theta()``.
    """
    spec = quick_spec(modality, dims)
    operator = build_operator(spec)

    if theta:
        operator.set_theta(theta)

    return operator
