"""pwm_core.recon.protocols
============================

Solver API protocols for OperatorGraph-first enforcement.

All registered solvers must consume a ``LinearLikeOperator`` (which
``GraphOperator`` satisfies) rather than modality-specific forward code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class LinearLikeOperator(Protocol):
    """Protocol for operators consumed by reconstruction solvers.

    ``GraphOperator`` satisfies this protocol.  Solvers must accept this
    interface rather than modality-specific physics code.

    Attributes
    ----------
    x_shape : tuple[int, ...]
        Expected input (object) shape.
    y_shape : tuple[int, ...]
        Expected output (measurement) shape.
    all_linear : bool
        True if every primitive in the graph is linear.
    """

    x_shape: Tuple[int, ...]
    y_shape: Tuple[int, ...]
    all_linear: bool

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the forward model: y = A(x)."""
        ...

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Apply the adjoint: x = A^T(y).  Only valid when all_linear is True."""
        ...
