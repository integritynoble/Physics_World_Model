"""pwm_core.graph.ir_types
==========================

Formal IR types for the OperatorGraph intermediate representation.

Types
-----
NodeTags          Per-node semantic tags (linear, stochastic, differentiable, stateful)
TensorSpec        Shape / dtype / unit / domain metadata for graph edges
ParameterSpec     Bounds, prior, parameterization, identifiability hint for learnable params
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# StrictBaseModel (local copy for self-containment)
# ---------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
    """Root model with extra='forbid' and NaN/Inf rejection."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="after")
    def _reject_nan_inf(self) -> "StrictBaseModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(
                    f"Field '{field_name}' contains {val!r}, which is not allowed."
                )
        return self


# ---------------------------------------------------------------------------
# NodeTags
# ---------------------------------------------------------------------------


class NodeTags(StrictBaseModel):
    """Per-node semantic tags derived from the bound primitive.

    Attributes
    ----------
    is_linear : bool
        True if the primitive implements a linear operator (adjoint exists).
    is_stochastic : bool
        True if the primitive involves randomness (noise, random sampling).
    is_differentiable : bool
        True if the primitive's forward is differentiable w.r.t. input.
    is_stateful : bool
        True if the primitive carries mutable state across calls.
    """

    is_linear: bool = True
    is_stochastic: bool = False
    is_differentiable: bool = True
    is_stateful: bool = False


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------


class TensorSpec(StrictBaseModel):
    """Shape / dtype / unit / domain metadata for a graph edge tensor.

    Attributes
    ----------
    shape : list[int]
        Expected tensor shape (may contain -1 for dynamic axes).
    dtype : str
        Numpy dtype string (e.g. ``float64``, ``complex128``).
    unit : str
        Physical unit (e.g. ``photons``, ``radians``, ``arbitrary``).
    domain : str
        Value domain hint (e.g. ``real_nonneg``, ``complex``, ``binary``).
    """

    shape: List[int] = Field(default_factory=lambda: [-1, -1])
    dtype: str = "float64"
    unit: str = "arbitrary"
    domain: str = "real"


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


class ParameterSpec(StrictBaseModel):
    """Metadata for a learnable / calibratable parameter.

    Attributes
    ----------
    name : str
        Parameter name (must match key in node's params dict).
    lower : float
        Lower bound for optimisation.
    upper : float
        Upper bound for optimisation.
    prior : str
        Prior distribution hint (``uniform``, ``log_uniform``, ``normal``).
    parameterization : str
        Transform applied before optimisation (``identity``, ``log``, ``logit``).
    identifiability_hint : str
        Hint from identifiability analysis (``identifiable``, ``weakly``,
        ``unidentifiable``, ``unknown``).
    """

    name: str
    lower: float = 0.0
    upper: float = 1.0
    prior: str = "uniform"
    parameterization: str = "identity"
    identifiability_hint: str = "unknown"
