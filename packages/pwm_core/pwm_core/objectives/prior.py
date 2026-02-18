"""pwm_core.objectives.prior
=============================

PriorSpec for regularization terms used alongside NLL objectives.

Supported kinds: tv, l1_wavelet, low_rank, deep_prior, l2, none.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
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


_VALID_PRIOR_KINDS = {"tv", "l1_wavelet", "low_rank", "deep_prior", "l2", "none"}


class PriorSpec(StrictBaseModel):
    """Regularization prior specification.

    Attributes
    ----------
    kind : str
        Prior type: ``tv``, ``l1_wavelet``, ``low_rank``, ``deep_prior``,
        ``l2``, ``none``.
    weight : float
        Regularization weight (lambda). Must be >= 0.
    params : dict
        Additional prior-specific parameters (e.g. wavelet name, rank).
    """

    kind: str = "none"
    weight: float = 0.0
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_kind(self) -> "PriorSpec":
        if self.kind not in _VALID_PRIOR_KINDS:
            raise ValueError(
                f"Invalid prior kind '{self.kind}'. "
                f"Must be one of: {sorted(_VALID_PRIOR_KINDS)}"
            )
        if self.weight < 0:
            raise ValueError(f"Prior weight must be >= 0, got {self.weight}")
        return self
