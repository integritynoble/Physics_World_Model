"""pwm_core.counterfactual.schema
==================================

Pydantic models for counterfactual pack manifests and scenario specifications.

Follows PWM conventions: ``extra="forbid"``, NaN/Inf rejection, strict typing.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

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
# Enums
# ---------------------------------------------------------------------------


class SplitKind(str, Enum):
    """Pack split: probe (public, moderate) or hidden (secret, harder)."""
    probe = "probe"
    hidden = "hidden"


class RegimeKind(str, Enum):
    """Scenario regime describing the type of perturbation applied."""
    nominal = "nominal"
    single_param = "single_param"
    compound = "compound"
    gate_flip = "gate_flip"
    oof = "oof"
    compute_trap = "compute_trap"


class RedTeamCategory(str, Enum):
    """Red Team injection categories from targeting_system.md S4."""
    mismatch_escalation = "mismatch_escalation"
    compound_mismatch = "compound_mismatch"
    gate_flip = "gate_flip"
    out_of_family = "out_of_family"
    compute_trap = "compute_trap"
    distribution_shift = "distribution_shift"
    misleading_consistency = "misleading_consistency"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class MismatchConfig(StrictBaseModel):
    """Single mismatch parameter specification."""
    name: str = Field(..., description="Parameter name (e.g. 'mask_dx')")
    value: float = Field(..., description="Applied value")
    unit: str = Field(..., description="Physical unit (e.g. 'px', 'deg')")
    range_min: float = Field(..., description="Minimum of sampling range")
    range_max: float = Field(..., description="Maximum of sampling range")


class NoiseConfig(StrictBaseModel):
    """Noise model parameters."""
    noise_alpha: Optional[float] = Field(
        None, description="Poisson peak (photons); None if Gaussian-only"
    )
    noise_sigma: float = Field(..., description="Gaussian noise std (normalized)")


class ScenarioSpec(StrictBaseModel):
    """Full specification for a single counterfactual scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    split: SplitKind
    regime: RegimeKind
    red_team_category: Optional[RedTeamCategory] = None
    scene_id: str = Field(..., description="Source scene/image/clip identifier")
    mismatch_params: List[MismatchConfig] = Field(default_factory=list)
    noise_config: NoiseConfig
    seed: int = Field(..., description="RNG seed for reproducibility")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g. oof injection type)",
    )

    model_config = ConfigDict(extra="forbid")


class ExpectedBaseline(StrictBaseModel):
    """Expected reconstruction quality from a specific solver."""
    solver_id: str
    scenario_id: str
    psnr_db: Optional[float] = None
    ssim: Optional[float] = None
    runtime_s: Optional[float] = None


class CounterfactualPackManifest(StrictBaseModel):
    """Top-level manifest for a counterfactual pack."""
    pack_id: str = Field(..., description="e.g. 'cassi_cfpack_v1'")
    version: str = Field(default="1.0.0")
    modality: str = Field(..., description="e.g. 'cassi', 'spc', 'cacti'")
    seeds: Dict[str, int] = Field(
        ..., description="{'probe': ..., 'hidden': ...}"
    )
    n_scenarios: int = Field(..., ge=1)
    regimes: List[str] = Field(default_factory=list)
    solvers: List[str] = Field(default_factory=list)
    scenarios: List[ScenarioSpec] = Field(default_factory=list)
    expected_baselines: List[ExpectedBaseline] = Field(default_factory=list)
    file_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="filepath -> SHA-256 hex digest",
    )

    model_config = ConfigDict(extra="forbid")
