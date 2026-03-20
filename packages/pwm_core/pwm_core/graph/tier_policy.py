"""pwm_core.graph.tier_policy
===============================

TierPolicy: select physics fidelity tier based on compute budget.

Tier levels:
  tier0_geometry  — geometric optics, ray matrices, identity ops
  tier1_approx    — Fourier/convolution approximations, angular spectrum, Radon
  tier2_full      — full-wave (FDTD, BPM), Maxwell equations
  tier3_learned   — learned/neural operators (NeRF, 3DGS, deep priors)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.graph.ir_types import PhysicsTier


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              ser_json_inf_nan="constants")

    @model_validator(mode="after")
    def _reject_nan_inf(self) -> "StrictBaseModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"Field '{field_name}' contains {val!r}")
        return self


class TierBudget(StrictBaseModel):
    """Compute budget for tier selection."""
    max_seconds: float = 1.0
    max_memory_mb: float = 1024.0
    accuracy: str = "medium"  # low / medium / high / maximum


# Per-modality minimum tier overrides
_MODALITY_MIN_TIER: Dict[str, PhysicsTier] = {
    "ct": PhysicsTier.tier1_approx,
    "mri": PhysicsTier.tier1_approx,
    "nerf": PhysicsTier.tier3_learned,
    "gaussian_splatting": PhysicsTier.tier3_learned,
}

# Tier ordering for comparison
_TIER_ORDER = {
    PhysicsTier.tier0_geometry: 0,
    PhysicsTier.tier1_approx: 1,
    PhysicsTier.tier2_full: 2,
    PhysicsTier.tier3_learned: 3,
}


class TierPolicy:
    """Select physics fidelity tier based on compute budget and modality."""

    def select_tier(self, modality: str, budget: TierBudget) -> PhysicsTier:
        """Select the appropriate physics tier.

        Parameters
        ----------
        modality : str
            Modality key (e.g., "widefield", "ct", "nerf").
        budget : TierBudget
            Compute budget constraints.

        Returns
        -------
        PhysicsTier
            Selected tier.
        """
        # Budget-based selection
        if budget.accuracy == "low" or budget.max_seconds < 0.1:
            tier = PhysicsTier.tier0_geometry
        elif budget.accuracy == "medium" or budget.max_seconds < 1.0:
            tier = PhysicsTier.tier1_approx
        elif budget.accuracy == "high" or budget.max_seconds < 60.0:
            tier = PhysicsTier.tier2_full
        else:
            tier = PhysicsTier.tier3_learned

        # Apply modality minimum
        min_tier = _MODALITY_MIN_TIER.get(modality)
        if min_tier is not None:
            if _TIER_ORDER[tier] < _TIER_ORDER[min_tier]:
                tier = min_tier

        return tier

    def suggest_primitives(self, modality: str, tier: PhysicsTier) -> List[str]:
        """Suggest primitive_ids appropriate for a modality at a given tier.

        Returns
        -------
        list[str]
            Recommended primitive_ids.
        """
        suggestions: Dict[str, Dict[PhysicsTier, List[str]]] = {
            "widefield": {
                PhysicsTier.tier0_geometry: ["identity"],
                PhysicsTier.tier1_approx: ["conv2d", "fourier_relay"],
                PhysicsTier.tier2_full: ["maxwell_interface"],
                PhysicsTier.tier3_learned: ["conv2d"],
            },
            "ct": {
                PhysicsTier.tier1_approx: ["ct_radon"],
                PhysicsTier.tier2_full: ["ct_radon"],
            },
            "mri": {
                PhysicsTier.tier1_approx: ["mri_kspace"],
                PhysicsTier.tier2_full: ["mri_kspace"],
            },
        }

        modality_map = suggestions.get(modality, {})
        return modality_map.get(tier, [])
