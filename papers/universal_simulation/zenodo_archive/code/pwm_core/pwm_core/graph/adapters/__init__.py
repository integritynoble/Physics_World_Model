"""pwm_core.graph.adapters
==========================

Physics Adapter Layer for Tier 0-3 integration.

Each adapter wraps a raw physics kernel into a valid PrimitiveOp,
handling validation, adjoint checking, shape normalization, tier tagging,
and serialization boilerplate.

Contributors implement only their physics forward/adjoint functions.
The adapter handles everything else.

Tiers
-----
Tier0Adapter  — Geometry: coordinate transforms, projection, ray tracing
Tier1Adapter  — Wave approx: Fourier optics, paraxial, Born approximation
Tier2Adapter  — Full transport: Maxwell, scattering, Monte Carlo wrapping
Tier3Adapter  — Learned surrogates: uncertainty calibration, error bars
"""

from pwm_core.graph.adapters.tier_adapter import TierAdapter
from pwm_core.graph.adapters.tier0_adapter import Tier0Adapter
from pwm_core.graph.adapters.tier1_adapter import Tier1Adapter
from pwm_core.graph.adapters.tier2_adapter import Tier2Adapter
from pwm_core.graph.adapters.tier3_adapter import Tier3Adapter

__all__ = [
    "TierAdapter",
    "Tier0Adapter",
    "Tier1Adapter",
    "Tier2Adapter",
    "Tier3Adapter",
]
