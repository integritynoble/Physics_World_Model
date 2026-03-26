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
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums for universal Source/Transport/Sensor/Noise decomposition (v3)
# ---------------------------------------------------------------------------


class PhysicsTier(str, Enum):
    """Level of physical fidelity for a primitive or graph node.

    Paper-to-code mapping (paper uses 1-indexed tiers)::

        Paper Tier 1 (linear, shift-invariant)   -> tier0_geometry
        Paper Tier 2 (linear, shift-variant)     -> tier1_approx
        Paper Tier 3 (nonlinear, ray/wave)       -> tier2_full
        Paper Tier 4 (full-wave / Monte Carlo)   -> tier3_learned
    """

    tier0_geometry = "tier0_geometry"
    tier1_approx = "tier1_approx"
    tier2_full = "tier2_full"
    tier3_learned = "tier3_learned"


class NodeRole(str, Enum):
    """Semantic role of a node in the universal forward model decomposition.

    The universal rule is: y ~ Noise(Sensor(Transport/Interaction(Source(x))))
    """

    source = "source"
    transport = "transport"
    interaction = "interaction"
    sensor = "sensor"
    noise = "noise"
    readout = "readout"
    utility = "utility"
    correction = "correction"


class CorrectionKind(str, Enum):
    """Kind of operator correction applied."""

    affine = "affine"
    residual = "residual"
    lut = "lut"
    field_map = "field_map"


class PhysicsSubrole(str, Enum):
    """Fine-grained subrole for element nodes in the canonical chain."""

    propagation = "propagation"      # free-space propagation (Fresnel, angular spectrum, acoustic wave)
    modulation = "modulation"        # coded mask, DMD, SIM pattern
    sampling = "sampling"            # Radon, k-space, random mask
    interaction = "interaction"      # carrier-type transition (photon->acoustic, photon->electron)
    transduction = "transduction"    # domain change within same carrier
    encoding = "encoding"            # temporal/spectral encoding (FLIM, spectral dispersion)
    relay = "relay"                  # Fourier relay, identity propagation segment


class CarrierType(str, Enum):
    """Physical carrier for the signal propagating through the imaging system."""

    photon = "photon"
    electron = "electron"
    acoustic = "acoustic"
    spin = "spin"
    particle_other = "particle_other"
    abstract = "abstract"


class DiffMode(str, Enum):
    """Differentiability mode of a primitive or node."""

    none = "none"
    forward_ad = "forward_ad"
    reverse_ad = "reverse_ad"
    both = "both"
    finite_diff = "finite_diff"


# ---------------------------------------------------------------------------
# Canonical Primitive enums (Finite Primitive Basis Theorem)
# ---------------------------------------------------------------------------


class CanonicalPrimitive(str, Enum):
    """The 11 canonical primitives from the Finite Primitive Basis Theorem.

    Every Tier-2 imaging forward model can be decomposed into a DAG of
    these 11 canonical operators (Theorem 1, FPB paper).
    """

    P = "propagate"       # Free-space wave propagation
    M = "modulate"        # Element-wise multiplication (mask, coil, absorption)
    Pi = "project"        # Radon line-integral projection
    F = "encode"          # Fourier-domain encoding (k-space)
    C = "convolve"        # Spatial convolution (PSF)
    Sigma = "accumulate"  # Summation over spectral/temporal axis
    D = "detect"          # Detector response (5 canonical families)
    S = "sample"          # Sub-sampling on index set
    W = "disperse"        # Wavelength-dependent spatial shift
    R = "scatter"         # Direction change and/or energy shift
    Lambda = "transform"  # Pointwise nonlinear physics (5 canonical families)


class PhysicsStageFamily(str, Enum):
    """The 4 physics-stage families from the FPB Theorem proof.

    Each canonical primitive belongs to exactly one stage family,
    reflecting its role in the physical imaging pipeline.
    """

    propagation = "propagation"                  # → {P, C}
    interaction = "interaction"                   # → {M, R, Λ}
    encoding_projection = "encoding_projection"   # → {Π, F}
    detection_readout = "detection_readout"        # → {Σ, S, W, D}


class TransformFamily(str, Enum):
    """The 5 canonical Transform (Λ) response families.

    Each Transform primitive implements one of these pointwise nonlinear
    physics functions, with at most 2 parameters each.  This bounded
    parametrisation prevents Λ from becoming a universal approximator.
    """

    beer_lambert = "beer_lambert"          # Λ(x) = exp(-μ·x), params: μ
    phase_wrapping = "phase_wrapping"      # Λ(x) = angle(exp(j·x)), params: none
    beam_hardening = "beam_hardening"      # Λ(x) = a₁·x + a₂·x², params: a₁, a₂
    stopping_power = "stopping_power"      # Λ(x) = a/x² (Bethe–Bloch), params: a
    saturation = "saturation"              # Λ(x) = x_max·(1 − exp(−x/x₀)), params: x_max, x₀


class DetectFamily(str, Enum):
    """The 5 canonical Detect response families.

    Each detector primitive implements one of these response functions,
    determining the measurement nonlinearity.
    """

    intensity_square_law = "intensity_square_law"  # η(x) = g|x|² (intensity detector)
    logarithmic = "logarithmic"                    # η(x) = g·log(1 + |x|²/x₀)
    sigmoid = "sigmoid"                            # η(x) = g·σ(|x|² - x₀)
    linear_field = "linear_field"                  # η(x) = gx (field-amplitude detector)
    coherent_field = "coherent_field"              # η(x) = g·Re[x·e^(iφ)]


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
    physics_tier: Optional[PhysicsTier] = None
    node_role: Optional[NodeRole] = None
    carrier_type: Optional[CarrierType] = None
    diff_mode: Optional[DiffMode] = None
    supports_vjp: bool = False
    supports_jvp: bool = False
    physics_subrole: Optional[PhysicsSubrole] = None
    canonical_id: Optional[CanonicalPrimitive] = None
    physics_stage: Optional[PhysicsStageFamily] = None
    detect_family: Optional[DetectFamily] = None
    transform_family: Optional[TransformFamily] = None


# ---------------------------------------------------------------------------
# DriftModel
# ---------------------------------------------------------------------------


class DriftModel(StrictBaseModel):
    """Model for parameter drift over time (e.g. thermal drift, bleaching).

    Attributes
    ----------
    kind : str
        Drift type: ``none``, ``linear``, ``exponential``, ``brownian``.
    rate : float
        Drift rate (units depend on kind).
    time_constant_s : float
        Characteristic time constant in seconds.
    amplitude : float
        Drift amplitude scaling factor.
    """

    kind: str = "none"
    rate: float = 0.0
    time_constant_s: float = 0.0
    amplitude: float = 0.0


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
    carrier_type: Optional[CarrierType] = None
    axes_labels: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PortSpec
# ---------------------------------------------------------------------------


class PortSpec(StrictBaseModel):
    """Named input/output port for multi-input nodes."""
    name: str = "default"
    tensor_spec: Optional[TensorSpec] = None
    required: bool = True


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
    drift_model: Optional[DriftModel] = None
    units: str = "dimensionless"


# ---------------------------------------------------------------------------
# PrimitiveRef — (registry, version, name) triple
# ---------------------------------------------------------------------------


class PrimitiveRef:
    """Reference to a primitive by (registry, version, name) triple.

    Backward-compatible: accepts either a string primitive_id or a triple.
    The canonical string form is ``"registry/version/name"`` (e.g.
    ``"general/v1/Transform"``).  A bare name like ``"Transform"`` is
    interpreted as ``("general", "v1", "Transform")``.
    """

    __slots__ = ("registry", "version", "name", "primitive_id")

    def __init__(
        self,
        primitive_id: str = "",
        registry: str = "general",
        version: str = "v1",
        name: str = "",
    ):
        if primitive_id and not name:
            # Parse from string like "general/v1/Transform" or just "Transform"
            parts = primitive_id.split("/")
            if len(parts) == 3:
                self.registry = parts[0]
                self.version = parts[1]
                self.name = parts[2]
            else:
                self.registry = registry
                self.version = version
                self.name = primitive_id
        else:
            self.registry = registry
            self.version = version
            self.name = name
        self.primitive_id = f"{self.registry}/{self.version}/{self.name}"

    def __repr__(self) -> str:
        return f"PrimitiveRef({self.registry}/{self.version}/{self.name})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.primitive_id == other or self.name == other
        if isinstance(other, PrimitiveRef):
            return self.primitive_id == other.primitive_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.primitive_id)

    @property
    def triple(self) -> Tuple[str, str, str]:
        """Return the (registry, version, name) triple."""
        return (self.registry, self.version, self.name)
