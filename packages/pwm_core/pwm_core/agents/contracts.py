"""
pwm_core.agents.contracts
=========================

Core pydantic v2 contracts for the PWM agent system.

Every agent-to-agent message is a validated pydantic model with strict mode.
Invalid structures, extra fields, NaN/Inf values are all rejected before
execution. This module defines ALL schemas used for inter-agent communication.

Design invariants
-----------------
* ``StrictBaseModel`` is the common ancestor -- no extra fields, no NaN/Inf.
* All bounded scores use ``Field(ge=0.0, le=1.0)``.
* Enums are ``str`` enums so they serialise as readable strings in JSON.
* Models are pure data containers; business logic lives in agent modules.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
    """Root model for all PWM agent contracts.

    Guarantees:
    * ``extra="forbid"`` -- unexpected fields are rejected.
    * ``validate_assignment=True`` -- mutations are re-validated.
    * No ``float`` field may contain NaN or Inf.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="after")
    def _reject_nan_inf(self) -> "StrictBaseModel":
        """Reject NaN and Inf in any float field (including Optional[float])."""
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(
                    f"Field '{field_name}' contains {val!r}, which is not "
                    "allowed. Check upstream computation."
                )
        return self


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModeRequested(str, Enum):
    """User-facing mode selector parsed from the prompt."""

    simulate = "simulate"
    operator_correction = "operator_correction"
    auto = "auto"


class OperatorType(str, Enum):
    """How the forward operator is characterised at plan time."""

    explicit_matrix = "explicit_matrix"
    linear_operator = "linear_operator"
    nonlinear_operator = "nonlinear_operator"
    unknown = "unknown"


class TransferKind(str, Enum):
    """Physical transfer mechanism of an optical element."""

    convolution = "convolution"
    modulation = "modulation"
    dispersion = "dispersion"
    projection = "projection"
    interference = "interference"
    propagation = "propagation"
    integration = "integration"
    sampling = "sampling"
    nonlinear = "nonlinear"
    identity = "identity"


class NoiseKind(str, Enum):
    """Noise source associated with an optical element."""

    none = "none"
    shot_poisson = "shot_poisson"
    read_gaussian = "read_gaussian"
    quantization = "quantization"
    fixed_pattern = "fixed_pattern"
    aberration = "aberration"
    alignment = "alignment"
    thermal = "thermal"
    acoustic = "acoustic"


class NoiseRegime(str, Enum):
    """Dominant noise regime for the imaging system."""

    photon_starved = "photon_starved"
    shot_limited = "shot_limited"
    read_limited = "read_limited"
    detector_limited = "detector_limited"
    background_limited = "background_limited"


class SignalPriorClass(str, Enum):
    """Signal prior family for recoverability estimation."""

    tv = "tv"
    wavelet_sparse = "wavelet_sparse"
    low_rank = "low_rank"
    deep_prior = "deep_prior"
    spectral_sparse = "spectral_sparse"
    joint_spatio_spectral = "joint_spatio_spectral"
    temporal_smooth = "temporal_smooth"


class ForwardModelType(str, Enum):
    """Type of the forward model operator."""

    explicit_matrix = "explicit_matrix"
    linear_operator = "linear_operator"
    nonlinear_operator = "nonlinear_operator"


# ---------------------------------------------------------------------------
# Intent models
# ---------------------------------------------------------------------------


class PlanIntent(StrictBaseModel):
    """What the user wants to do.  Parsed from prompt by the Plan Agent."""

    mode_requested: ModeRequested = ModeRequested.auto
    has_measured_y: bool = False
    has_operator_A: bool = False
    operator_type: OperatorType = OperatorType.unknown
    user_prompt: str
    raw_file_paths: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Modality selection
# ---------------------------------------------------------------------------


class ModalitySelection(StrictBaseModel):
    """Plan Agent output: which modality was selected and why.

    ``modality_key`` is NOT validated inside this model -- the model does not
    know the registry.  Validation happens immediately after parsing::

        selection = ModalitySelection.model_validate(raw)
        registry.assert_modality_exists(selection.modality_key)  # hard-fail
    """

    modality_key: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    fallback_modalities: List[str] = Field(default_factory=list, max_length=3)


# ---------------------------------------------------------------------------
# Imaging system
# ---------------------------------------------------------------------------


class ElementSpec(StrictBaseModel):
    """One optical element in the imaging system."""

    name: str
    element_type: Literal[
        "source",
        "lens",
        "mask",
        "modulator",
        "disperser",
        "filter",
        "beamsplitter",
        "detector",
        "transducer",
        "medium",
        "sample",
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    transfer_kind: TransferKind
    noise_kinds: List[NoiseKind] = Field(default_factory=list)
    transfer_equation: str = ""
    throughput: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Fraction of signal transmitted through this element.",
    )


class ImagingSystem(StrictBaseModel):
    """Complete imaging system design assembled by the System Agent.

    The element list must contain at least one detector or transducer.
    """

    modality_key: str
    elements: List[ElementSpec]
    forward_model_type: ForwardModelType
    forward_model_equation: str
    signal_dims: Dict[str, List[int]]
    wavelength_nm: Optional[float] = None
    spectral_range_nm: Optional[List[float]] = None

    @field_validator("elements")
    @classmethod
    def must_have_detector(cls, v: List[ElementSpec]) -> List[ElementSpec]:
        """Ensure the element chain includes at least one detector/transducer."""
        has_detector = any(
            e.element_type in ("detector", "transducer") for e in v
        )
        if not has_detector:
            raise ValueError(
                "Imaging system must include at least one detector/transducer"
            )
        return v


# ---------------------------------------------------------------------------
# Agent reports
# ---------------------------------------------------------------------------


class PhotonReport(StrictBaseModel):
    """Photon Agent output.  All numbers are deterministic."""

    n_photons_per_pixel: float
    snr_db: float
    noise_regime: NoiseRegime
    shot_noise_sigma: float
    read_noise_sigma: float
    total_noise_sigma: float
    feasible: bool
    quality_tier: Literal["excellent", "acceptable", "marginal", "insufficient"]
    throughput_chain: List[Dict[str, float]]
    noise_model: Literal["poisson", "gaussian", "mixed_poisson_gaussian"]
    explanation: str = ""
    recommended_levels: Optional[Dict[str, Dict[str, Any]]] = None
    noise_recipe: Optional[Dict[str, Any]] = None


class MismatchReport(StrictBaseModel):
    """Mismatch Agent output.

    Numbers are deterministic; family choice may be LLM-assisted.
    """

    modality_key: str
    mismatch_family: str
    parameters: Dict[str, Dict[str, Any]]
    severity_score: float = Field(ge=0.0, le=1.0)
    correction_method: str
    expected_improvement_db: float
    explanation: str = ""
    param_types: Optional[Dict[str, str]] = None
    subpixel_warnings: Optional[List[str]] = None


class RecoverabilityReport(StrictBaseModel):
    """Recoverability Agent output (formerly CompressionReport).

    Uses practical recoverability models -- calibration-table lookups with
    interpolation, *not* synthetic formulas.  Includes an interpolation
    confidence band so downstream agents know how trustworthy the estimate is.
    """

    compression_ratio: float
    noise_regime: NoiseRegime
    signal_prior_class: SignalPriorClass
    operator_diversity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Proxy for incoherence / pattern diversity.",
    )
    condition_number_proxy: float
    recoverability_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Empirical score from calibration-table lookup with interpolation."
        ),
    )
    recoverability_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the recoverability estimate: 1.0 = exact table "
            "match, lower = interpolated or extrapolated."
        ),
    )
    expected_psnr_db: float = Field(
        description="Expected PSNR from calibration table (dB)."
    )
    expected_psnr_uncertainty_db: float = Field(
        ge=0.0,
        description="Uncertainty band: +/- dB around expected_psnr_db.",
    )
    recommended_solver_family: str
    verdict: Literal["excellent", "sufficient", "marginal", "insufficient"]
    calibration_table_entry: Optional[Dict[str, Any]] = None
    explanation: str = ""


class BottleneckScores(StrictBaseModel):
    """Normalised bottleneck severity in each subsystem (0 = no issue, 1 = severe)."""

    photon: float = Field(ge=0.0, le=1.0)
    mismatch: float = Field(ge=0.0, le=1.0)
    compression: float = Field(ge=0.0, le=1.0)
    solver: float = Field(ge=0.0, le=1.0)


class Suggestion(StrictBaseModel):
    """One actionable suggestion from the Analysis Agent."""

    action: str
    priority: Literal["critical", "high", "medium", "low"]
    expected_improvement_db: float
    parameter_path: Optional[str] = None
    parameter_change: Optional[Dict[str, Any]] = None
    details: str


class SystemAnalysis(StrictBaseModel):
    """Analysis Agent output: bottleneck diagnosis and improvement plan."""

    primary_bottleneck: str
    bottleneck_scores: BottleneckScores
    suggestions: List[Suggestion]
    overall_verdict: str
    probability_of_success: float = Field(
        ge=0.0,
        le=1.0,
        description="Pre-flight likelihood of a useful reconstruction.",
    )
    explanation: str = ""


# ---------------------------------------------------------------------------
# Pre-flight report
# ---------------------------------------------------------------------------


class PreFlightReport(StrictBaseModel):
    """Summary shown to the user before heavy computation begins.

    The user must approve this report before reconstruction proceeds.
    """

    modality: ModalitySelection
    system: ImagingSystem
    photon: PhotonReport
    mismatch: MismatchReport
    recoverability: RecoverabilityReport
    analysis: SystemAnalysis
    estimated_runtime_s: float
    proceed_recommended: bool
    warnings: List[str] = Field(default_factory=list)
    what_to_upload: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Negotiation & veto
# ---------------------------------------------------------------------------


class VetoReason(StrictBaseModel):
    """A single reason an agent vetoes proceeding with reconstruction."""

    source: str
    reason: str
    suggested_resolution: List[str] = Field(default_factory=list)


class NegotiationResult(StrictBaseModel):
    """Outcome of the inter-agent negotiation loop.

    The ``AgentNegotiator`` resolves conflicts between Photon, Recoverability,
    and Mismatch agents.  If any veto is present, ``proceed`` is ``False``.
    """

    vetoes: List[VetoReason] = Field(default_factory=list)
    proceed: bool
    probability_of_success: float = Field(
        ge=0.0,
        le=1.0,
        description="Joint probability of useful reconstruction after negotiation.",
    )


# ---------------------------------------------------------------------------
# Operator adjoint check
# ---------------------------------------------------------------------------


class AdjointCheckReport(StrictBaseModel):
    """Result of ``PhysicsOperator.check_adjoint()`` self-test.

    Verifies ``<x, A^T y> == <Ax, y>`` for random vectors.
    """

    passed: bool
    n_trials: int
    max_relative_error: float
    mean_relative_error: float
    tolerance: float
    details: List[Dict[str, float]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Calibration-table interpolation result
# ---------------------------------------------------------------------------


class InterpolatedResult(StrictBaseModel):
    """Result of calibration-table lookup with nearest-neighbour interpolation.

    Used internally by the Recoverability Agent when the exact
    (compression-ratio, noise-regime, solver) triple is not in the table.
    """

    recoverability: float = Field(ge=0.0, le=1.0)
    expected_psnr_db: float
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 = exact table match, lower = interpolated.",
    )
    uncertainty_db: float = Field(
        ge=0.0,
        description="Uncertainty band around expected_psnr_db (dB).",
    )
    raw_entry: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# LLM selection result
# ---------------------------------------------------------------------------


class CorrectionResult(StrictBaseModel):
    """Result of operator parameter correction with uncertainty.

    Produced by the bootstrap_correction() pipeline in
    ``mismatch.uncertainty``.  Stores corrected parameters, their 95%
    confidence intervals, convergence history, and the full set of
    bootstrap bookkeeping needed for deterministic reproducibility
    (seeds + resampling indices), suitable for storage in a RunBundle.
    """

    theta_corrected: Dict[str, float]
    """Corrected operator parameters.  Keys are parameter names."""

    theta_uncertainty: Dict[str, List[float]]
    """95% confidence interval per parameter.  Keys match theta_corrected.
    Each value is [lower_bound, upper_bound]."""

    improvement_db: float
    """PSNR improvement in dB (corrected - uncorrected)."""

    n_evaluations: int = Field(gt=0)
    """Total number of forward model evaluations during correction."""

    convergence_curve: List[float]
    """PSNR (or loss) at each major iteration.  Length = number of
    iterations."""

    bootstrap_seeds: List[int]
    """RNG seeds used for each bootstrap resample.  Length = K (typically
    20).  Stored for deterministic reproducibility."""

    resampling_indices: List[List[int]]
    """Bootstrap resampling indices.  ``resampling_indices[k]`` is the
    list of sample indices used in bootstrap resample *k*.  Stored in
    RunBundle."""

    @field_validator("convergence_curve")
    @classmethod
    def _convergence_non_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("convergence_curve must be non-empty.")
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"convergence_curve[{i}] = {val!r} is not a finite number."
                )
        return v

    @model_validator(mode="after")
    def _validate_correction_result(self) -> "CorrectionResult":
        """Cross-field validation per the CorrectionResult schema contract."""
        # theta_uncertainty keys must be a superset of theta_corrected keys
        corrected_keys = set(self.theta_corrected.keys())
        uncertainty_keys = set(self.theta_uncertainty.keys())
        if not corrected_keys.issubset(uncertainty_keys):
            missing = corrected_keys - uncertainty_keys
            raise ValueError(
                f"theta_uncertainty is missing keys present in "
                f"theta_corrected: {missing}"
            )

        # Each CI must be [lower, upper] with lower <= upper
        for param, ci in self.theta_uncertainty.items():
            if len(ci) != 2:
                raise ValueError(
                    f"theta_uncertainty['{param}'] must have exactly 2 "
                    f"elements [lower, upper], got {len(ci)}."
                )
            lo, hi = ci
            if lo > hi:
                raise ValueError(
                    f"theta_uncertainty['{param}']: lower ({lo}) > "
                    f"upper ({hi})."
                )

        # bootstrap_seeds and resampling_indices must have the same length
        if len(self.bootstrap_seeds) != len(self.resampling_indices):
            raise ValueError(
                f"len(bootstrap_seeds) = {len(self.bootstrap_seeds)} != "
                f"len(resampling_indices) = {len(self.resampling_indices)}."
            )

        return self


class LLMSelectionResult(StrictBaseModel):
    """What the LLM returns when choosing domain concepts.

    Every key is validated against the YAML registry immediately after
    parsing via ``validate_llm_selection()``.  If the LLM hallucinates a
    key that does not exist in the registry, the system falls back to a
    deterministic default.
    """

    modality_key: str
    mismatch_family_id: str
    signal_prior_class: str
    solver_family_id: str
