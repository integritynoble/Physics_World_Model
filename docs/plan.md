# PWM Agent System — Plan v3 (Hardened)

> **Key v3 changes from v2:** LLM returns only registry IDs (mechanically enforced).
> Pydantic strict mode (`extra="forbid"`) everywhere. YAML registries validated by
> Pydantic schemas with provenance fields. Operator `serialize()` includes hashes +
> blob paths. Built-in `check_adjoint()` on every operator. Noise model uses variance
> dominance, not thresholds. CandidateCache bug fixed. Calibration table lookup with
> interpolation + uncertainty bands. CLI permit modes (`--auto-proceed`, `--force`).
> Expanded RunBundle provenance (git hash, seeds, platform). Registry integrity +
> contract fuzzing tests. Multi-LLM fallback (Gemini, Claude, OpenAI).
> `CompressedAgent` renamed to `RecoverabilityAgent`.

---

## Table of Contents

1. [Architecture: LLM vs Deterministic Split](#1-architecture-llm-vs-deterministic-split)
2. [Pydantic Contracts (Enforced Schemas)](#2-pydantic-contracts-enforced-schemas)
3. [YAML Registries](#3-yaml-registries)
4. [Unified Operator Interface](#4-unified-operator-interface)
5. [Imaging Modality Database (26 Modalities)](#5-imaging-modality-database-26-modalities)
6. [Plan Agent (Orchestrator)](#6-plan-agent-orchestrator)
7. [Photon Agent (Deterministic + LLM Narrative)](#7-photon-agent)
8. [Mismatch Agent (Deterministic + LLM Prior Selection)](#8-mismatch-agent)
9. [Recoverability Agent (Practical Recoverability Model)](#9-recoverability-agent)
10. [Analysis Agent + Self-Improvement Loop](#10-analysis-agent--self-improvement-loop)
11. [Physical Continuity Check & Agent Negotiation](#11-physical-continuity-check--agent-negotiation)
12. [Pre-Flight Report & Permit Step](#12-pre-flight-report--permit-step)
13. [Operating Modes & Intent Detection](#13-operating-modes--intent-detection)
14. [Element Visualization: Physics Stage vs Illustration Stage](#14-element-visualization)
15. [Per-Modality Metrics (Beyond PSNR)](#15-per-modality-metrics)
16. [UPWMI Operator Correction: Scoring, Caching, Budget Control](#16-upwmi-operator-correction)
17. [Hybrid Modalities](#17-hybrid-modalities)
18. [RunBundle Export + Interactive Viewer](#18-runbundle-export--interactive-viewer)
19. [Implementation Phases (Incremental, Prove-Value-First)](#19-implementation-phases)
20. [File Structure](#20-file-structure)
21. [Testing Strategy](#21-testing-strategy)
22. [Summary](#22-summary)

---

## 1. Architecture: LLM vs Deterministic Split

### 1.1 Design Principle

**The LLM does NOT compute physics.** It chooses priors and explains results.
**The LLM returns only registry IDs.** Every LLM output that selects a modality,
mismatch family, signal prior, or solver must be a key that exists in a YAML registry.
If validation fails, hard-fail or fall back to a deterministic default.

| Responsibility | Owner | Why |
|---|---|---|
| Modality mapping / routing | **LLM** | Semantic understanding of user intent |
| Propose candidate system designs | **LLM** | Creative, context-dependent |
| Choose parameter priors & mismatch families | **LLM** (returns registry IDs only) | Domain reasoning, but constrained |
| Generate human-readable explanations | **LLM** | Natural language generation |
| Photon budget computation | **Deterministic code** | Must be reproducible & auditable |
| Noise sampling | **Deterministic code** | Seeded RNG, exact formulas |
| Mismatch operator construction | **Deterministic code** | Matrix algebra, no hallucination |
| Recoverability metrics | **Deterministic code** | Calibration tables, not formulas |
| Reconstruction execution | **Deterministic code** | Existing solvers |
| Scoring + beam search (UPWMI) | **Deterministic code** | Must converge reliably |

**Result:** Same creativity, far less "numerical hallucination."

### 1.1b LLM Returns Only Registry IDs (Mechanical Enforcement)

Every LLM call that selects a domain concept must return a **key** validated against
the registry. The enforcement is mechanical, not trust-based:

```python
class LLMSelectionResult(StrictBaseModel):
    """What the LLM returns when choosing domain concepts."""
    modality_key: str           # Must exist in modalities.yaml
    mismatch_family_id: str     # Must exist in mismatch_db.yaml
    signal_prior_class: str     # Must exist in compression_db.yaml
    solver_family_id: str       # Must exist in solver_registry.yaml

def validate_llm_selection(selection: LLMSelectionResult,
                           registry: RegistryBuilder) -> LLMSelectionResult:
    """Hard-fail if LLM returned a hallucinated key."""
    registry.assert_modality_exists(selection.modality_key)
    registry.assert_mismatch_family_exists(
        selection.modality_key, selection.mismatch_family_id)
    registry.assert_signal_prior_exists(selection.signal_prior_class)
    registry.assert_solver_exists(selection.solver_family_id)
    return selection

def safe_llm_select(llm_client, prompt: str, registry: RegistryBuilder,
                    fallback: LLMSelectionResult) -> LLMSelectionResult:
    """Call LLM, validate result, fall back to deterministic default on failure."""
    try:
        raw = llm_client.select(prompt, available_keys=registry.all_keys())
        selection = LLMSelectionResult.model_validate(raw)
        return validate_llm_selection(selection, registry)
    except (ValidationError, RegistryKeyError) as e:
        logger.warning(f"LLM returned invalid key, falling back: {e}")
        return fallback
```

### 1.2 Agent ≠ LLM Call (Design Mantra)

An "Agent" is a **module with a contract**. It **must run without any LLM** and
produce deterministic outputs. The LLM is an optional enhancement for priors + narrative:

```python
class BaseAgent(ABC):
    """An agent is a module with a contract, not necessarily an LLM call.

    Design mantra: agents must run without LLM and produce deterministic outputs.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 registry: Optional[RegistryBuilder] = None):
        self.llm = llm_client      # Optional: used for priors + narrative
        self.registry = registry    # Required: source of truth

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        """Deterministic computation. May call self.llm for priors/explanation."""
        ...

    def explain(self, result: AgentResult) -> str:
        """Use LLM to generate human-readable narrative. Falls back to template."""
        if self.llm:
            return self.llm.narrate(result.to_dict())
        return result.default_narrative()
```

Photon, Mismatch, and Recoverability agents are **deterministic by default**. The LLM is called only to:
1. Select which prior/model family to use **(must return a registry ID, validated mechanically)**
2. Generate the human-readable explanation paragraph

### 1.3 Multi-LLM Support + API Key Management

**No hardcoded API keys.** Keys are loaded from environment. If Gemini doesn't work,
fall back to Claude, Sonnet, Haiku, or OpenAI GPT-4o-mini:

```python
class LLMProvider(str, Enum):
    gemini = "gemini"
    claude = "claude"
    openai = "openai"

class LLMClient:
    """Unified LLM client with provider fallback chain."""

    # Provider priority: try each in order until one works
    FALLBACK_CHAIN = [
        (LLMProvider.gemini, "PWM_GEMINI_API_KEY", "gemini-2.5-pro"),
        (LLMProvider.claude, "PWM_ANTHROPIC_API_KEY", "claude-sonnet-4-5-20250929"),
        (LLMProvider.openai, "PWM_OPENAI_API_KEY", "gpt-4o-mini"),
    ]

    def __init__(self, provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        if provider and api_key:
            # Explicit provider
            self.provider = LLMProvider(provider)
            self.api_key = api_key
            self.model = model or self._default_model(self.provider)
        else:
            # Auto-detect from environment, walk fallback chain
            self.provider, self.api_key, self.model = self._auto_detect()

    def _auto_detect(self):
        for provider, env_var, default_model in self.FALLBACK_CHAIN:
            key = os.environ.get(env_var)
            if key:
                return provider, key, default_model
        raise PWMError(
            "No LLM API key found. Set one of: "
            "PWM_GEMINI_API_KEY, PWM_ANTHROPIC_API_KEY, or PWM_OPENAI_API_KEY"
        )

    def select(self, prompt: str, available_keys: List[str]) -> dict:
        """Ask LLM to select from available registry keys."""
        system = (
            f"You must respond with ONLY keys from this list: {available_keys}. "
            "Return valid JSON. Do not invent new keys."
        )
        return self._call(system, prompt, response_format="json")

    def narrate(self, data: dict) -> str:
        """Generate human-readable explanation."""
        return self._call("Explain this imaging analysis result concisely.", str(data))

    def _call(self, system: str, user: str, **kwargs) -> Any:
        """Dispatch to provider-specific API."""
        if self.provider == LLMProvider.gemini:
            return self._call_gemini(system, user, **kwargs)
        elif self.provider == LLMProvider.claude:
            return self._call_claude(system, user, **kwargs)
        elif self.provider == LLMProvider.openai:
            return self._call_openai(system, user, **kwargs)
```

All API keys stored in `.env` (gitignored), never in source code.

### 1.4 Communication Flow

```
User Prompt
    │
    ▼
┌──────────────────────────────────────┐
│  Plan Agent (Orchestrator)           │
│  LLM: modality mapping, system       │
│  design proposal, parameter priors   │
│  Code: spec construction, validation │
└──────────┬───────────┬───────────┬───┘
           │           │           │
     ┌─────▼─────┐ ┌──▼──────┐ ┌─▼──────────┐
     │  Photon   │ │Mismatch │ │Recoverabil- │
     │  Agent    │ │ Agent   │ │ ity Agent   │
     │           │ │         │ │             │
     │ CODE:     │ │ CODE:   │ │ CODE:       │
     │ physics   │ │ operator│ │ recoverab-  │
     │ formulas  │ │ algebra │ │ ility model │
     │           │ │         │ │             │
     │ LLM:      │ │ LLM:    │ │ LLM:        │
     │ explain   │ │ choose  │ │ explain     │
     │ verdict   │ │ family  │ │ tradeoffs   │
     └─────┬─────┘ └──┬──────┘ └─┬──────────┘
           │           │           │
     ┌─────▼───────────▼───────────▼───┐
     │  Physical Continuity Check      │
     │  (deterministic: NA matching,   │
     │   spectral consistency, etc.)   │
     └─────────────┬───────────────────┘
                   │
     ┌─────────────▼───────────────────┐
     │  Pre-Flight Report + Permit     │
     │  (User approves before heavy    │
     │   reconstruction runs)          │
     └─────────────┬───────────────────┘
                   │  User: "Proceed"
     ┌─────────────▼───────────────────┐
     │  PWM Pipeline (Deterministic)   │
     │  Sim/Load → Recon → Metrics     │
     └─────────────┬───────────────────┘
                   │
     ┌─────────────▼───────────────────┐
     │  Analysis Agent                 │
     │  CODE: bottleneck scoring       │
     │  LLM: explanation + advice      │
     │  Optional: self-improvement loop│
     └─────────────┬───────────────────┘
                   │
     ┌─────────────▼───────────────────┐
     │  RunBundle Export + Viewer       │
     └─────────────────────────────────┘
```

---

## 2. Pydantic Contracts (Enforced Schemas)

Every agent-to-agent message is a validated pydantic model with **strict mode**.
Invalid structures, extra fields, NaN/Inf values are all rejected before execution.

### 2.1 Strict Base Model

All contracts inherit from `StrictBaseModel` which enforces `extra="forbid"` and
`validate_assignment=True`. This prevents silent acceptance of garbage:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import math

class StrictBaseModel(BaseModel):
    """All PWM contracts inherit from this. No extra fields, no NaN/Inf."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",  # serialize NaN/Inf as strings for debugging
    )

    @model_validator(mode="after")
    def _reject_nan_inf(self):
        """Reject NaN and Inf in any float field."""
        for field_name, field_info in self.model_fields.items():
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(
                    f"Field '{field_name}' contains {val}, which is not allowed. "
                    "Check upstream computation."
                )
        return self
```

### 2.2 Core Contracts

```python
# --- agents/contracts.py ---

# ── Intent ──

class ModeRequested(str, Enum):
    simulate = "simulate"
    operator_correction = "operator_correction"
    auto = "auto"

class OperatorType(str, Enum):
    explicit_matrix = "explicit_matrix"
    linear_operator = "linear_operator"
    nonlinear_operator = "nonlinear_operator"
    unknown = "unknown"

class PlanIntent(StrictBaseModel):
    """What the user wants to do. Parsed from prompt."""
    mode_requested: ModeRequested = ModeRequested.auto
    has_measured_y: bool = False
    has_operator_A: bool = False
    operator_type: OperatorType = OperatorType.unknown
    user_prompt: str
    raw_file_paths: List[str] = Field(default_factory=list)


# ── Modality Selection ──

class ModalitySelection(StrictBaseModel):
    """Plan Agent output: which modality and why.

    NOTE: modality_key is NOT validated inside the model (the model doesn't
    know the registry). Validation happens immediately after parsing:

        selection = ModalitySelection.model_validate(raw)
        registry.assert_modality_exists(selection.modality_key)  # hard-fail
    """
    modality_key: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    fallback_modalities: List[str] = Field(default_factory=list, max_length=3)


# ── Imaging System ──

class TransferKind(str, Enum):
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
    none = "none"
    shot_poisson = "shot_poisson"
    read_gaussian = "read_gaussian"
    quantization = "quantization"
    fixed_pattern = "fixed_pattern"
    aberration = "aberration"
    alignment = "alignment"
    thermal = "thermal"
    acoustic = "acoustic"

class ElementSpec(StrictBaseModel):
    """One optical element in the imaging system."""
    name: str
    element_type: Literal["source", "lens", "mask", "modulator",
                          "disperser", "filter", "beamsplitter",
                          "detector", "transducer", "medium", "sample"]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    transfer_kind: TransferKind
    noise_kinds: List[NoiseKind] = Field(default_factory=list)
    transfer_equation: str = ""
    throughput: float = Field(ge=0.0, le=1.0, default=1.0,
                              description="Fraction of signal transmitted")

class ForwardModelType(str, Enum):
    explicit_matrix = "explicit_matrix"
    linear_operator = "linear_operator"
    nonlinear_operator = "nonlinear_operator"

class ImagingSystem(StrictBaseModel):
    """Complete imaging system design."""
    modality_key: str
    elements: List[ElementSpec]
    forward_model_type: ForwardModelType
    forward_model_equation: str
    signal_dims: Dict[str, List[int]]   # {"x": [256,256,28], "y": [283,256]}
    wavelength_nm: Optional[float] = None
    spectral_range_nm: Optional[List[float]] = None

    @field_validator("elements")
    @classmethod
    def must_have_detector(cls, v):
        has_detector = any(e.element_type in ("detector", "transducer") for e in v)
        if not has_detector:
            raise ValueError("Imaging system must include at least one detector/transducer")
        return v


# ── Agent Reports ──

class NoiseRegime(str, Enum):
    photon_starved = "photon_starved"    # <100 photons/pixel
    shot_limited = "shot_limited"        # 100-10k photons/pixel
    read_limited = "read_limited"        # read noise > shot noise
    detector_limited = "detector_limited" # quantization/saturation
    background_limited = "background_limited"

class PhotonReport(StrictBaseModel):
    """Photon Agent output. All numbers are deterministic."""
    n_photons_per_pixel: float
    snr_db: float
    noise_regime: NoiseRegime
    shot_noise_sigma: float
    read_noise_sigma: float
    total_noise_sigma: float
    feasible: bool
    quality_tier: Literal["excellent", "acceptable", "marginal", "insufficient"]
    throughput_chain: List[Dict[str, float]]  # per-element throughput
    noise_model: Literal["poisson", "gaussian", "mixed_poisson_gaussian"]
    explanation: str = ""  # Gemini-generated narrative

class MismatchReport(StrictBaseModel):
    """Mismatch Agent output. Numbers are deterministic; family choice may be LLM-assisted."""
    modality_key: str
    mismatch_family: str              # e.g., "alignment+dispersion"
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {value, range, unit, ...}
    severity_score: float = Field(ge=0.0, le=1.0)
    correction_method: str
    expected_improvement_db: float
    explanation: str = ""

class SignalPriorClass(str, Enum):
    tv = "tv"
    wavelet_sparse = "wavelet_sparse"
    low_rank = "low_rank"
    deep_prior = "deep_prior"
    spectral_sparse = "spectral_sparse"
    joint_spatio_spectral = "joint_spatio_spectral"
    temporal_smooth = "temporal_smooth"

class RecoverabilityReport(StrictBaseModel):
    """Recoverability Agent output (formerly CompressionReport).
    Practical recoverability model, not fake formulas.
    Includes interpolation confidence and uncertainty band."""
    compression_ratio: float
    noise_regime: NoiseRegime
    signal_prior_class: SignalPriorClass
    operator_diversity_score: float = Field(ge=0.0, le=1.0,
        description="Proxy for incoherence / pattern diversity")
    condition_number_proxy: float
    recoverability_score: float = Field(ge=0.0, le=1.0,
        description="Empirical: from calibration table lookup with interpolation")
    recoverability_confidence: float = Field(ge=0.0, le=1.0,
        description="Confidence: 1.0 = exact table match, lower = interpolated")
    expected_psnr_db: float = Field(
        description="Expected PSNR from calibration table")
    expected_psnr_uncertainty_db: float = Field(ge=0.0,
        description="Uncertainty band: ± dB around expected_psnr_db")
    recommended_solver_family: str
    verdict: Literal["excellent", "sufficient", "marginal", "insufficient"]
    calibration_table_entry: Optional[Dict[str, Any]] = None
    explanation: str = ""

class BottleneckScores(StrictBaseModel):
    photon: float = Field(ge=0.0, le=1.0)
    mismatch: float = Field(ge=0.0, le=1.0)
    compression: float = Field(ge=0.0, le=1.0)
    solver: float = Field(ge=0.0, le=1.0)

class Suggestion(StrictBaseModel):
    action: str
    priority: Literal["critical", "high", "medium", "low"]
    expected_improvement_db: float
    parameter_path: Optional[str] = None  # dot-path into ExperimentSpec
    parameter_change: Optional[Dict[str, Any]] = None
    details: str

class SystemAnalysis(StrictBaseModel):
    """Analysis Agent output."""
    primary_bottleneck: str
    bottleneck_scores: BottleneckScores
    suggestions: List[Suggestion]
    overall_verdict: str
    probability_of_success: float = Field(ge=0.0, le=1.0,
        description="Pre-flight: likelihood of useful reconstruction")
    explanation: str = ""


# ── Pre-Flight Report ──

class PreFlightReport(StrictBaseModel):
    """Shown to user before heavy computation. User must approve."""
    modality: ModalitySelection
    system: ImagingSystem
    photon: PhotonReport
    mismatch: MismatchReport
    recoverability: RecoverabilityReport
    analysis: SystemAnalysis
    estimated_runtime_s: float
    proceed_recommended: bool
    warnings: List[str] = Field(default_factory=list)
    what_to_upload: Optional[List[str]] = None  # For Mode 2 fallback
```

### 2.3 Validation Rules Summary

| Rule | Enforced By |
|------|-------------|
| No extra fields accepted | `StrictBaseModel(extra="forbid")` |
| Assignment triggers revalidation | `validate_assignment=True` |
| No NaN or Inf in float fields | `StrictBaseModel._reject_nan_inf` model validator |
| Element list must include ≥1 detector/transducer | `ImagingSystem.must_have_detector` |
| Each element must declare `transfer_kind` and `noise_kinds` | `ElementSpec` required fields |
| Forward model must declare operator type | `ImagingSystem.forward_model_type` |
| All scores in [0,1] | pydantic `Field(ge=0.0, le=1.0)` |
| Modality key must exist in registry | `registry.assert_modality_exists()` **after** parsing |
| LLM-returned IDs must exist in registry | `validate_llm_selection()` hard-fail |
| RecoverabilityReport uses calibration table, not formulas | `recoverability_score` from lookup |
| Intent schema detects Mode 1 vs Mode 2 | `PlanIntent` model |

### 2.4 Build-Time Literal Types from YAML (Optional Safety Layer)

For maximum safety, generate `Literal[...]` union types from YAML keys at build time.
This catches invalid strings at type-check time, not just runtime:

```python
# scripts/generate_literals.py — run as part of CI / pre-commit
import yaml

def generate_modality_literal(yaml_path: str, output_path: str):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    keys = list(data["modalities"].keys())
    literal_str = ", ".join(f'"{k}"' for k in keys)
    code = f'from typing import Literal\nModalityKey = Literal[{literal_str}]\n'
    with open(output_path, "w") as f:
        f.write(code)

# Generated file (auto, do not edit):
# agents/_generated_literals.py
# ModalityKey = Literal["cassi", "spc", "mri", "ct", "oct", ...]
# MismatchFamilyKey = Literal["alignment+dispersion", "sos_autofocus", ...]
# SolverFamilyKey = Literal["gap_tv", "mst", "fista", "pnp_admm", ...]
# SignalPriorKey = Literal["tv", "wavelet_sparse", "low_rank", ...]
```

Usage (optional, for teams that want static type checking):
```python
class ModalitySelection(StrictBaseModel):
    modality_key: ModalityKey  # Literal type, caught by mypy/pyright
```

---

## 3. YAML Registries

### 3.1 Design Principle

**YAML is the source of truth. README is the display layer.**

All modality data lives in versioned YAML files that are loaded by a registry builder at startup. README can be auto-generated from YAML for consistency.

### 3.2 Registry Files

```
packages/pwm_core/contrib/
├── modalities.yaml           # 26 modalities: keys, descriptions, keywords, elements
├── mismatch_db.yaml          # Per-modality mismatch parameters
├── photon_db.yaml            # Per-modality photon models + throughput chains
├── compression_db.yaml       # Per-modality recoverability calibration tables
├── solver_registry.yaml      # Existing: solver tiers per modality (UPDATED)
├── metrics_db.yaml           # NEW: per-modality metric sets
└── casepacks/                # Existing: validated experiment templates
```

### 3.3 Per-Registry Pydantic Schemas (CI Validation)

Every YAML file has a corresponding Pydantic model. CI loads YAML, validates,
and cross-references:

```python
# --- agents/registry_schemas.py ---

class ElementYaml(StrictBaseModel):
    name: str
    element_type: str
    transfer_kind: str
    throughput: float = Field(ge=0.0, le=1.0)
    noise_kinds: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ModalityYaml(StrictBaseModel):
    display_name: str
    category: str
    keywords: List[str]
    description: str
    signal_dims: Dict[str, List[int]]
    forward_model_type: str
    forward_model_equation: str
    default_solver: str
    elements: List[ElementYaml]
    wavelength_range_nm: Optional[List[float]] = None

class ModalitiesFileYaml(StrictBaseModel):
    version: str
    modalities: Dict[str, ModalityYaml]

class MismatchParamYaml(StrictBaseModel):
    range: List[float]
    typical_error: float
    unit: str
    description: str = ""

class MismatchModalityYaml(StrictBaseModel):
    parameters: Dict[str, MismatchParamYaml]
    severity_weights: Dict[str, float]
    correction_method: str

class MismatchDbFileYaml(StrictBaseModel):
    version: str
    modalities: Dict[str, MismatchModalityYaml]

class CalibrationProvenance(StrictBaseModel):
    """Every calibration table entry must have provenance."""
    dataset_id: str
    seed_set: List[int]
    operator_version: str
    solver_version: str
    date_generated: str          # ISO 8601
    notes: str = ""

class CalibrationEntry(StrictBaseModel):
    cr: float
    noise: str
    solver: str
    recoverability: float = Field(ge=0.0, le=1.0)
    expected_psnr_db: float
    provenance: CalibrationProvenance

class CalibrationModalityYaml(StrictBaseModel):
    signal_prior_class: str
    entries: List[CalibrationEntry]

class CompressionDbFileYaml(StrictBaseModel):
    version: str
    calibration_tables: Dict[str, CalibrationModalityYaml]

class PhotonModelYaml(StrictBaseModel):
    model_id: str               # Code dispatches on this, NOT eval'd formula
    parameters: Dict[str, Any]
    description: str = ""

class PhotonDbFileYaml(StrictBaseModel):
    version: str
    modalities: Dict[str, PhotonModelYaml]

class MetricSetYaml(StrictBaseModel):
    description: str
    metrics: List[str]
    modalities: List[str]

class MetricsDbFileYaml(StrictBaseModel):
    version: str
    metric_sets: Dict[str, MetricSetYaml]
```

CI script:
```python
# tests/test_registry_integrity.py
def test_all_yaml_valid():
    """Load every YAML, validate against Pydantic schema, cross-reference."""
    modalities = ModalitiesFileYaml.model_validate(load_yaml("modalities.yaml"))
    mismatch = MismatchDbFileYaml.model_validate(load_yaml("mismatch_db.yaml"))
    compression = CompressionDbFileYaml.model_validate(load_yaml("compression_db.yaml"))
    photon = PhotonDbFileYaml.model_validate(load_yaml("photon_db.yaml"))
    metrics = MetricsDbFileYaml.model_validate(load_yaml("metrics_db.yaml"))

    mod_keys = set(modalities.modalities.keys())

    # Cross-reference: every key in sub-registries must exist in modalities
    for key in mismatch.modalities:
        assert key in mod_keys, f"mismatch_db has orphan key: {key}"
    for key in compression.calibration_tables:
        assert key in mod_keys, f"compression_db has orphan key: {key}"
    for key in photon.modalities:
        assert key in mod_keys, f"photon_db has orphan key: {key}"
    for ms in metrics.metric_sets.values():
        for m in ms.modalities:
            assert m in mod_keys, f"metrics_db references unknown modality: {m}"

    # Every modality must appear in at least mismatch + photon + compression
    for key in mod_keys:
        assert key in mismatch.modalities, f"{key} missing from mismatch_db"
        assert key in photon.modalities, f"{key} missing from photon_db"
        assert key in compression.calibration_tables, f"{key} missing from compression_db"
```

### 3.4 Example: modalities.yaml (excerpt)

```yaml
version: "1.0"

modalities:
  cassi:
    display_name: "CASSI (Coded Aperture Spectral Imaging)"
    category: "compressive"
    keywords:
      - hyperspectral
      - spectral imaging
      - CASSI
      - coded aperture spectral
      - snapshot spectral
    description: >
      Coded Aperture Snapshot Spectral Imaging uses a binary mask and
      dispersive element to compress a 3D hyperspectral datacube into
      a single 2D detector measurement.
    signal_dims:
      x: [256, 256, 28]
      y: [283, 256]
    forward_model_type: linear_operator
    forward_model_equation: "y = H * Phi * D * x + noise"
    default_solver: mst
    wavelength_range_nm: [450, 650]
    elements:
      - name: "Broadband Source"
        element_type: source
        transfer_kind: identity
        throughput: 1.0
      - name: "Objective Lens"
        element_type: lens
        transfer_kind: convolution
        noise_kinds: [aberration]
        throughput: 0.92
        parameters:
          na: 0.25
          focal_length_mm: 50
      - name: "Coded Aperture (DMD)"
        element_type: mask
        transfer_kind: modulation
        noise_kinds: [alignment]
        throughput: 0.50
        parameters:
          pattern: binary_random
          density: 0.5
      - name: "Dispersive Prism"
        element_type: disperser
        transfer_kind: dispersion
        noise_kinds: [alignment]
        throughput: 0.95
        parameters:
          shift_per_band_px: 1.0
      - name: "Relay Lens"
        element_type: lens
        transfer_kind: convolution
        noise_kinds: [aberration]
        throughput: 0.90
      - name: "2D Detector"
        element_type: detector
        transfer_kind: integration
        noise_kinds: [shot_poisson, read_gaussian, quantization]
        throughput: 0.80
        parameters:
          pixel_size_um: 6.5
          read_noise_e: 3.0
          qe: 0.80

  oct:
    display_name: "Optical Coherence Tomography"
    category: "medical"
    keywords: [OCT, optical coherence, retinal imaging, SD-OCT, SS-OCT]
    # ... (full definition)

  light_field:
    display_name: "Light Field Imaging"
    category: "computational_photography"
    keywords: [light field, plenoptic, Lytro, microlens array, 4D imaging]
    # ... (full definition)

  # ... all 26 modalities
```

### 3.5 Example: compression_db.yaml (calibration tables with provenance)

```yaml
version: "1.0"
description: >
  Empirical recoverability calibration tables per modality.
  Each entry is measured from synthetic benchmark runs (not from formulas).
  Every entry MUST include provenance to prevent silent drift.

calibration_tables:
  cassi:
    signal_prior_class: joint_spatio_spectral
    entries:
      - cr: 0.036   # 1/28
        noise: shot_limited
        solver: gap_tv
        recoverability: 0.55
        expected_psnr_db: 30.5
        provenance:
          dataset_id: "kaist_cave_10scenes"
          seed_set: [42, 123, 456]
          operator_version: "cassi_operator_v1.2"
          solver_version: "gap_tv_v1.0"
          date_generated: "2025-10-15"
          notes: "3 seeds, 256x256x28, binary random mask density=0.5"
      - cr: 0.036
        noise: shot_limited
        solver: mst
        recoverability: 0.82
        expected_psnr_db: 34.8
        provenance:
          dataset_id: "kaist_cave_10scenes"
          seed_set: [42, 123, 456]
          operator_version: "cassi_operator_v1.2"
          solver_version: "mst_v2.0_tsa"
          date_generated: "2025-10-15"
          notes: "MST with TSA fusion, pretrained weights"
      # ... more entries

  spc:
    signal_prior_class: wavelet_sparse
    entries:
      - cr: 0.10
        noise: shot_limited
        solver: pnp_fista
        recoverability: 0.65
        expected_psnr_db: 27.0
        provenance:
          dataset_id: "bsd68_grayscale"
          seed_set: [42, 123]
          operator_version: "spc_operator_v1.0"
          solver_version: "pnp_fista_drunet_v1.0"
          date_generated: "2025-09-20"
          notes: "Gaussian measurement matrix, 10% sampling"
      # ... more entries

  # ... all modalities
```

### 3.6 Registry Builder (with assertion helpers for LLM validation)

```python
class RegistryKeyError(PWMError):
    """Raised when a key doesn't exist in any registry."""
    pass

class RegistryBuilder:
    """Loads YAML registries, validates via Pydantic schemas, provides assertion helpers."""

    def __init__(self, contrib_dir: str):
        self.contrib_dir = Path(contrib_dir)
        # Load + validate each YAML against its Pydantic schema
        self._modalities = ModalitiesFileYaml.model_validate(
            self._load_raw("modalities.yaml"))
        self._mismatch = MismatchDbFileYaml.model_validate(
            self._load_raw("mismatch_db.yaml"))
        self._photon = PhotonDbFileYaml.model_validate(
            self._load_raw("photon_db.yaml"))
        self._compression = CompressionDbFileYaml.model_validate(
            self._load_raw("compression_db.yaml"))
        self._metrics = MetricsDbFileYaml.model_validate(
            self._load_raw("metrics_db.yaml"))
        self._validate_cross_references()

    # --- Assertion helpers (used after LLM calls) ---

    def assert_modality_exists(self, key: str):
        if key not in self._modalities.modalities:
            raise RegistryKeyError(
                f"Modality '{key}' not in registry. "
                f"Valid: {list(self._modalities.modalities.keys())}")

    def assert_mismatch_family_exists(self, modality_key: str, family_id: str):
        self.assert_modality_exists(modality_key)
        entry = self._mismatch.modalities.get(modality_key)
        if entry is None or family_id != entry.correction_method:
            raise RegistryKeyError(
                f"Mismatch family '{family_id}' not valid for modality '{modality_key}'")

    def assert_solver_exists(self, solver_id: str):
        # Check across all modality solver entries
        if not self._solver_registry_contains(solver_id):
            raise RegistryKeyError(f"Solver '{solver_id}' not in solver_registry")

    def assert_signal_prior_exists(self, prior_class: str):
        valid = {e.signal_prior_class
                 for e in self._compression.calibration_tables.values()}
        if prior_class not in valid:
            raise RegistryKeyError(
                f"Signal prior '{prior_class}' not in compression_db. Valid: {valid}")

    def all_keys(self) -> Dict[str, List[str]]:
        """All valid keys, passed to LLM so it can only select from these."""
        return {
            "modality_keys": list(self._modalities.modalities.keys()),
            "solver_ids": self._all_solver_ids(),
            "signal_prior_classes": list({
                e.signal_prior_class
                for e in self._compression.calibration_tables.values()
            }),
        }

    def get_modality(self, key: str) -> ModalityYaml:
        self.assert_modality_exists(key)
        return self._modalities.modalities[key]

    def list_modalities(self) -> List[str]:
        return list(self._modalities.modalities.keys())

    def _validate_cross_references(self):
        """Every modality key in sub-registries must exist in modalities.yaml."""
        mod_keys = set(self._modalities.modalities.keys())
        for key in self._mismatch.modalities:
            assert key in mod_keys, f"Orphan in mismatch_db: {key}"
        for key in self._compression.calibration_tables:
            assert key in mod_keys, f"Orphan in compression_db: {key}"
        for key in self._photon.modalities:
            assert key in mod_keys, f"Orphan in photon_db: {key}"
```

---

## 4. Unified Operator Interface

### 4.1 Extended PhysicsOperator Protocol

All 26 modalities implement one common interface. Extends existing `physics/base.py`:

```python
class PhysicsOperator(Protocol):
    """Unified operator interface for all 26 modalities."""

    # --- Core (existing, unchanged) ---
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...
    def set_theta(self, theta: Dict[str, Any]) -> None: ...
    def get_theta(self) -> Dict[str, Any]: ...
    def info(self) -> Dict[str, Any]: ...

    # --- New required properties ---
    @property
    def x_shape(self) -> Tuple[int, ...]: ...

    @property
    def y_shape(self) -> Tuple[int, ...]: ...

    @property
    def is_linear(self) -> bool: ...

    @property
    def supports_autodiff(self) -> bool: ...

    # --- New required methods ---
    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Full serialization for RunBundle reproducibility.

        Must include ALL params needed to rebuild forward+adjoint identically.
        For large arrays (masks, PSFs, trajectories, learned weights):
          - Save to data_dir/<name>.npy
          - Store only path + SHA256 hash in the returned dict

        Returns:
            {"operator_id": ..., "theta": ..., "x_shape": ..., "y_shape": ...,
             "blobs": [{"name": "mask", "path": "data/mask.npy",
                        "sha256": "abc123...", "shape": [256,256]}],
             "metadata": {...}}
        """
        ...

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    data_dir: Optional[str] = None) -> "PhysicsOperator":
        """Reconstruct operator from serialized data + blob files."""
        ...

    def metadata(self) -> OperatorMetadata:
        """Units, axes, wavelength info, etc."""
        ...

    def check_adjoint(self, n_trials: int = 3, tol: float = 1e-5,
                      seed: int = 0) -> "AdjointCheckReport":
        """Built-in self-test: verify <x, A^T y> == <Ax, y> for random vectors.

        Every operator gets this for free. Catches bugs early.
        Returns AdjointCheckReport with pass/fail + relative errors.
        """
        ...
```

```python
class AdjointCheckReport(StrictBaseModel):
    """Result of check_adjoint() self-test."""
    passed: bool
    n_trials: int
    max_relative_error: float
    mean_relative_error: float
    tolerance: float
    details: List[Dict[str, float]] = Field(default_factory=list)
    # Each detail: {"trial": 0, "inner_Ax_y": ..., "inner_x_ATy": ..., "rel_err": ...}

class OperatorMetadata(StrictBaseModel):
    """Rich metadata for any operator."""
    modality: str
    operator_id: str
    x_shape: List[int]
    y_shape: List[int]
    is_linear: bool
    supports_autodiff: bool
    axes: Dict[str, str]          # {"dim0": "height_px", "dim1": "width_px", "dim2": "wavelength_nm"}
    wavelength_nm: Optional[float] = None
    wavelength_range_nm: Optional[List[float]] = None
    units: Dict[str, str] = Field(default_factory=dict)  # {"x": "photons", "y": "detector_counts"}
    sampling_info: Optional[Dict[str, Any]] = None        # {"type": "cartesian", "acceleration": 4}
```

### 4.2 BaseOperator Enhancement

```python
@dataclass
class BaseOperator:
    """Enhanced convenience base class."""
    operator_id: str
    theta: Dict[str, Any]
    _x_shape: Tuple[int, ...]
    _y_shape: Tuple[int, ...]
    _is_linear: bool = True
    _supports_autodiff: bool = False

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return self._x_shape

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return self._y_shape

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def supports_autodiff(self) -> bool:
        return self._supports_autodiff

    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Serialize with blob support for large arrays."""
        import hashlib
        result = {
            "operator_id": self.operator_id,
            "theta": {},
            "x_shape": list(self._x_shape),
            "y_shape": list(self._y_shape),
            "blobs": [],
        }
        for k, v in self.theta.items():
            if isinstance(v, np.ndarray) and v.size > 1000:
                # Large array: save as blob, store path + hash
                if data_dir:
                    blob_path = os.path.join(data_dir, f"{k}.npy")
                    np.save(blob_path, v)
                    sha = hashlib.sha256(v.tobytes()).hexdigest()
                    result["blobs"].append({
                        "name": k, "path": blob_path,
                        "sha256": sha, "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    })
                else:
                    result["theta"][k] = v.tolist()  # fallback: inline
            else:
                result["theta"][k] = v
        return result

    def check_adjoint(self, n_trials: int = 3, tol: float = 1e-5,
                      seed: int = 0) -> AdjointCheckReport:
        """Built-in adjoint consistency test."""
        rng = np.random.default_rng(seed)
        details = []
        max_err = 0.0
        for trial in range(n_trials):
            x = rng.standard_normal(self._x_shape).astype(np.float64)
            y = rng.standard_normal(self._y_shape).astype(np.float64)
            Ax = self.forward(x)
            ATy = self.adjoint(y)
            inner_Ax_y = float(np.sum(Ax.ravel() * y.ravel()))
            inner_x_ATy = float(np.sum(x.ravel() * ATy.ravel()))
            denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
            rel_err = abs(inner_Ax_y - inner_x_ATy) / denom
            max_err = max(max_err, rel_err)
            details.append({"trial": trial, "inner_Ax_y": inner_Ax_y,
                            "inner_x_ATy": inner_x_ATy, "rel_err": rel_err})
        return AdjointCheckReport(
            passed=max_err < tol, n_trials=n_trials,
            max_relative_error=max_err,
            mean_relative_error=float(np.mean([d["rel_err"] for d in details])),
            tolerance=tol, details=details,
        )

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality=self.operator_id.split("_")[0],
            operator_id=self.operator_id,
            x_shape=list(self._x_shape),
            y_shape=list(self._y_shape),
            is_linear=self._is_linear,
            supports_autodiff=self._supports_autodiff,
            axes={},
        )
```

Each new modality operator (light_field, dot, photoacoustic, oct, flim, phase_retrieval, integral, fpm) implements this interface.

---

## 5. Imaging Modality Database (26 Modalities)

All 26 modalities are registered in the YAML databases (`modalities.yaml`, `solver_registry.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `compression_db.yaml`, `metrics_db.yaml`).

| # | Modality | Key | Category | Forward Model | Default Solver | PSNR (dB) |
|---|---|---|---|---|---|---|
| 1 | Widefield Microscopy | `widefield` | microscopy | PSF convolution | Richardson-Lucy | 27.31 |
| 2 | Widefield Low-Dose | `widefield_lowdose` | microscopy | PSF convolution + Poisson noise | PnP | 27.78 |
| 3 | Confocal Live-Cell | `confocal_livecell` | microscopy | Pinhole-filtered PSF | Richardson-Lucy | 26.27 |
| 4 | Confocal 3D | `confocal_3d` | microscopy | 3D PSF convolution | 3D Richardson-Lucy | 29.01 |
| 5 | SIM | `sim` | microscopy | Structured illumination frequency mixing | Wiener-SIM | 27.48 |
| 6 | CASSI | `cassi` | compressive | y = Σ_l M(x,y)·X(x, y-s(l), l) | MST / GAP-TV | 34.81 |
| 7 | SPC | `spc` | compressive | y = Φ·x (random sensing matrix) | PnP-FISTA + DRUNet | 30.90 |
| 8 | CACTI | `cacti` | compressive | y = Σ_t mask_t · frame_t | GAP-TV | 32.79 |
| 9 | Lensless | `lensless` | computational | y = PSF * x (mask-modulated) | ADMM-TV | 34.66 |
| 10 | Light-Sheet | `lightsheet` | microscopy | Sheet illumination + stripe artifacts | Stripe Removal | 28.05 |
| 11 | CT | `ct` | medical | y = Radon transform (sinogram) | PnP-SART + DRUNet | 27.97 |
| 12 | MRI | `mri` | medical | y = F_u · S · x (undersampled k-space) | PnP-ADMM + DRUNet | 48.25 |
| 13 | Ptychography | `ptychography` | coherent | y_j = \|F{P_j · x}\|^2 (overlapping probes) | ePIE / PtychoNN | 59.47 |
| 14 | Holography | `holography` | coherent | y = \|E_ref + E_obj\|^2 (interference) | Angular Spectrum | 42.52 |
| 15 | NeRF | `nerf` | neural_rendering | Volume rendering integral along rays | SIREN | 61.35 |
| 16 | 3D Gaussian Splatting | `gaussian_splatting` | neural_rendering | Splatting 3D Gaussians to 2D | 2D Gaussian Opt | 30.47 |
| 17 | Matrix (Generic) | `matrix` | compressive | y = A·x (generic linear) | FISTA-TV | 25.79 |
| 18 | Panorama Multifocal | `panorama` | computational | Multi-focus image fusion | Neural Fusion | 27.78 |
| 19 | Light Field | `light_field` | computational | L(x,y,u,v) → 2D via microlens array | Shift-and-sum | 32.0 |
| 20 | Diffuse Optical Tomography | `dot` | medical | y = J(μ_a, μ_s')·x (diffusion Jacobian) | L-BFGS+TV | 28.0 |
| 21 | Photoacoustic Imaging | `photoacoustic` | medical | y = R·Γ·μ_a·Φ (acoustic propagation) | Back-projection | 30.0 |
| 22 | OCT | `oct` | medical | y(k) = \|E_r + E_s(k)\|^2 → FFT depth | FFT recon | 35.0 |
| 23 | FLIM | `flim` | microscopy | y(t) = IRF * Σ a_i·exp(-t/τ_i) | Phasor | 28.0 |
| 24 | Phase Retrieval (CDI) | `phase_retrieval` | coherent | y = \|F{x}\|^2 (Fourier magnitude only) | HIO | 32.0 |
| 25 | Integral Photography | `integral` | computational | I = ∫ L(x,y,u,v)·T(u,v) dudv | Depth estimation | 30.0 |
| 26 | Fourier Ptychography | `fpm` | microscopy | y_j = \|F^{-1}{P(k-k_j)·O(k)}\|^2 | Seq. phase retrieval | 35.0 |

### Category Breakdown

| Category | Count | Modalities |
|----------|-------|------------|
| Microscopy | 7 | widefield, widefield_lowdose, confocal_livecell, confocal_3d, sim, lightsheet, flim, fpm |
| Compressive | 4 | cassi, spc, cacti, matrix |
| Medical | 4 | ct, mri, dot, photoacoustic, oct |
| Coherent | 3 | ptychography, holography, phase_retrieval |
| Computational | 3 | lensless, panorama, light_field, integral |
| Neural Rendering | 2 | nerf, gaussian_splatting |

### Registry Coverage

| Registry | Entries | Purpose |
|----------|---------|---------|
| `modalities.yaml` | 26 | Element chains, signal dims, forward models, upload templates |
| `solver_registry.yaml` | 26 | Tiered solver mappings (traditional_cpu, best_quality, famous_dl, etc.) |
| `mismatch_db.yaml` | 26 | Mismatch parameters, severity weights, correction methods |
| `photon_db.yaml` | 26 | Photon models + noise parameters per modality |
| `compression_db.yaml` | 26 | Calibration tables with full provenance |
| `metrics_db.yaml` | 9 sets | Per-modality metric sets (PSNR, SSIM, SAM, LPIPS, phase_error, etc.) |

---

## 6. Plan Agent (Orchestrator)

### 6.1 Responsibilities

| What it does | How |
|---|---|
| Parse user prompt → `PlanIntent` | Gemini semantic parsing |
| Map prompt → modality | Keyword fast-path + Gemini fallback |
| Propose imaging system design | Gemini proposes from `modalities.yaml` templates, code validates |
| Choose parameter priors | Gemini selects from registry, code fills values |
| Dispatch to sub-agents | Code orchestration |
| Construct `ExperimentSpec` | Deterministic code |
| Generate pre-flight report | Code computes, Gemini narrates |
| Reject non-imaging prompts | Keyword + Gemini classification |

### 6.2 Modality Mapping (Two-Stage)

```python
class PlanAgent(BaseAgent):
    def map_prompt_to_modality(self, prompt: str) -> ModalitySelection:
        # Stage 1: Deterministic keyword matching (fast, no API call)
        for key, spec in self.registry.items():
            for kw in spec.keywords:
                if kw.lower() in prompt.lower():
                    return ModalitySelection(
                        modality_key=key, confidence=0.95,
                        reasoning=f"Keyword match: '{kw}'")

        # Stage 2: Gemini semantic matching (only if no keyword hit)
        if self.gemini:
            result = self.gemini.select_modality(
                prompt=prompt,
                available_modalities=self.registry.list_modalities()
            )
            return ModalitySelection(**result)

        raise PWMError("Could not map prompt to any modality")
```

### 6.3 Non-Imaging Prompt Handling

```python
def handle_non_imaging_prompt(self, prompt: str) -> dict:
    return {
        "status": "not_imaging",
        "message": (
            "Your prompt doesn't appear to be related to computational imaging. "
            "PWM supports 26 imaging modalities. "
            "Please describe an imaging scenario."
        ),
        "supported_categories": [
            "microscopy (widefield, confocal, SIM, light-sheet)",
            "compressive (CASSI, SPC, CACTI)",
            "medical (CT, MRI, OCT, photoacoustic, DOT)",
            "coherent (ptychography, holography, phase retrieval, FPM)",
            "neural rendering (NeRF, 3DGS)",
            "other (light field, FLIM, lensless, panorama, integral)",
        ],
        "suggested_prompts": [
            "Simulate a CASSI system for hyperspectral imaging of tissue",
            "Design a low-dose CT reconstruction pipeline",
            "Set up a light field camera for depth estimation",
            "Reconstruct an MRI scan with 4x acceleration",
        ],
    }
```

---

## 7. Photon Agent

### 7.1 Division of Labor

| Deterministic (Code) | LLM-Assisted (Gemini) |
|---|---|
| N_photons = source_power * throughput_chain * QE * exposure | Choose which throughput model to use |
| SNR = N / sqrt(N + read^2 + dark*t) | Generate explanation paragraph |
| Regime classification thresholds | Suggest practical improvements |
| Noise model selection (Poisson/Gaussian/mixed) | — |

### 7.2 Deterministic Computation

```python
class PhotonAgent(BaseAgent):
    def run(self, context: AgentContext) -> PhotonReport:
        system = context.imaging_system
        budget = context.budget

        # 1. Walk the element chain, compute cumulative throughput
        throughput_chain = []
        cumulative_throughput = 1.0
        for elem in system.elements:
            cumulative_throughput *= elem.throughput
            throughput_chain.append({
                "element": elem.name,
                "throughput": elem.throughput,
                "cumulative": cumulative_throughput,
            })

        # 2. Compute photons at detector (deterministic formula from photon_db.yaml)
        model = self.registry.get_photon_model(system.modality_key)
        n_photons = model.compute_detector_photons(budget, cumulative_throughput)

        # 3. Compute SNR (exact formula, no hallucination)
        detector = self._find_detector(system.elements)
        read_noise = detector.parameters.get("read_noise_e", 3.0)
        dark_current = detector.parameters.get("dark_current_e_per_s", 0.01)
        exposure = budget.get("exposure_s", 1.0)

        shot_noise = np.sqrt(max(n_photons, 0))
        total_noise = np.sqrt(
            max(n_photons, 0) + read_noise**2 + dark_current * exposure
        )
        snr = n_photons / max(total_noise, 1e-10)
        snr_db = float(20 * np.log10(max(snr, 1e-10)))

        # 4. Classify regime (deterministic thresholds)
        regime = self._classify_regime(n_photons, read_noise, dark_current, exposure)

        # 5. Feasibility verdict
        quality_tier = self._quality_tier(snr_db)

        # 6. Noise model (variance dominance, not threshold-only)
        noise_model = self._classify_noise_model(
            n_photons, read_noise, dark_current, exposure)

        report = PhotonReport(
            n_photons_per_pixel=n_photons,
            snr_db=snr_db,
            noise_regime=regime,
            shot_noise_sigma=shot_noise,
            read_noise_sigma=read_noise,
            total_noise_sigma=total_noise,
            feasible=quality_tier != "insufficient",
            quality_tier=quality_tier,
            throughput_chain=throughput_chain,
            noise_model=noise_model,
        )

        # 7. Optional: LLM narrative
        if self.llm:
            report.explanation = self.llm.narrate(report.model_dump())

        return report

    @staticmethod
    def _classify_noise_model(n_photons: float, read_noise: float,
                               dark_current: float, exposure: float) -> str:
        """Determine noise regime by comparing variance contributions.

        shot_variance = n_photons  (Poisson: variance == mean)
        read_variance = read_noise^2
        dark_variance = dark_current * exposure

        Dominant source determines model label.
        """
        shot_var = max(n_photons, 0)
        read_var = read_noise ** 2
        dark_var = dark_current * exposure
        total_var = shot_var + read_var + dark_var

        if total_var < 1e-30:
            return "mixed_poisson_gaussian"

        shot_frac = shot_var / total_var
        read_frac = read_var / total_var

        if shot_frac > 0.9:
            return "poisson"
        elif read_frac > 0.5:
            return "gaussian"
        else:
            return "mixed_poisson_gaussian"

```

### 7.3 Photon Models (YAML stores model_id, code computes)

**Important:** YAML stores `model_id` + parameters, NOT formula strings.
Executing formulas from YAML (eval-style) is a security and debugging footgun.
Instead, code implements `PhotonModel.compute()` per `model_id`:

```python
class PhotonModelRegistry:
    """Code-side implementation of per-modality photon models.
    YAML stores model_id + parameters. Code dispatches on model_id."""

    MODELS = {
        "microscopy_fluorescence": _microscopy_fluorescence,
        "ct_xray": _ct_xray,
        "mri_thermal": _mri_thermal,
        "photoacoustic_optical": _photoacoustic_optical,
        "oct_interferometric": _oct_interferometric,
        "generic_detector": _generic_detector,
    }

    def compute(self, model_id: str, params: Dict) -> float:
        if model_id not in self.MODELS:
            raise RegistryKeyError(f"Unknown photon model: {model_id}")
        return self.MODELS[model_id](params)

def _microscopy_fluorescence(p: Dict) -> float:
    """N = P * QE * (NA/n)^2/(4pi) * t / (h*v)"""
    return p["power_w"] * p["qe"] * (p["na"] / p["n_medium"])**2 / (4*np.pi) \
           * p["exposure_s"] / (6.626e-34 * 3e8 / (p["wavelength_nm"] * 1e-9))

def _ct_xray(p: Dict) -> float:
    """N = I0 * exp(-mu*L) * eta_det"""
    return p["tube_current_photons"] * np.exp(-p["mu"] * p["L"]) * p["eta_det"]

# ... one function per model_id
```

YAML reference (photon_db.yaml):
```yaml
modalities:
  cassi:
    model_id: "microscopy_fluorescence"   # dispatches to code function
    parameters:
      power_w: 0.001
      wavelength_nm: 550
      na: 0.25
      n_medium: 1.0
      qe: 0.80
    description: "Fluorescence model for CASSI broadband source"
```

Per-modality formula reference (documentation only, NOT executed):

| Modality | model_id | Formula (docs) | Key Parameters |
|---|---|---|---|
| Microscopy | `microscopy_fluorescence` | N = P * QE * (NA/n)^2/(4pi) * t / (hv) | Power, NA, magnification |
| CT | `ct_xray` | N = I0 * exp(-mu*L) * eta_det | mAs, kVp, filtration |
| MRI | `mri_thermal` | SNR ~ B0 * sqrt(Voxel * N_avg / BW) | B0, voxel, bandwidth |
| Photoacoustic | `photoacoustic_optical` | N_abs * Gamma * acoustic_coupling | Fluence, mu_a |
| OCT | `oct_interferometric` | N = P * sensitivity * t | Power, sensitivity |

---

## 8. Mismatch Agent

### 8.1 Division of Labor

| Deterministic (Code) | LLM-Assisted (Gemini) |
|---|---|
| Operator distance computation | Choose which mismatch family to activate |
| Residual structure analysis (autocorrelation, Fourier) | Explain why mismatch matters |
| Severity scoring (weighted sum from mismatch_db.yaml) | Suggest calibration priority |
| Mismatch operator construction | — |

### 8.2 Mismatch Database (mismatch_db.yaml)

Each modality defines its mismatch parameters, severity weights, and correction methods. This is loaded from `mismatch_db.yaml` (source of truth), not from a Python dict.

**Key entries (all 26 modalities have definitions):**

```yaml
# mismatch_db.yaml (excerpt)
version: "1.0"

modalities:
  cassi:
    parameters:
      mask_dx:
        range: [-3.0, 3.0]
        typical_error: 0.5
        unit: pixels
        description: "Mask x-shift"
      mask_dy:
        range: [-3.0, 3.0]
        typical_error: 0.5
        unit: pixels
      dispersion_step:
        range: [-0.5, 0.5]
        typical_error: 0.1
        unit: pixels_per_band
      psf_sigma:
        range: [0.5, 3.0]
        typical_error: 0.2
        unit: pixels
      gain:
        range: [0.5, 1.5]
        typical_error: 0.1
        unit: ratio
    severity_weights:
      mask_dx: 0.25
      mask_dy: 0.25
      dispersion_step: 0.30
      psf_sigma: 0.10
      gain: 0.10
    correction_method: UPWMI_beam_search

  photoacoustic:
    parameters:
      speed_of_sound:
        range: [1400, 1600]
        typical_error: 20
        unit: m/s
      transducer_position:
        range: [-1.0, 1.0]
        typical_error: 0.3
        unit: mm
      acoustic_attenuation:
        range: [0, 1.0]
        typical_error: 0.2
        unit: dB/cm/MHz
    severity_weights:
      speed_of_sound: 0.50
      transducer_position: 0.30
      acoustic_attenuation: 0.20
    correction_method: sos_autofocus

  # ... all 26 modalities
```

### 8.3 Shared Spectral Profile

Per feedback item on "Physical Continuity": the Photon Agent and Mismatch Agent share a **Spectral Profile** so that chromatic aberration data matches the chosen wavelength:

```python
class SpectralProfile(BaseModel):
    """Shared between Photon and Mismatch agents for consistency."""
    wavelength_nm: Optional[float] = None
    wavelength_range_nm: Optional[List[float]] = None
    spectral_power_distribution: Optional[List[float]] = None
```

---

## 9. Recoverability Agent (Practical Recoverability Model)

### 9.1 Why the Old Math Was Wrong

Plan v1 used: `min_cr_exact = 2 * sparsity * log(n / sparsity)`. This mixes k-sparsity with sparsity ratio and gives unstable guidance. **Replaced with empirical calibration tables.**

### 9.2 Recoverability Model (4-Factor Stack + Interpolation)

Instead of one formula, the Recoverability Agent uses a **pragmatic scoring stack**
with table interpolation and confidence reporting:

```python
class RecoverabilityAgent(BaseAgent):
    def run(self, context: AgentContext) -> RecoverabilityReport:
        system = context.imaging_system
        modality = system.modality_key

        # Factor 1: Signal prior class (from registry)
        prior_class = self.registry.get_signal_prior(modality)

        # Factor 2: Operator quality (deterministic computation)
        diversity_score = self._compute_operator_diversity(system)
        condition_proxy = self._compute_condition_proxy(system)

        # Factor 3: Noise level (from PhotonReport, already computed)
        noise_regime = context.photon_report.noise_regime

        # Factor 4: Empirical lookup with INTERPOLATION (from compression_db.yaml)
        cr = self._compute_compression_ratio(system)
        lookup = self._interpolated_lookup(
            modality, cr, noise_regime, context.solver_family
        )

        recoverability = lookup.recoverability
        expected_psnr = lookup.expected_psnr_db
        confidence = lookup.confidence        # 1.0 = exact match, lower = interpolated
        uncertainty = lookup.uncertainty_db    # ± dB band

        # Verdict (deterministic thresholds)
        if recoverability >= 0.85:
            verdict = "excellent"
        elif recoverability >= 0.60:
            verdict = "sufficient"
        elif recoverability >= 0.35:
            verdict = "marginal"
        else:
            verdict = "insufficient"

        # Recommended solver family
        best_solver = self._find_best_solver_for_cr(modality, cr, noise_regime)

        return RecoverabilityReport(
            compression_ratio=cr,
            noise_regime=noise_regime,
            signal_prior_class=prior_class,
            operator_diversity_score=diversity_score,
            condition_number_proxy=condition_proxy,
            recoverability_score=recoverability,
            recoverability_confidence=confidence,
            expected_psnr_db=expected_psnr,
            expected_psnr_uncertainty_db=uncertainty,
            recommended_solver_family=best_solver,
            verdict=verdict,
            calibration_table_entry=lookup.raw_entry,
        )

    def _interpolated_lookup(self, modality, cr, noise_regime, solver_family):
        """Nearest-neighbor + linear interpolation between table entries.

        Returns InterpolatedResult with confidence based on distance
        from nearest table points.
        """
        entries = self.registry.get_calibration_entries(modality)
        # Filter by noise regime + solver
        matched = [e for e in entries
                   if e.noise == noise_regime and e.solver == solver_family]
        if not matched:
            # Fallback: match only by noise regime, any solver
            matched = [e for e in entries if e.noise == noise_regime]
        if not matched:
            matched = entries  # Use all entries as last resort

        # Sort by CR distance
        matched.sort(key=lambda e: abs(e.cr - cr))

        if len(matched) == 1 or abs(matched[0].cr - cr) < 1e-6:
            # Exact or near-exact match
            e = matched[0]
            return InterpolatedResult(
                recoverability=e.recoverability,
                expected_psnr_db=e.expected_psnr_db,
                confidence=1.0 if abs(e.cr - cr) < 1e-6 else 0.8,
                uncertainty_db=0.5,
                raw_entry=e.model_dump(),
            )

        # Linear interpolation between two nearest
        e1, e2 = matched[0], matched[1]
        if abs(e2.cr - e1.cr) < 1e-10:
            alpha = 0.5
        else:
            alpha = (cr - e1.cr) / (e2.cr - e1.cr)
            alpha = np.clip(alpha, 0, 1)

        rec = e1.recoverability * (1 - alpha) + e2.recoverability * alpha
        psnr = e1.expected_psnr_db * (1 - alpha) + e2.expected_psnr_db * alpha
        dist = min(abs(cr - e1.cr), abs(cr - e2.cr))
        conf = float(np.clip(1.0 - dist * 5.0, 0.3, 1.0))

        return InterpolatedResult(
            recoverability=float(rec),
            expected_psnr_db=float(psnr),
            confidence=conf,
            uncertainty_db=float(abs(e2.expected_psnr_db - e1.expected_psnr_db) * 0.3),
            raw_entry={"interpolated_from": [e1.model_dump(), e2.model_dump()]},
        )
```

### 9.3 Operator Diversity Score (Deterministic)

```python
def _compute_operator_diversity(self, system: ImagingSystem) -> float:
    """Proxy for measurement incoherence / pattern diversity.
    Based on modality-specific heuristics, not LLM."""
    modality = system.modality_key

    if modality == "spc":
        pattern_type = system.elements_by_type("mask")[0].parameters.get("pattern", "gaussian")
        if pattern_type == "hadamard":
            return 0.95  # Orthogonal, best incoherence
        elif pattern_type == "gaussian":
            return 0.90  # Near-optimal RIP
        else:
            return 0.70  # Binary, suboptimal

    elif modality == "cassi":
        mask_density = system.elements_by_type("mask")[0].parameters.get("density", 0.5)
        return float(np.clip(4 * mask_density * (1 - mask_density), 0, 1))

    elif modality == "mri":
        acceleration = system.signal_dims.get("acceleration", 4)
        return float(np.clip(1.0 / acceleration * 4, 0.2, 1.0))

    # Default: moderate diversity
    return 0.70
```

---

## 10. Analysis Agent + Self-Improvement Loop

### 10.1 Bottleneck Scoring (Deterministic)

```python
class AnalysisAgent(BaseAgent):
    def run(self, context: AgentContext) -> SystemAnalysis:
        photon = context.photon_report
        mismatch = context.mismatch_report
        recoverability = context.recoverability_report

        # Deterministic scoring
        scores = BottleneckScores(
            photon=self._photon_score(photon),
            mismatch=self._mismatch_score(mismatch),
            compression=self._recoverability_score(recoverability),
            solver=self._solver_score(context.recon_result),
        )

        primary = max(scores.dict(), key=scores.dict().get)

        # Probability of success (pre-flight)
        p_success = self._probability_of_success(photon, recoverability, mismatch)

        suggestions = self._generate_suggestions(scores, photon, mismatch, recoverability)

        return SystemAnalysis(
            primary_bottleneck=primary,
            bottleneck_scores=scores,
            suggestions=suggestions,
            overall_verdict=self._verdict(scores),
            probability_of_success=p_success,
        )
```

### 10.2 Self-Improvement Loop (Optional)

When the bottleneck is photon or compression, the system can automatically propose 2-3 parameter changes, run quick low-res proxy simulations, and show predicted improvement tradeoffs:

```python
class SelfImprovementLoop:
    """Optional: propose parameter changes, run quick proxy, show tradeoffs."""

    MAX_ITERATIONS = 3
    PROXY_RESOLUTION_FACTOR = 0.25  # 4x downscale for speed

    def run(self, analysis: SystemAnalysis, context: AgentContext) -> List[DesignAlternative]:
        alternatives = []

        if analysis.bottleneck_scores.photon > 0.4:
            # Propose: 2x exposure, 4x exposure, higher NA
            for factor in [2.0, 4.0]:
                alt_budget = context.budget.copy()
                alt_budget["exposure_s"] *= factor
                proxy_result = self._run_proxy(context, alt_budget)
                alternatives.append(DesignAlternative(
                    change=f"Exposure x{factor}",
                    parameter_path="states.budget.photon_budget.exposure_time",
                    new_value=alt_budget["exposure_s"],
                    predicted_psnr_improvement_db=proxy_result.psnr_delta,
                    cost="acquisition_time",
                ))

        if analysis.bottleneck_scores.compression > 0.4:
            for cr_boost in [1.5, 2.0]:
                # e.g., for SPC: increase sampling_rate; for MRI: reduce acceleration
                alt = self._propose_cr_increase(context, cr_boost)
                proxy_result = self._run_proxy(context, alt.budget)
                alternatives.append(alt)

        return alternatives

    def _run_proxy(self, context, modified_budget) -> ProxyResult:
        """Low-res, fast reconstruction to predict improvement."""
        # Downsample by PROXY_RESOLUTION_FACTOR
        # Run lightweight solver (e.g., FISTA with few iterations)
        # Return delta-PSNR estimate
        ...
```

---

## 11. Physical Continuity Check & Agent Negotiation

### 11.1 Conservation & Constraint Layer (Deterministic)

Before the pre-flight report, a deterministic check ensures the system is physically valid:

```python
class PhysicalContinuityChecker:
    """Deterministic checks for physical validity of the imaging system."""

    def check(self, system: ImagingSystem, photon: PhotonReport,
              mismatch: MismatchReport) -> List[str]:
        warnings = []

        # 1. Aperture matching: NA and f/# between relay lenses and sensors
        lenses = [e for e in system.elements if e.element_type == "lens"]
        detectors = [e for e in system.elements if e.element_type == "detector"]
        for lens in lenses:
            for det in detectors:
                if not self._na_compatible(lens, det):
                    warnings.append(
                        f"NA mismatch: {lens.name} (NA={lens.parameters.get('na')}) "
                        f"may vignette signal at {det.name}")

        # 2. Spectral consistency: wavelength chosen must match chromatic data
        if system.wavelength_nm:
            for elem in system.elements:
                if elem.transfer_kind == TransferKind.dispersion:
                    if not self._dispersion_valid_for_wavelength(elem, system.wavelength_nm):
                        warnings.append(
                            f"Dispersion model in {elem.name} not calibrated "
                            f"for wavelength {system.wavelength_nm} nm")

        # 3. Dimension chain: x_shape → forward → y_shape must be consistent
        if not self._dimension_chain_valid(system):
            warnings.append("Signal dimensions through element chain are inconsistent")

        # 4. Throughput chain sanity
        total_throughput = 1.0
        for elem in system.elements:
            total_throughput *= elem.throughput
        if total_throughput < 0.01:
            warnings.append(f"Total throughput is very low ({total_throughput:.4f})")

        return warnings
```

### 11.2 Agent Veto & Negotiation Loop

The Recoverability Agent and Photon Agent can conflict (high compression requires high SNR). A negotiation loop resolves this:

```python
class AgentNegotiator:
    """Resolves conflicts between agent reports before proceeding."""

    def negotiate(self, photon: PhotonReport, recoverability: RecoverabilityReport,
                  mismatch: MismatchReport) -> NegotiationResult:
        vetoes = []

        # Conflict: low photons + high compression → unrecoverable
        if (photon.quality_tier in ("marginal", "insufficient") and
                recoverability.verdict in ("marginal", "insufficient")):
            vetoes.append(VetoReason(
                source="photon+compressed",
                reason="Low SNR combined with high compression makes recovery unlikely",
                suggested_resolution=[
                    "Increase exposure time or source power",
                    "Reduce compression ratio (more measurements)",
                    "Use advanced deep-learning solver (if available)",
                ],
            ))

        # Conflict: severe mismatch + no correction planned
        if mismatch.severity_score > 0.7:
            vetoes.append(VetoReason(
                source="mismatch",
                reason=f"Mismatch severity {mismatch.severity_score:.2f} is high",
                suggested_resolution=[
                    "Enable operator correction (UPWMI)",
                    "Re-calibrate the system",
                ],
            ))

        return NegotiationResult(
            vetoes=vetoes,
            proceed=len(vetoes) == 0,
            probability_of_success=self._compute_joint_probability(
                photon, compression, mismatch),
        )
```

---

## 12. Pre-Flight Report & Permit Step

Before any heavy reconstruction runs, the system shows the user a summary and asks for approval.

### 12.1 Pre-Flight Report Generation

```python
class PreFlightReportBuilder:
    def build(self, modality: ModalitySelection, system: ImagingSystem,
              photon: PhotonReport, mismatch: MismatchReport,
              recoverability: RecoverabilityReport, analysis: SystemAnalysis,
              negotiation: NegotiationResult) -> PreFlightReport:

        warnings = []
        for veto in negotiation.vetoes:
            warnings.append(f"[{veto.source}] {veto.reason}")

        what_to_upload = None
        if (analysis.primary_bottleneck == "mismatch" and
                not context.plan_intent.has_measured_y):
            what_to_upload = [
                "Measured data y (numpy .npy/.npz or .mat)",
                "Forward operator A (if available)",
                "Calibration metadata (mask pattern, PSF, etc.)",
            ]

        return PreFlightReport(
            modality=modality,
            system=system,
            photon=photon,
            mismatch=mismatch,
            compression=compression,
            analysis=analysis,
            estimated_runtime_s=self._estimate_runtime(system, analysis),
            proceed_recommended=negotiation.proceed,
            warnings=warnings,
            what_to_upload=what_to_upload,
        )
```

### 12.2 User-Facing Pre-Flight Summary

```
╔══════════════════════════════════════════════════════════╗
║  PWM Pre-Flight Report                                  ║
╠══════════════════════════════════════════════════════════╣
║  Modality:    CASSI (confidence: 0.99)                  ║
║  System:      6 elements (objective → mask → prism →    ║
║               relay → detector)                         ║
║                                                         ║
║  Photon:      SNR 14.1 dB (marginal, photon-starved)    ║
║  Mismatch:    Severity 0.35 (moderate, mask+dispersion)  ║
║  Compression: CR 3.6%, recoverability 0.60 (marginal)   ║
║                                                         ║
║  Primary bottleneck: PHOTON                             ║
║  Probability of success: 0.58                           ║
║  Expected PSNR: ~28 dB with MST solver                  ║
║  Estimated runtime: 45 seconds                          ║
║                                                         ║
║  ⚠ Warning: Low SNR + high compression.                 ║
║    Consider increasing exposure or reducing CR.          ║
║                                                         ║
║  [Proceed]  [Modify Parameters]  [Cancel]               ║
╚══════════════════════════════════════════════════════════╝
```

### 12.3 Permit Step (CLI + Programmatic Modes)

The user must explicitly approve. If they choose "Modify Parameters," the Plan Agent
re-runs with updated budget/system design. **No heavy computation runs without user consent.**

For batch runs / CI, the permit step supports programmatic overrides:

```python
class PermitMode(str, Enum):
    interactive = "interactive"   # Default: show report, wait for user
    auto_proceed = "auto_proceed" # Only proceed if proceed_recommended is True
    force = "force"               # Always proceed (logs a big warning)

# CLI flags:
# pwm run spec.yaml                           → interactive (default)
# pwm run spec.yaml --auto-proceed            → auto if recommended
# pwm run spec.yaml --force                   → always proceed (WARNING logged)

def check_permit(report: PreFlightReport, mode: PermitMode) -> bool:
    if mode == PermitMode.interactive:
        return _show_report_and_ask(report)
    elif mode == PermitMode.auto_proceed:
        if report.proceed_recommended:
            logger.info("Pre-flight OK, auto-proceeding")
            return True
        else:
            logger.warning(
                "Pre-flight NOT recommended. Use --force to override. "
                f"Warnings: {report.warnings}"
            )
            return False
    elif mode == PermitMode.force:
        logger.warning(
            "⚠ FORCE MODE: proceeding despite pre-flight warnings. "
            f"Warnings: {report.warnings}"
        )
        return True
```

### 12.4 Modality-Aware Upload Templates

Instead of a generic "what to upload" list, provide a structured template per modality.
Loaded from `modalities.yaml`:

```yaml
# In modalities.yaml, per modality:
cassi:
  upload_template:
    required:
      - {name: "y.npy", description: "2D detector measurement", shape: "[H+S-1, W]"}
      - {name: "mask.npy", description: "Binary coded aperture pattern", shape: "[H, W]"}
    required_params:
      - {name: "dispersion_step", type: "float", description: "Pixels per spectral band"}
    optional:
      - {name: "psf_sigma", type: "float", description: "PSF blur sigma in pixels"}
      - {name: "wavelength_range", type: "[float, float]", description: "nm range"}

oct:
  upload_template:
    required:
      - {name: "interferogram.npy", description: "Raw spectral interferogram", shape: "[N_axial, N_lateral]"}
    required_params:
      - {name: "center_wavelength_nm", type: "float"}
      - {name: "bandwidth_nm", type: "float"}
    optional:
      - {name: "reference_arm.npy", description: "Reference arm spectrum"}
```

Pre-flight fallback display:
```
To use operator correction for CASSI, please provide:
  Required files:
    ☐ y.npy — 2D detector measurement [H+S-1, W]
    ☐ mask.npy — Binary coded aperture pattern [H, W]
  Required parameters:
    ☐ dispersion_step (float) — Pixels per spectral band
  Optional:
    ☐ psf_sigma (float) — PSF blur sigma in pixels
    ☐ wavelength_range ([float, float]) — nm range
```

---

## 13. Operating Modes & Intent Detection

### 13.1 Intent Schema (Explicit)

```python
class PlanIntent(BaseModel):
    mode_requested: ModeRequested    # simulate | operator_correction | auto
    has_measured_y: bool
    has_operator_A: bool
    operator_type: OperatorType      # explicit_matrix | linear_operator | nonlinear_operator | unknown
    user_prompt: str
    raw_file_paths: List[str]
```

### 13.2 Mode Resolution Logic

```python
def resolve_mode(intent: PlanIntent) -> str:
    if intent.mode_requested == ModeRequested.operator_correction:
        if intent.has_measured_y and intent.has_operator_A:
            return "mode2_operator_correction"
        elif intent.has_measured_y:
            return "mode2_operator_correction"  # PWM will build A from modality
        else:
            # Fallback with clear explanation
            return "mode1_simulate_fallback"
    elif intent.mode_requested == ModeRequested.simulate:
        return "mode1_simulate"
    else:  # auto
        if intent.has_measured_y:
            return "mode2_operator_correction"
        return "mode1_simulate"
```

**Fallback behavior:** If user requests operator correction but has no (y, A):
```
"No measured data provided. Falling back to simulation mode.
To use operator correction mode, please provide:
  ☐ Measured data y (.npy, .npz, .mat, .tif, .h5)
  ☐ Forward operator A (optional: matrix or operator ID)
  ☐ Calibration metadata (optional: mask pattern, PSF)"
```

### 13.3 Mode 1: Prompt-Driven Simulation + Reconstruction

```
User Prompt → Plan Agent (intent + modality + system design + priors)
           → [Photon | Mismatch | Recoverability] agents (parallel, deterministic)
           → Physical Continuity Check
           → Agent Negotiation
           → Pre-Flight Report → USER PERMIT
           → Build A_ideal + A_real (with mismatch + noise from agents)
           → Simulate: y_ideal, y_real
           → Reconstruct: x_hat_ideal, x_hat_real, x_hat_corrected
           → Per-modality metrics
           → Analysis Agent + optional Self-Improvement Loop
           → RunBundle Export
```

### 13.4 Mode 2: Operator Correction (measured y + A)

```
User provides: y, A (or operator ID)
           → Plan Agent (detect modality, intent)
           → Photon Agent (estimate noise from y statistics — deterministic)
           → Mismatch Agent (quantify mismatch from residuals — deterministic)
           → Recoverability Agent (analyze CR from A dimensions — deterministic)
           → Pre-Flight Report → USER PERMIT
           → UPWMI Algorithm 1 (deterministic beam search)
           → Reconstruct with calibrated operator
           → Per-modality metrics
           → Analysis Agent
           → RunBundle Export
```

### 13.5 Mode 3: RunBundle Export + Interactive Viewer

Follows Mode 1 or 2 automatically.

---

## 14. Element Visualization: Physics Stage vs Illustration Stage

### 14.1 Two Separate Concerns

| Physics Stage (always generated, reproducible) | Illustration Stage (optional, licensed) |
|---|---|
| `before.npy` / `before.png` — signal entering element | Component photos / vendor images |
| `after.npy` / `after.png` — signal exiting element | Schematic diagrams |
| `noise_map.npy` / `noise_map.png` — noise at this stage | 3D renderings of optical setup |
| `transfer_plot.png` — transfer function visualization | |
| Deterministic, seeded, bit-exact | May require external assets |

### 14.2 Physics Stage Visualizer

```python
class PhysicsStageVisualizer:
    """Generates deterministic before/after/noise images per element."""

    def visualize_chain(self, system: ImagingSystem, x_gt: np.ndarray,
                        seed: int = 42) -> List[ElementVisualization]:
        rng = np.random.default_rng(seed)
        signal = x_gt.copy()
        results = []

        for elem in system.elements:
            before = signal.copy()

            # Apply deterministic transfer function
            signal = self._apply_transfer(elem, signal)

            # Add deterministic noise (seeded)
            noise = self._sample_noise(elem, signal, rng)
            signal_noisy = signal + noise

            results.append(ElementVisualization(
                element_name=elem.name,
                before=before,          # np.ndarray, saved as .npy + .png
                after=signal,           # Before noise
                after_noisy=signal_noisy,
                noise_map=noise,
                metadata={
                    "transfer_kind": elem.transfer_kind.value,
                    "throughput": elem.throughput,
                    "snr_at_element": self._compute_local_snr(signal, noise),
                },
            ))

            signal = signal_noisy  # Propagate noisy version

        return results
```

### 14.3 Illustration Stage (AssetManager)

```python
class AssetManager:
    """Manages illustration assets with licensing tracking."""

    def __init__(self, source: Literal["local_pack", "generated_svg", "external_fetch_disabled"]):
        self.source = source

    def get_component_image(self, element_type: str, modality: str) -> Optional[str]:
        """Returns path to illustration image, or None if unavailable."""
        if self.source == "local_pack":
            path = self._local_path(element_type, modality)
            return path if path.exists() else None
        elif self.source == "generated_svg":
            return self._generate_svg_schematic(element_type, modality)
        return None

    def write_license_manifest(self, output_dir: str):
        """Always writes asset_license.json alongside illustration assets."""
        ...
```

### 14.4 Phase-Space Visualization (for coherent modalities)

For Phase Retrieval, FPM, Holography, and Ptychography, visualization shows **both Magnitude and Phase**:

```python
class CoherentVisualizer(PhysicsStageVisualizer):
    """Extended visualizer for complex-valued signals."""

    def visualize_element(self, elem, signal_complex):
        return {
            "magnitude_before": np.abs(signal_complex),
            "phase_before": np.angle(signal_complex),
            "magnitude_after": np.abs(output),
            "phase_after": np.angle(output),
        }
```

### 14.5 Intermediary Noise Injection

Per physical continuity feedback: the Mismatch Agent can inject noise at intermediate elements (not just the detector):

```python
class IntermediaryNoiseInjector:
    """Inject noise at non-detector elements (stray light, mechanical vibration, etc.)."""

    def inject(self, elem: ElementSpec, signal: np.ndarray,
               mismatch_report: MismatchReport, rng) -> np.ndarray:
        if elem.element_type == "mask" and NoiseKind.alignment in elem.noise_kinds:
            # Stray light at mask plane
            stray = rng.poisson(mismatch_report.parameters.get("stray_light", {}).get("value", 0),
                                size=signal.shape)
            return signal + stray
        if elem.element_type == "transducer" and NoiseKind.acoustic in elem.noise_kinds:
            # Acoustic interference
            interference = rng.normal(0, mismatch_report.parameters.get("acoustic_noise", {}).get("value", 0.01),
                                      size=signal.shape)
            return signal + interference
        return signal
```

---

## 15. Per-Modality Metrics (Beyond PSNR)

### 15.1 Metric Sets (from metrics_db.yaml)

```yaml
# metrics_db.yaml
version: "1.0"

metric_sets:
  intensity_2d:
    description: "Standard 2D intensity images"
    metrics: [psnr, ssim, nrmse]
    modalities: [widefield, widefield_lowdose, confocal_livecell, confocal_3d,
                 sim, lensless, lightsheet, matrix, panorama, integral]

  hyperspectral:
    description: "Spectral datacubes"
    metrics: [psnr, ssim, sam, ergas]
    modalities: [cassi]

  video_temporal:
    description: "Video / temporal sequences"
    metrics: [psnr, ssim, temporal_consistency]
    modalities: [cacti]

  compressive_spatial:
    description: "Compressive sensing (spatial)"
    metrics: [psnr, ssim, nrmse]
    modalities: [spc]

  medical_volumetric:
    description: "Medical imaging (CT/MRI)"
    metrics: [psnr, ssim, nrmse]
    modalities: [ct, mri]

  phase_imaging:
    description: "Complex-valued / phase objects"
    metrics: [phase_rmse, magnitude_psnr, wrapped_phase_error, ssim]
    modalities: [ptychography, holography, phase_retrieval, fpm]

  neural_rendering:
    description: "Novel view synthesis"
    metrics: [psnr, ssim, lpips_optional, depth_error_optional]
    modalities: [nerf, gaussian_splatting]

  diffuse_tomography:
    description: "Diffuse optical / photoacoustic"
    metrics: [nrmse, cnr, spatial_resolution_fwhm]
    modalities: [dot, photoacoustic]

  oct_axial:
    description: "OCT (axial depth)"
    metrics: [psnr, ssim, snr_db, axial_resolution]
    modalities: [oct]

  lifetime:
    description: "Fluorescence lifetime"
    metrics: [lifetime_rmse_ns, phasor_distance, chi_squared]
    modalities: [flim]

  light_field_4d:
    description: "Light field / integral imaging"
    metrics: [psnr, ssim, depth_error_mm, angular_consistency]
    modalities: [light_field]
```

### 15.2 Metric Implementations

```python
# New metric functions (added to analysis/metrics.py)

def spectral_angle_mapper(x_hat, x_true):
    """SAM for hyperspectral data. Lower is better."""
    ...

def temporal_consistency(video_hat, video_true):
    """Frame-to-frame difference consistency."""
    ...

def phase_rmse(phase_hat, phase_true):
    """RMSE on unwrapped phase."""
    ...

def wrapped_phase_error(phase_hat, phase_true):
    """Error accounting for 2pi wrapping."""
    ...

def contrast_to_noise_ratio(recon, background_roi, signal_roi):
    """CNR for DOT/PAT."""
    ...

def lifetime_rmse(tau_hat, tau_true):
    """RMSE on fluorescence lifetime in nanoseconds."""
    ...
```

### 15.3 Acceptance Criteria (Updated)

Old: "within 1 dB of published"

New:
- **Within expected synthetic range** (modality-specific, from calibration tables)
- **Stable across seeds** (run with 3 different seeds, check variance < 0.5 dB)
- **Passes regression tests** (no degradation from previous baseline)

---

## 16. UPWMI Operator Correction: Scoring, Caching, Budget Control

### 16.1 Unified Scoring Function

```python
def upwmi_score(theta: Dict, y: np.ndarray, operator_factory: Callable,
                proxy_solver: str, weights: ScoringWeights) -> float:
    """Unified scoring for operator correction beam search.

    score = alpha * residual_norm
          + beta  * residual_whiteness
          + gamma * prior_plausibility
          + delta * recon_proxy_cost
    """
    # Build operator with candidate theta
    A_theta = operator_factory(theta)

    # Quick proxy reconstruction
    x_proxy = run_proxy_solver(y, A_theta, solver_id=proxy_solver, max_iters=40)

    # Residual
    y_hat = A_theta.forward(x_proxy)
    residual = y - y_hat

    # Terms
    res_norm = float(np.linalg.norm(residual))
    # Per-modality residual whiteness (modality-aware hook)
    whiteness = modality_residual_whiteness(modality_key, residual)
    prior = theta_prior_plausibility(theta, mismatch_db_entry)
    proxy_cost = float(np.sum(np.abs(x_proxy)))  # Sparsity proxy

    return (weights.alpha * res_norm +
            weights.beta * (1.0 - whiteness) +
            weights.gamma * (1.0 - prior) +
            weights.delta * proxy_cost)

def modality_residual_whiteness(modality: str, residual: np.ndarray) -> float:
    """Per-modality residual whiteness scoring.

    Residual structure differs across modalities:
    - CT: streak artifacts in sinogram domain
    - CASSI: spectral mixing in dispersion direction
    - MRI: k-space ringing patterns
    - Photoacoustic: acoustic reverberations

    Each modality extracts relevant features, then combines into [0,1] score.
    """
    features = residual_features(modality, residual)
    # Combine features into single whiteness score
    autocorr = features.get("autocorrelation_lag1", 0.0)
    hf_ratio = features.get("hf_energy_ratio", 0.5)
    structure = features.get("structured_artifact_score", 0.0)
    return float(np.clip(1.0 - abs(autocorr) - structure + hf_ratio * 0.5, 0, 1))

def residual_features(modality: str, residual: np.ndarray) -> Dict[str, float]:
    """Extract modality-specific residual features."""
    base = {
        "autocorrelation_lag1": float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
            if len(residual) > 1 else 0.0,
        "hf_energy_ratio": _hf_energy_ratio(residual),
    }
    # Modality-specific features
    if modality == "ct":
        base["streak_score"] = _ct_streak_score(residual)
        base["structured_artifact_score"] = base["streak_score"]
    elif modality in ("cassi", "cacti"):
        base["spectral_mixing_score"] = _spectral_mixing_score(residual)
        base["structured_artifact_score"] = base["spectral_mixing_score"]
    elif modality == "mri":
        base["kspace_ringing_score"] = _kspace_ringing_score(residual)
        base["structured_artifact_score"] = base["kspace_ringing_score"]
    else:
        base["structured_artifact_score"] = 0.0
    return base
```

### 16.2 Caching (Bug Fixed + Process-Safe)

```python
class CandidateCache:
    """Cache operator builds and proxy reconstructions to avoid redundant work.

    v3 fixes:
    - self.max_size is now properly assigned
    - Stable float serialization (round to avoid numpy dtype issues)
    - Optional disk backend for multi-process beam search
    """

    def __init__(self, max_size: int = 100,
                 disk_path: Optional[str] = None):
        self.max_size = max_size        # BUG FIX: was missing in v2
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._disk_path = disk_path     # Optional SQLite/LMDB for parallel

    def key(self, theta: Dict) -> str:
        """Deterministic cache key with stable float serialization."""
        # Round floats to 10 decimal places to avoid numpy dtype instability
        def _stable(v):
            if isinstance(v, (float, np.floating)):
                return round(float(v), 10)
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, np.ndarray):
                return round(float(np.sum(v)), 10)  # hash-like summary
            return str(v)
        stable_theta = {k: _stable(v) for k, v in sorted(theta.items())}
        return json.dumps(stable_theta, sort_keys=True)

    def get(self, theta: Dict) -> Optional[Tuple[np.ndarray, float]]:
        k = self.key(theta)
        if k in self._cache:
            return self._cache[k]
        if self._disk_path:
            return self._disk_get(k)
        return None

    def put(self, theta: Dict, x_proxy: np.ndarray, score: float):
        k = self.key(theta)
        if len(self._cache) < self.max_size:
            self._cache[k] = (x_proxy, score)
        if self._disk_path:
            self._disk_put(k, x_proxy, score)
```

### 16.3 Budget Guardrails

```python
class UPWMIBudget(BaseModel):
    max_candidates: int = 24
    max_refinements_per_candidate: int = 8
    max_total_runtime_s: float = 300.0
    early_stop_plateau_delta: float = 1e-3
    early_stop_patience: int = 3
    proxy_solver: str = "fista"
    proxy_max_iters: int = 40
```

### 16.4 Physics-Informed Jacobian (Active Learning Extension)

Instead of blind grid search, use sensitivity information when available:

```python
class ActiveLearningSearch:
    """Use physics-informed Jacobian to guide beam search direction."""

    def refine_with_gradient(self, theta_init: Dict, y: np.ndarray,
                             operator_factory: Callable) -> Dict:
        """If operator supports autodiff, use gradient of residual w.r.t. theta."""
        A = operator_factory(theta_init)

        if A.supports_autodiff:
            # Use torch autograd or jax.grad
            grad_theta = self._compute_residual_gradient(y, A, theta_init)
            # Take gradient step (bounded)
            theta_new = {}
            for k, v in theta_init.items():
                step = -self.learning_rate * grad_theta.get(k, 0.0)
                theta_new[k] = np.clip(v + step, *self.bounds[k])
            return theta_new
        else:
            # Fall back to finite-difference local search
            return self._finite_difference_search(theta_init, y, operator_factory)
```

---

## 17. Hybrid Modalities

### 17.1 Design

Some applications combine two modalities (e.g., Fluorescence + Photoacoustic). The Plan Agent can recognize this and build a hybrid system:

```python
class HybridModalitySpec(BaseModel):
    primary: str           # e.g., "photoacoustic"
    secondary: str         # e.g., "widefield"
    fusion_strategy: str   # "parallel_recon_then_fuse" | "joint_forward"
    shared_parameters: Dict[str, Any]  # e.g., shared sample, shared wavelength

# Recognized hybrid combinations:
HYBRID_DB = {
    ("photoacoustic", "widefield"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": "Photoacoustic for deep vascular imaging + widefield for surface fluorescence",
    },
    ("oct", "confocal"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": "OCT for depth + confocal for lateral resolution",
    },
    ("fpm", "holography"): {
        "fusion_strategy": "joint_forward",
        "description": "FPM synthetic aperture + holographic phase",
    },
}
```

When the Plan Agent detects a hybrid scenario:
1. Builds two parallel forward models
2. Runs reconstruction for each
3. Applies a fusion step (weighted average, wavelet fusion, or learned fusion)
4. Reports combined metrics

---

## 18. RunBundle Export + Interactive Viewer

### 18.1 RunBundle Structure (Updated with Expanded Provenance)

```
RunBundle/
├── spec.json
├── provenance.json           # EXPANDED: see Section 18.1b
├── data/
│   ├── x_true.npy
│   ├── y.npy
│   └── data_manifest.json
├── results/
│   ├── recon/
│   │   ├── xhat_ideal.npy
│   │   ├── xhat_real.npy
│   │   └── xhat_corrected.npy
│   ├── metrics.json                   # Per-modality metric set
│   ├── report.json
│   └── report.md
├── agents/
│   ├── plan_intent.json               # PlanIntent pydantic
│   ├── modality_selection.json        # ModalitySelection pydantic
│   ├── imaging_system.json            # ImagingSystem pydantic
│   ├── photon_report.json             # PhotonReport pydantic
│   ├── mismatch_report.json           # MismatchReport pydantic
│   ├── recoverability_report.json     # RecoverabilityReport pydantic
│   ├── analysis_report.json           # SystemAnalysis pydantic
│   ├── preflight_report.json          # PreFlightReport pydantic
│   └── negotiation_result.json
├── visualizations/
│   ├── physics_stage/                 # Always generated, reproducible
│   │   ├── element_01_before.npy
│   │   ├── element_01_before.png
│   │   ├── element_01_after.npy
│   │   ├── element_01_after.png
│   │   ├── element_01_noise_map.npy
│   │   ├── element_01_noise_map.png
│   │   └── ...
│   ├── illustrations/                 # Optional, with licensing
│   │   ├── system_schematic.svg
│   │   └── asset_license.json
│   ├── photon_budget.png
│   └── compression_analysis.png
├── internal_state/
│   ├── operator_fit/
│   │   ├── candidates.json
│   │   ├── scores.json
│   │   ├── cache_stats.json
│   │   └── theta_best.json
│   └── self_improvement/              # If loop was run
│       ├── alternatives.json
│       └── proxy_results.json
├── codegen/
│   └── reproduce.py
└── what_if/                           # Interactive viewer data
    ├── photon_sensitivity.json        # Pre-computed: PSNR vs photon budget
    └── cr_sensitivity.json            # Pre-computed: PSNR vs compression ratio
```

### 18.1b Expanded Provenance (Auditable Artifacts)

RunBundles are "auditable artifacts," not just outputs. provenance.json must include
everything needed to reproduce the run exactly:

```python
class RunBundleProvenance(StrictBaseModel):
    """Captured at run time, written to provenance.json."""
    # Source code
    git_commit_hash: str
    git_dirty: bool                      # True if uncommitted changes
    git_branch: str

    # Environment
    python_version: str                  # e.g. "3.11.5"
    platform: str                        # e.g. "Linux-6.14.0-x86_64"
    cpu_info: str                        # e.g. "AMD EPYC 7B13"
    gpu_info: Optional[str] = None       # e.g. "NVIDIA A100 40GB"
    package_versions: Dict[str, str]     # {"numpy": "1.26.4", "torch": "2.1.0", ...}
    lockfile_hash: Optional[str] = None  # SHA256 of requirements.txt / poetry.lock

    # Reproducibility
    rng_seeds: Dict[str, int]            # {"global": 42, "photon": 42, "recon": 42}
    operator_serialization_hash: str     # SHA256 of operator.serialize() output
    dataset_hash: Optional[str] = None   # SHA256 of input data files
    dataset_id: Optional[str] = None     # Human-readable dataset identifier

    # Timing
    start_time_utc: str                  # ISO 8601
    end_time_utc: str
    total_runtime_s: float

def capture_provenance() -> RunBundleProvenance:
    import platform, subprocess, hashlib, sys
    return RunBundleProvenance(
        git_commit_hash=subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip(),
        git_dirty=bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).decode().strip()),
        git_branch=subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(),
        python_version=sys.version,
        platform=platform.platform(),
        cpu_info=platform.processor(),
        gpu_info=_detect_gpu(),
        package_versions=_get_package_versions(),
        rng_seeds={},  # Filled per-stage
        operator_serialization_hash="",  # Filled after operator build
        start_time_utc=datetime.utcnow().isoformat(),
        end_time_utc="",  # Filled at end
        total_runtime_s=0.0,
    )
```

### 18.1c Illustrations Toggle

Illustrations (component photos, schematics) are separate from physics-stage
visualizations and can be toggled:

```
# CLI flag
pwm run spec.yaml --illustrations=off     # No illustrations (headless CI)
pwm run spec.yaml --illustrations=svg     # Auto-generated SVG schematics
pwm run spec.yaml --illustrations=local_pack  # Use bundled asset images
```

### 18.2 Interactive Viewer ("What-If" Sliders)

The viewer includes pre-computed sensitivity curves so users can explore tradeoffs without re-running:

```python
class WhatIfPrecomputer:
    """Pre-compute sensitivity curves for the interactive viewer."""

    def compute_photon_sensitivity(self, context, n_points=5) -> dict:
        """PSNR vs photon budget (using quick proxy)."""
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        results = []
        for m in multipliers:
            budget = context.budget.copy()
            budget["max_photons"] *= m
            proxy = self._quick_proxy(context, budget)
            results.append({"multiplier": m, "estimated_psnr_db": proxy.psnr})
        return {"photon_sensitivity": results}

    def compute_cr_sensitivity(self, context, n_points=5) -> dict:
        """PSNR vs compression ratio (from calibration table interpolation)."""
        ...
```

---

## 19. Implementation Phases (Incremental, Prove-Value-First)

### Phase 1: Agent Infrastructure + Registry + RunBundle (No New Modalities)

**Goal:** Prove the agent pipeline works end-to-end with existing 18 modalities.

**Deliverables:**
1. `agents/contracts.py` — All pydantic models with `StrictBaseModel` (Section 2)
2. `agents/llm_client.py` — Multi-LLM wrapper (Gemini/Claude/OpenAI, env-based keys)
3. `agents/base.py` — BaseAgent (runs without LLM, LLM optional)
4. `agents/registry_schemas.py` — Pydantic schemas for all YAML files
5. `agents/_generated_literals.py` — Build-time Literal types from YAML
6. `contrib/modalities.yaml` — 18 existing modalities (with upload_template)
7. `contrib/mismatch_db.yaml` — Mismatch DB for existing modalities
8. `contrib/photon_db.yaml` — Photon models (model_id, not formulas)
9. `contrib/compression_db.yaml` — Calibration tables with provenance fields
10. `contrib/metrics_db.yaml` — Per-modality metric sets
11. `agents/registry.py` — RegistryBuilder with assert_* helpers for LLM validation
12. `agents/plan_agent.py` — Orchestrator (registry-ID-only LLM output)
13. `agents/photon_agent.py` — Variance-dominance noise model + LLM narrative
14. `agents/mismatch_agent.py` — Deterministic + LLM prior selection (validated)
15. `agents/recoverability_agent.py` — Interpolation + confidence (renamed)
16. `agents/analysis_agent.py` — Bottleneck scoring + suggestions
17. `agents/negotiator.py` — Agent veto/negotiation
18. `agents/continuity_checker.py` — Physical continuity checks
19. `agents/preflight.py` — Pre-flight + CLI modes (--auto-proceed, --force)
20. `agents/physics_stage_visualizer.py` — Deterministic before/after images
21. Update `physics/base.py` — check_adjoint() + enhanced serialize()
22. Update `core/runner.py` — Agent pipeline + permit step
23. Update `core/runbundle/` — Expanded provenance + agent outputs
24. `tests/test_registry_integrity.py` — Orphan detection + cross-ref
25. `tests/test_contract_fuzzing.py` — Hypothesis-based fuzzing

**Test:** End-to-end for CASSI, CT, MRI (3 representative modalities). All agents
run without LLM (deterministic path). Registry integrity passes. Contract fuzzing passes.

### Phase 2: Add OCT + Light Field End-to-End (2 New Modalities)

**Goal:** Prove the pattern for adding new modalities works before scaling to 6 more.

**Deliverables:**
1. `physics/oct/oct_operator.py` — OCT forward model (unified interface)
2. `recon/oct_solver.py` — FFT-based, BM3D post-processing
3. `physics/light_field/lf_operator.py` — Light field forward model
4. `recon/light_field_solver.py` — Shift-and-sum
5. Add to `modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `compression_db.yaml`
6. Add to `solver_registry.yaml`
7. Add benchmarks for OCT + Light Field in `run_all.py`
8. CasePacks for both

**Test:** End-to-end Mode 1 for OCT + Light Field. Verify metrics pass.

### Phase 3: Operator Correction for OCT + Light Field + UPWMI Enhancements

**Goal:** Prove operator correction works for new modalities.

**Deliverables:**
1. Add OCT + Light Field to `test_operator_correction.py`
2. `agents/upwmi.py` — Unified scoring function, caching, budget guardrails
3. Active learning extension (gradient-based refinement for autodiff operators)
4. Self-improvement loop implementation

**Test:** Mode 2 for OCT + Light Field. Verify >3 dB improvement.

### Phase 4: Remaining 6 New Modalities (Patterns Established)

**Goal:** Scale to all 26 modalities using established patterns from Phase 2-3.

**Deliverables (per modality):**
1. `physics/<modality>/<modality>_operator.py`
2. `recon/<modality>_solver.py`
3. YAML registry entries
4. CasePack
5. Benchmark
6. Operator correction test

Modalities: DOT, Photoacoustic, FLIM, Phase Retrieval, Integral, FPM.

### Phase 5: Interactive Viewer + Hybrid Modalities + Polish

**Goal:** Complete feature set.

**Deliverables:**
1. `agents/what_if_precomputer.py` — Sensitivity curves
2. `agents/asset_manager.py` — Illustration stage + licensing
3. `agents/hybrid.py` — Hybrid modality support
4. Update README.md from YAML (auto-generation)
5. Full documentation update

---

## 20. File Structure

```
packages/pwm_core/
├── pwm_core/
│   ├── agents/                        # NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── contracts.py               # All pydantic schemas (Section 2)
│   │   ├── base.py                    # BaseAgent (LLM optional)
│   │   ├── llm_client.py              # Multi-LLM (Gemini/Claude/OpenAI)
│   │   ├── registry.py                # RegistryBuilder (loads YAML)
│   │   ├── registry_schemas.py        # Pydantic schemas for YAML files
│   │   ├── _generated_literals.py     # Auto-generated Literal types (CI)
│   │   ├── plan_agent.py              # Orchestrator
│   │   ├── photon_agent.py            # Deterministic photon computation
│   │   ├── mismatch_agent.py          # Deterministic mismatch analysis
│   │   ├── recoverability_agent.py     # Practical recoverability model (renamed)
│   │   ├── analysis_agent.py          # Bottleneck scoring + suggestions
│   │   ├── negotiator.py              # Agent veto / negotiation
│   │   ├── continuity_checker.py      # Physical continuity validation
│   │   ├── preflight.py               # Pre-flight report builder
│   │   ├── upwmi.py                   # Unified UPWMI scoring/caching/budget
│   │   ├── self_improvement.py        # Design advisor loop
│   │   ├── physics_stage_visualizer.py # Deterministic element visualization
│   │   ├── asset_manager.py           # Illustration assets + licensing
│   │   ├── what_if_precomputer.py     # Interactive viewer sensitivity curves
│   │   └── hybrid.py                  # Hybrid modality support
│   │
│   ├── physics/                       # Existing + NEW operator files
│   │   ├── base.py                    # UPDATED: unified interface
│   │   ├── oct/
│   │   │   └── oct_operator.py        # Phase 2
│   │   ├── light_field/
│   │   │   └── lf_operator.py         # Phase 2
│   │   ├── diffuse_optical/
│   │   │   └── dot_operator.py        # Phase 4
│   │   ├── photoacoustic/
│   │   │   └── pa_operator.py         # Phase 4
│   │   ├── flim/
│   │   │   └── flim_operator.py       # Phase 4
│   │   ├── phase_retrieval/
│   │   │   └── cdi_operator.py        # Phase 4
│   │   ├── integral/
│   │   │   └── integral_operator.py   # Phase 4
│   │   └── fpm/
│   │       └── fpm_operator.py        # Phase 4
│   │
│   ├── recon/                         # Existing + NEW solver files
│   │   ├── oct_solver.py              # Phase 2
│   │   ├── light_field_solver.py      # Phase 2
│   │   ├── dot_solver.py              # Phase 4
│   │   ├── photoacoustic_solver.py    # Phase 4
│   │   ├── flim_solver.py            # Phase 4
│   │   ├── phase_retrieval_solver.py  # Phase 4
│   │   ├── integral_solver.py         # Phase 4
│   │   └── fpm_solver.py             # Phase 4
│   │
│   ├── analysis/
│   │   └── metrics.py                 # UPDATED: phase_rmse, SAM, CNR, etc.
│   │
│   └── core/
│       ├── runner.py                  # UPDATED: agent pipeline integration
│       └── runbundle/                 # UPDATED: agent output saving
│
├── contrib/
│   ├── modalities.yaml                # NEW: 26-modality source of truth
│   ├── mismatch_db.yaml              # NEW: mismatch parameters
│   ├── photon_db.yaml                # NEW: photon models
│   ├── compression_db.yaml           # NEW: recoverability calibration tables
│   ├── metrics_db.yaml               # NEW: per-modality metric sets
│   ├── solver_registry.yaml          # UPDATED: add 8 new modalities
│   └── casepacks/                    # UPDATED: add 8 new CasePacks
│
└── benchmarks/
    ├── run_all.py                    # UPDATED: add new modality benchmarks
    └── test_operator_correction.py   # UPDATED: add new modality corrections
```

**Phase 1: ~25 new files (agents/ + tests/) + 5 YAML registries + updates**
**Phase 2: +4 new files (2 operators + 2 solvers)**
**Phase 3: +2 new files (upwmi.py, self_improvement.py) + updates**
**Phase 4: +12 new files (6 operators + 6 solvers)**
**Phase 5: +3 new files (viewer, assets, hybrid)**

---

## 21. Testing Strategy

### 21.1 Unit Tests

- **Pydantic contract tests:** Round-trip serialization, validation rejection, extra field rejection
- **Strict mode tests:** Verify `extra="forbid"` rejects unknown fields, NaN/Inf rejected
- **Registry tests:** All YAML cross-references valid, no orphan keys (see 21.1b)
- **Agent tests:** Each agent independently **without any LLM** (deterministic path only)
- **Operator tests:** `check_adjoint()` on every operator, forward/adjoint consistency
- **Metric tests:** Known inputs → known outputs

### 21.1b Registry Integrity Test (Fail on ANY Orphan)

```python
# tests/test_registry_integrity.py

def test_no_orphan_modality_keys():
    """Every key in every sub-registry must exist in modalities.yaml."""
    registry = RegistryBuilder(CONTRIB_DIR)
    # This already asserts in __init__, but be explicit:
    mod_keys = set(registry.list_modalities())

    for key in load_yaml("mismatch_db.yaml")["modalities"]:
        assert key in mod_keys, f"Orphan mismatch key: {key}"
    for key in load_yaml("compression_db.yaml")["calibration_tables"]:
        assert key in mod_keys, f"Orphan compression key: {key}"
    for key in load_yaml("photon_db.yaml")["modalities"]:
        assert key in mod_keys, f"Orphan photon key: {key}"

def test_no_unknown_solver_ids():
    """Every solver_id in compression_db must exist in solver_registry.yaml."""
    solvers = load_yaml("solver_registry.yaml")
    valid_ids = _collect_all_solver_ids(solvers)
    compression = load_yaml("compression_db.yaml")
    for modality, table in compression["calibration_tables"].items():
        for entry in table["entries"]:
            assert entry["solver"] in valid_ids, \
                f"Unknown solver '{entry['solver']}' in compression_db[{modality}]"

def test_no_unknown_metric_names():
    """Every metric name in metrics_db must map to an implemented function."""
    from pwm_core.analysis.metrics import METRIC_FUNCTIONS
    metrics = load_yaml("metrics_db.yaml")
    for ms in metrics["metric_sets"].values():
        for m in ms["metrics"]:
            assert m in METRIC_FUNCTIONS, f"Unknown metric: {m}"

def test_no_missing_required_fields():
    """Load all YAML through Pydantic schemas — any missing field fails."""
    ModalitiesFileYaml.model_validate(load_yaml("modalities.yaml"))
    MismatchDbFileYaml.model_validate(load_yaml("mismatch_db.yaml"))
    CompressionDbFileYaml.model_validate(load_yaml("compression_db.yaml"))
    PhotonDbFileYaml.model_validate(load_yaml("photon_db.yaml"))
    MetricsDbFileYaml.model_validate(load_yaml("metrics_db.yaml"))

def test_every_modality_has_all_registries():
    """Every modality key must appear in ALL sub-registries."""
    mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
    for key in mod_keys:
        assert key in load_yaml("mismatch_db.yaml")["modalities"], \
            f"{key} missing from mismatch_db"
        assert key in load_yaml("compression_db.yaml")["calibration_tables"], \
            f"{key} missing from compression_db"
        assert key in load_yaml("photon_db.yaml")["modalities"], \
            f"{key} missing from photon_db"

def test_calibration_provenance_present():
    """Every calibration table entry must have provenance fields."""
    compression = CompressionDbFileYaml.model_validate(
        load_yaml("compression_db.yaml"))
    for modality, table in compression.calibration_tables.items():
        for i, entry in enumerate(table.entries):
            assert entry.provenance is not None, \
                f"Missing provenance: compression_db[{modality}].entries[{i}]"
            assert entry.provenance.dataset_id, \
                f"Empty dataset_id: compression_db[{modality}].entries[{i}]"
```

### 21.1c Contract Fuzzing (Random-but-Valid Specs)

Generate random-but-bounded ExperimentSpec variations and ensure the pipeline
doesn't crash:

```python
# tests/test_contract_fuzzing.py
import hypothesis
from hypothesis import given, strategies as st

def modality_strategy():
    """Generate valid modality keys."""
    registry = RegistryBuilder(CONTRIB_DIR)
    return st.sampled_from(registry.list_modalities())

def noise_regime_strategy():
    return st.sampled_from(list(NoiseRegime))

def signal_prior_strategy():
    return st.sampled_from(list(SignalPriorClass))

@given(
    modality=modality_strategy(),
    n_photons=st.floats(min_value=1, max_value=1e8),
    read_noise=st.floats(min_value=0, max_value=100),
    cr=st.floats(min_value=0.001, max_value=1.0),
)
def test_photon_agent_never_crashes(modality, n_photons, read_noise, cr):
    """PhotonAgent must produce a valid report for any bounded input."""
    agent = PhotonAgent(registry=RegistryBuilder(CONTRIB_DIR))
    context = make_context(modality=modality, n_photons=n_photons,
                           read_noise=read_noise, cr=cr)
    report = agent.run(context)
    # Must validate as strict pydantic
    PhotonReport.model_validate(report.model_dump())

@given(modality=modality_strategy())
def test_preflight_always_produces_report(modality):
    """Pre-flight must produce a report for any valid modality."""
    context = make_default_context(modality)
    agents_output = run_all_agents(context)
    report = PreFlightReportBuilder().build(**agents_output)
    PreFlightReport.model_validate(report.model_dump())

@given(
    modality=modality_strategy(),
    noise=noise_regime_strategy(),
    prior=signal_prior_strategy(),
)
def test_recoverability_agent_never_crashes(modality, noise, prior):
    """RecoverabilityAgent must handle any valid (modality, noise, prior) combo."""
    agent = RecoverabilityAgent(registry=RegistryBuilder(CONTRIB_DIR))
    context = make_context(modality=modality, noise_regime=noise,
                           signal_prior=prior)
    report = agent.run(context)
    RecoverabilityReport.model_validate(report.model_dump())
```

### 21.2 Integration Tests

- **Full pipeline:** CASSI, CT, MRI end-to-end (Phase 1)
- **New modality:** OCT, Light Field end-to-end (Phase 2)
- **Mode 2:** Operator correction for 5 representative modalities
- **Pre-flight:** Verify permit step blocks on veto conditions
- **Negotiation:** Verify conflict detection (low photon + high CR)
- **check_adjoint:** Every operator passes adjoint test in CI

### 21.3 Acceptance Criteria (Per-Modality)

| Test | Criterion |
|---|---|
| PSNR in expected range | Within calibration table ± 1 dB |
| Stable across seeds | Variance < 0.5 dB over 3 seeds |
| Regression check | No degradation from previous baseline |
| Operator correction | >3 dB improvement when mismatch is present |
| Phase metrics (coherent) | phase_rmse < 0.3 rad on synthetic data |
| Spectral metrics (CASSI) | SAM < 5 degrees on KAIST test |

---

## 22. Summary

### What Changed from Plan v2 → v3

| # | Change | Rationale | Section |
|---|---|---|---|
| 1 | LLM returns only registry IDs, mechanically enforced | Prevent hallucinated mismatch families / solver names | 1.1b |
| 2 | `StrictBaseModel(extra="forbid", validate_assignment=True)` everywhere | Reject garbage structures silently accepted by v2 | 2.1 |
| 3 | NaN/Inf rejection via model validator | Float fields can't contain NaN/Inf | 2.1 |
| 4 | `ModalitySelection` validator removed; registry validation outside model | v2 validator was a no-op; now `registry.assert_modality_exists()` | 2.2 |
| 5 | Build-time `Literal[...]` from YAML keys (optional) | Static type checking catches invalid strings | 2.4 |
| 6 | Per-registry Pydantic schemas (`ModalitiesFileYaml`, etc.) | YAML validated by Pydantic + CI cross-ref tests | 3.3 |
| 7 | Calibration table entries require provenance fields | Prevent silent drift when solvers/operators change | 3.5 |
| 8 | `serialize()` includes blob paths + SHA256 hashes for large arrays | Masks, PSFs, trajectories reproducible | 4.2 |
| 9 | `check_adjoint(n_trials, tol, seed)` on every operator | Catches bugs early, saves weeks | 4.1 |
| 10 | Noise model uses variance dominance, not threshold-only | Correct physics: shot vs read vs dark variance fractions | 7.2 |
| 11 | YAML stores `model_id` + parameters only; code implements compute() | No eval-style formula execution from YAML | 7.3 |
| 12 | `CandidateCache.max_size` bug fixed | Was missing `self.max_size = max_size` | 16.2 |
| 13 | Deterministic + process-safe caching (stable float serialization, disk backend) | Parallel beam search safe | 16.2 |
| 14 | Per-modality `residual_features()` hook for whiteness scoring | CT streaks ≠ CASSI spectral mixing ≠ MRI ringing | 16.1 |
| 15 | Table lookup with interpolation + confidence + uncertainty band | Real CR/noise rarely match table exactly | 9.2 |
| 16 | CLI permit modes: `--auto-proceed`, `--force` | Batch/CI runs need programmatic override | 12.3 |
| 17 | Modality-aware upload templates from YAML | Structured "what to provide" per modality | 12.4 |
| 18 | Expanded provenance: git hash, dirty, python, platform, CPU/GPU, seeds, hashes | RunBundles become auditable artifacts | 18.1b |
| 19 | Registry integrity test: fails on any orphan key/field | No silent mismatches between YAML files | 21.1b |
| 20 | Contract fuzzing with Hypothesis | Random-but-valid inputs never crash | 21.1c |
| 21 | Multi-LLM fallback: Gemini → Claude → OpenAI | Works if any provider is available | 1.3 |
| 22 | Renamed `CompressedAgent` → `RecoverabilityAgent` | Matches actual function | throughout |
| 23 | `--illustrations=off\|svg\|local_pack` toggle | Separate physics stage from illustration stage | 18.1c |
| 24 | Agents must run without LLM (design mantra in code) | Deterministic outputs guaranteed | 1.2 |

### Cumulative Changes (v1 → v2 → v3)

| # | v1 → v2 Change | v2 → v3 Hardening |
|---|---|---|
| API keys | Hardcoded → env variable | Multi-LLM fallback chain |
| LLM role | All agents = Gemini calls | LLM returns only registry IDs (enforced) |
| Pydantic | Mentioned loosely | `extra="forbid"`, NaN/Inf rejected, model validators |
| YAML registries | Python dicts | Pydantic schemas + provenance + CI cross-ref |
| Recoverability | Fake CS formula → calibration tables | + interpolation + confidence + uncertainty |
| Operators | serialize() = id/theta/shapes | + blob paths + SHA256 hashes + check_adjoint() |
| Noise model | Threshold-based | Variance dominance |
| Photon formulas | "Stored in YAML" | YAML stores model_id, code dispatches |
| UPWMI cache | Had max_size bug | Fixed + stable float serialization + disk backend |
| UPWMI scoring | Generic whiteness | Per-modality residual_features() hook |
| Pre-flight | Show and ask | + CLI `--auto-proceed` / `--force` + upload templates |
| Provenance | Basic | git hash, dirty, platform, GPU, seeds, hashes |
| Testing | Unit + integration | + registry integrity + contract fuzzing |

### Component Summary

| Component | Type | Phase | Files |
|---|---|---|---|
| Pydantic contracts (strict) | NEW | 1 | `agents/contracts.py` |
| LLM client (multi-provider) | NEW | 1 | `agents/llm_client.py` |
| BaseAgent (LLM optional) | NEW | 1 | `agents/base.py` |
| YAML registries (5 files) | NEW | 1 | `contrib/*.yaml` |
| Registry schemas | NEW | 1 | `agents/registry_schemas.py` |
| Registry builder | NEW | 1 | `agents/registry.py` |
| Generated Literals | NEW | 1 | `agents/_generated_literals.py` |
| Plan Agent | NEW | 1 | `agents/plan_agent.py` |
| Photon Agent (deterministic) | NEW | 1 | `agents/photon_agent.py` |
| Mismatch Agent (deterministic) | NEW | 1 | `agents/mismatch_agent.py` |
| Recoverability Agent (renamed) | NEW | 1 | `agents/recoverability_agent.py` |
| Analysis Agent | NEW | 1 | `agents/analysis_agent.py` |
| Agent Negotiator | NEW | 1 | `agents/negotiator.py` |
| Physical Continuity Checker | NEW | 1 | `agents/continuity_checker.py` |
| Pre-flight Builder (+CLI modes) | NEW | 1 | `agents/preflight.py` |
| Physics Stage Visualizer | NEW | 1 | `agents/physics_stage_visualizer.py` |
| Unified operator interface (+check_adjoint) | UPDATE | 1 | `physics/base.py` |
| Per-modality metrics | UPDATE | 1 | `analysis/metrics.py` |
| Pipeline integration | UPDATE | 1 | `core/runner.py` |
| RunBundle (+expanded provenance) | UPDATE | 1 | `core/runbundle/` |
| Registry integrity tests | NEW | 1 | `tests/test_registry_integrity.py` |
| Contract fuzzing tests | NEW | 1 | `tests/test_contract_fuzzing.py` |
| OCT operator + solver | NEW | 2 | `physics/oct/`, `recon/oct_solver.py` |
| Light Field operator + solver | NEW | 2 | `physics/light_field/`, `recon/light_field_solver.py` |
| UPWMI unified scoring/caching (fixed) | NEW | 3 | `agents/upwmi.py` |
| Self-improvement loop | NEW | 3 | `agents/self_improvement.py` |
| 6 remaining operators + solvers | NEW | 4 | `physics/*/`, `recon/*.py` |
| What-if precomputer | NEW | 5 | `agents/what_if_precomputer.py` |
| Asset manager (+illustrations toggle) | NEW | 5 | `agents/asset_manager.py` |
| Hybrid modality support | NEW | 5 | `agents/hybrid.py` |
