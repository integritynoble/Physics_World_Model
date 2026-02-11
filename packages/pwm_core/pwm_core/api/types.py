"""
pwm_core.api.types

Pydantic models for the stable PWM API.

These models are intentionally "transport-friendly":
- JSON-serializable
- Versioned
- Designed for RunBundle persistence

They are used by:
- CLI (`pwm run`, `pwm calib-recon`, `pwm fit-operator`)
- AI_Scientist adapter (`pwm_AI_Scientist`)
- Viewer (`pwm view`)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


# -----------------------------
# Common helpers
# -----------------------------

JSON = Dict[str, Any]


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class ValidationMessage(BaseModel):
    severity: Severity
    code: str
    message: str
    path: Optional[str] = None


class ValidationReport(BaseModel):
    ok: bool = True
    messages: List[ValidationMessage] = Field(default_factory=list)
    auto_repair_patch: Optional[JSON] = None  # RFC-6902-like or PWM patch format


# -----------------------------
# Operator input (new mode)
# -----------------------------

class OperatorKind(str, Enum):
    """How the forward operator is provided."""
    parametric = "parametric"   # A(theta), e.g., CASSI with dispersion params
    matrix = "matrix"           # explicit A (dense/sparse) or file reference
    callable = "callable"       # user provided forward/adjoint callables (advanced)


class OperatorParametric(BaseModel):
    operator_id: str = Field(..., description="Registered operator id, e.g., 'cassi'.")
    theta_init: Optional[JSON] = Field(default=None, description="Initial guess for theta.")
    theta_space: Optional[JSON] = Field(default=None, description="Search space spec for theta.")
    assets: Optional[JSON] = Field(default=None, description="Mask/PSF/etc references or inline small assets.")


class OperatorMatrix(BaseModel):
    source: str = Field(..., description="Path or URI to matrix payload (npz/pt/mat).")
    format: Optional[str] = Field(default=None, description="Hint: 'npz', 'pt', 'mat', 'scipy_sparse'.")
    x_shape: Optional[List[int]] = None
    y_shape: Optional[List[int]] = None
    dtype: Optional[str] = None


class OperatorCallable(BaseModel):
    module: str = Field(..., description="Python module path that defines forward/adjoint.")
    symbol_forward: str = Field("forward", description="Function name for forward(x, theta)->y.")
    symbol_adjoint: str = Field("adjoint", description="Function name for adjoint(y, theta)->x.")
    sandbox: bool = Field(False, description="If True, attempt to sandbox (best-effort).")


class OperatorInput(BaseModel):
    kind: OperatorKind
    parametric: Optional[OperatorParametric] = None
    matrix: Optional[OperatorMatrix] = None
    callable: Optional[OperatorCallable] = None


# -----------------------------
# ExperimentSpec (v0.2.1)
# -----------------------------

class InputMode(str, Enum):
    simulate = "simulate"
    measured = "measured"


class TaskKind(str, Enum):
    simulate_recon_analyze = "simulate_recon_analyze"
    reconstruct_only = "reconstruct_only"
    calibrate_and_reconstruct = "calibrate_and_reconstruct"
    fit_operator_only = "fit_operator_only"
    qc_report = "qc_report"
    design_sweep = "design_sweep"


class ExperimentInput(BaseModel):
    mode: InputMode
    y_source: Optional[str] = Field(
        default=None, description="For measured mode: path/URI to y tensor/array."
    )
    x_source: Optional[str] = Field(
        default=None, description="Optional ground truth (for benchmarking)."
    )
    operator: Optional[OperatorInput] = Field(
        default=None, description="For measured mode: operator descriptor."
    )


class PhysicsState(BaseModel):
    modality: str
    dims: Optional[JSON] = None
    rendering: Optional[JSON] = None  # for NeRF/3DGS style


class BudgetState(BaseModel):
    photon_budget: Optional[JSON] = None  # max_photons, exposure_time, bleaching, ...
    measurement_budget: Optional[JSON] = None  # sampling_rate, #views, #frames...


class CalibrationState(BaseModel):
    theta: Optional[JSON] = None
    psf: Optional[JSON] = None
    pattern: Optional[JSON] = None
    camera: Optional[JSON] = None


class EnvironmentState(BaseModel):
    background_level: Optional[float] = None
    attenuation: Optional[JSON] = None
    scattering: Optional[JSON] = None
    scatter: Optional[JSON] = None
    stripe_artifacts: Optional[JSON] = None


class SampleState(BaseModel):
    motion: Optional[JSON] = None
    labeling: Optional[JSON] = None
    blinking: Optional[JSON] = None


class SensorState(BaseModel):
    shot_noise: Optional[JSON] = None
    read_noise_sigma: Optional[float] = None
    saturation_level: Optional[float] = None
    quantization_bits: Optional[int] = None
    fixed_pattern_noise: Optional[JSON] = None
    nonlinearity: Optional[JSON] = None


class ComputeState(BaseModel):
    device: Optional[str] = None  # cpu/cuda
    max_seconds: Optional[float] = None
    max_memory_gb: Optional[float] = None
    streaming: Optional[bool] = None
    tile_size: Optional[List[int]] = None


class TaskState(BaseModel):
    kind: TaskKind
    notes: Optional[str] = None


class MismatchFitOperator(BaseModel):
    enabled: bool = False
    theta_space: Optional[JSON] = None
    search: Optional[JSON] = None
    proxy_recon: Optional[JSON] = None
    scoring: Optional[JSON] = None
    stop: Optional[JSON] = None


class MismatchSpec(BaseModel):
    enabled: bool = False
    fit_operator: Optional[MismatchFitOperator] = None
    theta_true: Optional[JSON] = None
    theta_model: Optional[JSON] = None


class SolverSpec(BaseModel):
    id: str
    params: JSON = Field(default_factory=dict)


class ReconPortfolio(BaseModel):
    solvers: List[SolverSpec] = Field(default_factory=list)


class ReconSpec(BaseModel):
    portfolio: ReconPortfolio = Field(default_factory=ReconPortfolio)


class AnalysisSpec(BaseModel):
    residual_tests: List[str] = Field(default_factory=list)
    advisor: JSON = Field(default_factory=dict)


class RunBundlePolicy(BaseModel):
    data_policy: JSON = Field(default_factory=lambda: {"mode": "auto", "copy_threshold_mb": 100})


class ExportSpec(BaseModel):
    runbundle: RunBundlePolicy = Field(default_factory=RunBundlePolicy)


class ExperimentStates(BaseModel):
    physics: PhysicsState
    budget: Optional[BudgetState] = None
    calibration: Optional[CalibrationState] = None
    environment: Optional[EnvironmentState] = None
    sample: Optional[SampleState] = None
    sensor: Optional[SensorState] = None
    compute: Optional[ComputeState] = None
    task: TaskState


class ExperimentSpec(BaseModel):
    version: Literal["0.2.1"] = "0.2.1"
    id: str
    input: ExperimentInput
    states: ExperimentStates
    mismatch: Optional[MismatchSpec] = None
    recon: ReconSpec = Field(default_factory=ReconSpec)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)


# -----------------------------
# Diagnosis + Action
# -----------------------------

class ActionOp(str, Enum):
    set = "set"
    multiply = "multiply"
    add = "add"
    optimize = "optimize"
    enable = "enable"
    disable = "disable"


class Action(BaseModel):
    knob: str = Field(..., description="Dot path to a spec field, e.g. 'states.budget.photon_budget.max_photons'")
    op: ActionOp
    val: Optional[Any] = Field(default=None, description="Value for set/add/multiply.")
    reason: Optional[str] = None
    evidence: Optional[JSON] = None
    safety: Optional[JSON] = None  # e.g., max dose warnings


class DiagnosisResult(BaseModel):
    verdict: str
    confidence: float = 0.5
    bottleneck: Optional[str] = None  # dose-limited, drift-limited, calibration-limited...
    evidence: JSON = Field(default_factory=dict)
    suggested_actions: List[Action] = Field(default_factory=list)


# -----------------------------
# Calib/Recon results for new mode
# -----------------------------

class CalibResult(BaseModel):
    operator_id: str
    theta_best: JSON = Field(default_factory=dict)
    best_score: float = 0.0
    num_evals: int = 0
    logs: List[JSON] = Field(default_factory=list)


class ReconResult(BaseModel):
    solver_id: str
    xhat_ref: str  # path within RunBundle or URI
    yhat_ref: Optional[str] = None
    metrics: JSON = Field(default_factory=dict)
    runtime_s: Optional[float] = None


class CalibReconResult(BaseModel):
    spec_id: str
    calib: Optional[CalibResult] = None
    recon: List[ReconResult] = Field(default_factory=list)
    diagnosis: Optional[DiagnosisResult] = None
    runbundle_path: Optional[str] = None
