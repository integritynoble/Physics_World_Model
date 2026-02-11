"""
pwm_core.agents.registry_schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pydantic v2 schemas for validating all YAML registry files in the PWM system.

Each top-level ``*FileYaml`` class corresponds to one YAML file under
``contrib/``.  All schemas use ``extra="forbid"`` (inherited from
``StrictBaseModel``) so that typos, deprecated fields, or rogue additions
are caught at load time rather than silently ignored.

Validated registries
--------------------
- ``modalities.yaml``   -> :class:`ModalitiesFileYaml`
- ``mismatch_db.yaml``  -> :class:`MismatchDbFileYaml`
- ``compression_db.yaml`` -> :class:`CompressionDbFileYaml`
- ``photon_db.yaml``    -> :class:`PhotonDbFileYaml`
- ``metrics_db.yaml``   -> :class:`MetricsDbFileYaml`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .contracts import StrictBaseModel

# ═══════════════════════════════════════════════════════════════════════════════
# modalities.yaml
# ═══════════════════════════════════════════════════════════════════════════════


class ElementYaml(StrictBaseModel):
    """One optical / physical element in the imaging chain.

    Parameters
    ----------
    name : str
        Human-readable element label (e.g. ``"coded_aperture"``).
    element_type : str
        Role in the system, e.g. ``"source"``, ``"lens"``, ``"mask"``,
        ``"modulator"``, ``"detector"``, ``"transducer"``.
    transfer_kind : str
        How the element transforms the signal, e.g.
        ``"attenuation"``, ``"phase_modulation"``, ``"dispersion"``.
    throughput : float
        Fraction of signal power transmitted (0.0 -- 1.0).
    noise_kinds : list[str]
        Noise mechanisms introduced by this element, e.g.
        ``["shot_noise", "read_noise"]``.
    parameters : dict[str, Any]
        Element-specific physical parameters (free-form).
    """

    name: str
    element_type: str
    transfer_kind: str
    throughput: float = Field(ge=0.0, le=1.0)
    noise_kinds: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class UploadFileYaml(StrictBaseModel):
    """Specification for one file expected in an upload bundle.

    Parameters
    ----------
    name : str
        Filename (e.g. ``"y.npy"``).
    description : str
        What the file contains.
    shape : str or None
        Expected tensor shape as a human-readable string, e.g.
        ``"[H+S-1, W]"``.  ``None`` when shape is unconstrained.
    """

    name: str
    description: str
    shape: Optional[str] = None


class UploadParamYaml(StrictBaseModel):
    """Specification for one scalar/structured parameter expected in an upload.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"dispersion_step"``).
    type : str
        Expected Python/JSON type as a human-readable string
        (e.g. ``"float"``, ``"[float, float]"``).
    description : str
        What the parameter controls.
    """

    name: str
    type: str
    description: str


class UploadTemplateYaml(StrictBaseModel):
    """Upload template describing what files and parameters a modality needs.

    Parameters
    ----------
    required : list[UploadFileYaml]
        Files that must be provided.
    required_params : list[UploadParamYaml]
        Scalar parameters that must be provided.
    optional : list[UploadFileYaml | UploadParamYaml]
        Additional files or parameters that are useful but not mandatory.
    """

    required: List[UploadFileYaml] = Field(default_factory=list)
    required_params: List[UploadParamYaml] = Field(default_factory=list)
    optional: List[Union[UploadFileYaml, UploadParamYaml]] = Field(
        default_factory=list
    )


class ModalityYaml(StrictBaseModel):
    """Schema for one modality entry inside ``modalities.yaml``.

    Parameters
    ----------
    display_name : str
        Human-friendly modality name shown in UI/CLI.
    category : str
        Broad grouping, e.g. ``"spectral"``, ``"microscopy"``,
        ``"tomography"``.
    keywords : list[str]
        Search keywords used by the LLM modality-selection step.
    description : str
        One-paragraph plain-English description of the modality.
    signal_dims : dict[str, list[int]]
        Mapping from signal label to its expected dimensionality,
        e.g. ``{"spectral_cube": [28, 256, 256]}``.
    forward_model_type : str
        Operator class, e.g. ``"linear_operator"``,
        ``"nonlinear_operator"``.
    forward_model_equation : str
        LaTeX or short-form equation, e.g. ``"y = M * diag(phi) * x + n"``.
    default_solver : str
        Registry key of the recommended reconstruction solver.
    elements : list[ElementYaml]
        Ordered list of optical/physical elements.
    wavelength_range_nm : list[float] or None
        ``[min_nm, max_nm]`` operating wavelength band, if applicable.
    upload_template : UploadTemplateYaml or None
        Upload specification for measured-data mode.
    """

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
    upload_template: Optional[UploadTemplateYaml] = None
    default_template_id: Optional[str] = None
    requires_x_interaction: Optional[bool] = None
    acceptance_tier: Optional[str] = None


class ModalitiesFileYaml(StrictBaseModel):
    """Top-level schema for ``modalities.yaml``.

    Parameters
    ----------
    version : str
        Schema version string (e.g. ``"1.0"``).
    modalities : dict[str, ModalityYaml]
        Mapping from modality key (e.g. ``"cassi"``) to its definition.
    """

    version: str
    modalities: Dict[str, ModalityYaml]


# ═══════════════════════════════════════════════════════════════════════════════
# mismatch_db.yaml
# ═══════════════════════════════════════════════════════════════════════════════


class MismatchParamYaml(StrictBaseModel):
    """One mismatch parameter and its expected error range.

    Parameters
    ----------
    range : list[float]
        ``[low, high]`` bounds for the parameter value.
    typical_error : float
        Representative error magnitude used for severity scoring.
    unit : str
        Physical unit (e.g. ``"pixels"``, ``"radians"``).
    description : str
        Human-readable explanation of this mismatch source.
    param_type : str or None
        Physics type classification.  Valid values:
        ``"spatial_shift"``, ``"rotation"``, ``"scale"``, ``"blur"``,
        ``"offset"``, ``"timing"``, ``"position"``.
    """

    range: List[float]
    typical_error: float
    unit: str
    description: str = ""
    param_type: Optional[str] = None


class MismatchModalityYaml(StrictBaseModel):
    """Mismatch specification for one modality.

    Parameters
    ----------
    parameters : dict[str, MismatchParamYaml]
        Named mismatch parameters (e.g. ``"dispersion_shift"``).
    severity_weights : dict[str, float]
        Per-parameter weights used to compute an aggregate mismatch
        severity score.
    correction_method : str
        Recommended correction strategy, e.g. ``"operator_refit"``,
        ``"admm_correction"``.
    """

    parameters: Dict[str, MismatchParamYaml]
    severity_weights: Dict[str, float]
    correction_method: str


class MismatchDbFileYaml(StrictBaseModel):
    """Top-level schema for ``mismatch_db.yaml``.

    Parameters
    ----------
    version : str
        Schema version string.
    modalities : dict[str, MismatchModalityYaml]
        Per-modality mismatch definitions keyed by modality key.
    """

    version: str
    modalities: Dict[str, MismatchModalityYaml]


# ═══════════════════════════════════════════════════════════════════════════════
# compression_db.yaml
# ═══════════════════════════════════════════════════════════════════════════════


class CalibrationProvenance(StrictBaseModel):
    """Provenance metadata for a calibration-table entry.

    Every calibration entry must carry provenance so that results are
    reproducible and auditable.

    Parameters
    ----------
    dataset_id : str
        Identifier of the dataset used for calibration (e.g.
        ``"kaist_28ch_v2"``).
    seed_set : list[int]
        Random seeds used during the calibration run.
    operator_version : str
        Git tag or hash of the operator code at calibration time.
    solver_version : str
        Git tag or hash of the solver code at calibration time.
    date_generated : str
        ISO-8601 date string (e.g. ``"2025-04-12"``).
    notes : str
        Free-form notes about the calibration run.
    """

    dataset_id: str
    seed_set: List[int]
    operator_version: str
    solver_version: str
    date_generated: str
    notes: str = ""


class CalibrationEntry(StrictBaseModel):
    """One row in a calibration table.

    Parameters
    ----------
    cr : float
        Compression ratio (must be non-negative).
    noise : str
        Noise regime label, e.g. ``"clean"``, ``"sigma_25"``,
        ``"poisson_peak_1e4"``.
    solver : str
        Solver registry key used for reconstruction.
    recoverability : float
        Predicted recoverability score (0.0 -- 1.0).
    expected_psnr_db : float
        Expected reconstruction PSNR in dB.
    provenance : CalibrationProvenance
        Provenance for this specific entry.
    """

    cr: float
    noise: str
    solver: str
    recoverability: float = Field(ge=0.0, le=1.0)
    expected_psnr_db: float
    provenance: CalibrationProvenance


class CalibrationModalityYaml(StrictBaseModel):
    """Calibration table for one modality.

    Parameters
    ----------
    signal_prior_class : str
        Signal class assumption, e.g. ``"natural_scene"``,
        ``"fluorescence_sparse"``.
    entries : list[CalibrationEntry]
        Ordered list of calibration measurements.
    """

    signal_prior_class: str
    entries: List[CalibrationEntry]


class CompressionDbFileYaml(StrictBaseModel):
    """Top-level schema for ``compression_db.yaml``.

    Parameters
    ----------
    version : str
        Schema version string.
    calibration_tables : dict[str, CalibrationModalityYaml]
        Per-modality calibration tables keyed by modality key.
    """

    version: str
    description: Optional[str] = None
    calibration_tables: Dict[str, CalibrationModalityYaml]


# ═══════════════════════════════════════════════════════════════════════════════
# photon_db.yaml
# ═══════════════════════════════════════════════════════════════════════════════


class PhotonLevelYaml(StrictBaseModel):
    """One photon-level entry (bright / standard / low_light).

    Parameters
    ----------
    n_photons : float or None
        Target photon count for photon-based modalities.
    snr_proxy : float or None
        SNR proxy for thermal-noise modalities (MRI).
    scenario : str
        Human-readable scenario description.
    read_sigma_fraction : float
        Read-noise sigma as a fraction of sqrt(n_photons).
    sigma : float or None
        Direct sigma for gaussian noise model.
    """

    n_photons: Optional[float] = None
    snr_proxy: Optional[float] = None
    scenario: str = ""
    read_sigma_fraction: float = 0.01
    sigma: Optional[float] = None


class PhotonModelYaml(StrictBaseModel):
    """Photon-budget model specification for one modality.

    The ``model_id`` is a dispatch key used by the photon agent to select
    the appropriate deterministic computation.  The actual formula is
    implemented in code, never eval'd from the YAML.

    Parameters
    ----------
    model_id : str
        Code-side dispatch key (e.g. ``"poisson_gaussian_mixed"``).
    parameters : dict[str, Any]
        Model-specific parameters (free-form).
    description : str
        Human-readable explanation of the photon model.
    noise_model : str or None
        Noise model override (``"poisson_gaussian"``, ``"poisson"``,
        ``"gaussian"``).
    photon_levels : dict[str, PhotonLevelYaml] or None
        Per-scenario photon levels (bright / standard / low_light).
    """

    model_id: str
    parameters: Dict[str, Any]
    description: str = ""
    noise_model: Optional[str] = None
    photon_levels: Optional[Dict[str, PhotonLevelYaml]] = None


class PhotonDbFileYaml(StrictBaseModel):
    """Top-level schema for ``photon_db.yaml``.

    Parameters
    ----------
    version : str
        Schema version string.
    modalities : dict[str, PhotonModelYaml]
        Per-modality photon model definitions keyed by modality key.
    """

    version: str
    modalities: Dict[str, PhotonModelYaml]


# ═══════════════════════════════════════════════════════════════════════════════
# metrics_db.yaml
# ═══════════════════════════════════════════════════════════════════════════════


class MetricSetYaml(StrictBaseModel):
    """A named collection of quality metrics applicable to specific modalities.

    Parameters
    ----------
    description : str
        What this metric set measures (e.g. ``"Spectral fidelity metrics"``).
    metrics : list[str]
        Metric identifiers (e.g. ``["psnr", "ssim", "sam"]``).
    modalities : list[str]
        Modality keys to which this metric set applies.
    """

    description: str
    metrics: List[str]
    modalities: List[str]


class MetricsDbFileYaml(StrictBaseModel):
    """Top-level schema for ``metrics_db.yaml``.

    Parameters
    ----------
    version : str
        Schema version string.
    metric_sets : dict[str, MetricSetYaml]
        Named metric sets keyed by set identifier.
    """

    version: str
    metric_sets: Dict[str, MetricSetYaml]
