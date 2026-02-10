"""PWM Agent System — strict pydantic contracts, deterministic computation, optional LLM.

Design mantra: agents must run without LLM and produce deterministic outputs.
The LLM is an optional enhancement for priors + narrative.
"""

from .contracts import (
    StrictBaseModel,
    PlanIntent,
    ModeRequested,
    ModalitySelection,
    ImagingSystem,
    ElementSpec,
    PhotonReport,
    MismatchReport,
    RecoverabilityReport,
    SystemAnalysis,
    PreFlightReport,
    BottleneckScores,
    Suggestion,
    NoiseRegime,
    SignalPriorClass,
    TransferKind,
    NoiseKind,
    ForwardModelType,
    CorrectionResult,
    LLMSelectionResult,
    InterpolatedResult,
)
from .base import BaseAgent, AgentContext
from .registry import RegistryBuilder, RegistryKeyError
from .upwmi import (
    UPWMIBudget,
    ScoringWeights,
    upwmi_score,
    residual_features,
    CandidateCache,
    ActiveLearningSearch,
)
from .self_improvement import DesignAlternative, SelfImprovementLoop
from .what_if_precomputer import (
    SensitivityPoint,
    SensitivityCurve,
    WhatIfPrecomputer,
)
from .asset_manager import (
    AssetLicense,
    AssetEntry,
    AssetManifest,
    AssetManager,
)
from .hybrid import (
    HybridModalitySpec,
    FusionResult,
    HybridModalityManager,
)

__all__ = [
    "StrictBaseModel",
    "PlanIntent",
    "ModeRequested",
    "ModalitySelection",
    "ImagingSystem",
    "ElementSpec",
    "PhotonReport",
    "MismatchReport",
    "RecoverabilityReport",
    "SystemAnalysis",
    "PreFlightReport",
    "BottleneckScores",
    "Suggestion",
    "NoiseRegime",
    "SignalPriorClass",
    "TransferKind",
    "NoiseKind",
    "ForwardModelType",
    "CorrectionResult",
    "LLMSelectionResult",
    "InterpolatedResult",
    "BaseAgent",
    "AgentContext",
    "RegistryBuilder",
    "RegistryKeyError",
    "UPWMIBudget",
    "ScoringWeights",
    "upwmi_score",
    "residual_features",
    "CandidateCache",
    "ActiveLearningSearch",
    "DesignAlternative",
    "SelfImprovementLoop",
    "SensitivityPoint",
    "SensitivityCurve",
    "WhatIfPrecomputer",
    "AssetLicense",
    "AssetEntry",
    "AssetManifest",
    "AssetManager",
    "HybridModalitySpec",
    "FusionResult",
    "HybridModalityManager",
]
