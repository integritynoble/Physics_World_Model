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
    LLMSelectionResult,
    InterpolatedResult,
)
from .base import BaseAgent, AgentContext
from .registry import RegistryBuilder, RegistryKeyError

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
    "LLMSelectionResult",
    "InterpolatedResult",
    "BaseAgent",
    "AgentContext",
    "RegistryBuilder",
    "RegistryKeyError",
]
