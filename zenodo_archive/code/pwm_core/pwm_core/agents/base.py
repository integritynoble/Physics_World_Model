"""pwm_core.agents.base

BaseAgent abstract class and shared data structures for the PWM agent system.

Design mantra
-------------
Agents must run without LLM and produce deterministic outputs.  The LLM is an
optional enhancement layer used solely for narrative generation and prior
elicitation; every agent's ``run()`` path must succeed with ``llm_client=None``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from .llm_client import LLMClient

if TYPE_CHECKING:
    from .registry import RegistryBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared context passed between agents
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Immutable snapshot of everything an agent may need to make decisions.

    ``modality_key`` is the only required field -- it anchors every look-up
    into the registry YAML files.  All other fields are progressively filled
    in as the pipeline advances through successive agents.

    Parameters
    ----------
    modality_key : str
        Registry key for the imaging modality (e.g. ``"cassi"``, ``"sim"``).
    imaging_system : Any, optional
        Physics operator or system descriptor produced by
        ``PhysicsFactory.build()``.
    budget : dict, optional
        Resource / photon budget dictionary (keys vary by modality).
    photon_report : Any, optional
        Output of the ``PhotonAgent``.
    mismatch_report : Any, optional
        Output of the ``MismatchAgent``.
    recoverability_report : Any, optional
        Output of the ``RecoverabilityAgent``.
    recon_result : Any, optional
        ``ReconResult`` or similar from a solver run.
    plan_intent : Any, optional
        High-level plan produced by the ``PlanAgent``.
    solver_family : str, optional
        Registry key for the solver family (e.g. ``"mst"``, ``"gap_tv"``).
    """

    modality_key: str

    imaging_system: Any = None
    budget: Optional[Dict[str, Any]] = field(default_factory=dict)
    photon_report: Any = None
    mismatch_report: Any = None
    recoverability_report: Any = None
    recon_result: Any = None
    plan_intent: Any = None
    solver_family: Optional[str] = None


# ---------------------------------------------------------------------------
# Base result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Minimal result envelope returned by every agent.

    Subclasses should add domain-specific fields (metrics, reports, etc.)
    and override ``default_narrative`` to provide a human-readable summary.

    Parameters
    ----------
    success : bool
        Whether the agent completed without error.
    error : str, optional
        Human-readable error description if ``success`` is ``False``.
    raw_data : dict, optional
        Arbitrary structured data the agent wishes to expose for
        downstream inspection or serialisation.
    """

    success: bool
    error: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def default_narrative(self) -> str:
        """Return a deterministic, LLM-free summary of this result.

        Subclasses should override this to produce modality-aware text.
        The base implementation falls back to a minimal status string.
        """
        if self.success:
            return "Agent completed successfully."
        return f"Agent failed: {self.error}"


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for all PWM agents.

    Design mantra: **Agents must run without LLM and produce deterministic
    outputs.**  The ``llm_client`` is an *optional* enhancement used for
    narrative generation or soft prior elicitation; the ``run()`` method must
    never raise when ``llm_client is None``.

    Parameters
    ----------
    llm_client : LLMClient, optional
        If provided, agents may use it for narrative / prior enrichment.
    registry : RegistryBuilder
        Source of truth for all modality, solver, and calibration metadata.
        Required by most agents; use ``_require_registry`` to guard access.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        registry: Optional["RegistryBuilder"] = None,
    ) -> None:
        self.llm = llm_client
        self.registry = registry

    # ----- abstract interface ------------------------------------------------

    @abstractmethod
    def run(self, context: AgentContext) -> Any:
        """Execute the agent logic and return a result.

        Implementations **must** succeed when ``self.llm is None``.

        Parameters
        ----------
        context : AgentContext
            Shared pipeline state.

        Returns
        -------
        Any
            Typically an ``AgentResult`` subclass, but callers should
            rely on duck-typing rather than a concrete type.
        """
        ...

    # ----- narrative helper --------------------------------------------------

    def explain(self, result: Any) -> str:
        """Produce a human-readable explanation of *result*.

        Strategy:
        1. If an LLM client is available, ask it to narrate the result.
        2. Otherwise, call ``result.default_narrative()`` if the method
           exists.
        3. Fall back to ``str(result)``.

        Parameters
        ----------
        result : Any
            The object returned by ``run()``.

        Returns
        -------
        str
            A human-readable narrative.
        """
        if self.llm is not None:
            try:
                prompt = (
                    f"Summarise the following computational-imaging agent "
                    f"result in 2-3 plain-English sentences:\n\n{result}"
                )
                return self.llm.generate(prompt)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "LLM narrative generation failed; falling back to "
                    "deterministic summary.",
                    exc_info=True,
                )

        # Deterministic path -- no LLM needed.
        if hasattr(result, "default_narrative") and callable(
            result.default_narrative
        ):
            return result.default_narrative()

        return str(result)

    # ----- internal helpers --------------------------------------------------

    def _require_registry(self) -> "RegistryBuilder":
        """Return ``self.registry``, raising if it was not provided.

        Raises
        ------
        RuntimeError
            If the agent was instantiated without a registry.
        """
        if self.registry is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a RegistryBuilder "
                f"instance but was initialised with registry=None."
            )
        return self.registry
