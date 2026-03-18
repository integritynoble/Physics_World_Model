"""pwm_core.agents.system_discern

SystemDiscernAgent: user text description -> ImagingSystemSpec + candidate
graph_template_ids.

Design mantra
-------------
Must run without LLM and produce deterministic outputs.  The LLM is an
optional enhancement for semantic understanding; the deterministic path
uses keyword matching against registry modality descriptions and element
chains.

Inputs
------
- User natural-language description of their imaging system.
- Optional: metadata dict with extra context (wavelength, detector type,
  file paths, etc.).

Outputs
-------
- ``ImagingSystemSpec``: the best-matching ``ImagingSystem`` from the
  registry, with confidence bands.
- Top-K candidate ``graph_template_id`` s when the match is ambiguous.
- "What additional info would help" suggestions when confidence is low.

All outputs are registry IDs -- never freeform hallucinated strings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentContext, AgentResult
from .contracts import ImagingSystem
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SystemDiscernResult(AgentResult):
    """Result of the SystemDiscernAgent.

    Attributes
    ----------
    imaging_system : ImagingSystem or None
        Best-matching imaging system from the registry.
    candidate_template_ids : list[str]
        Top-K candidate graph template IDs (modality keys).  The first
        element is the primary match.
    confidence : float
        Confidence in the primary match (0.0 to 1.0).
    info_suggestions : list[str]
        Suggestions for additional information that would improve the
        match, when confidence is low.
    """

    imaging_system: Optional[Any] = None
    candidate_template_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    info_suggestions: List[str] = field(default_factory=list)

    def default_narrative(self) -> str:
        if not self.success:
            return f"SystemDiscernAgent failed: {self.error}"
        if not self.candidate_template_ids:
            return "No matching imaging modality found."
        primary = self.candidate_template_ids[0]
        n_cand = len(self.candidate_template_ids)
        conf_str = f"{self.confidence:.0%}"
        narrative = (
            f"Identified {primary} as the primary modality match "
            f"(confidence: {conf_str})"
        )
        if n_cand > 1:
            others = ", ".join(self.candidate_template_ids[1:])
            narrative += f", with {n_cand - 1} alternative(s): {others}"
        narrative += "."
        if self.info_suggestions:
            narrative += (
                " To improve confidence, provide: "
                + "; ".join(self.info_suggestions)
                + "."
            )
        return narrative


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SystemDiscernAgent(BaseAgent):
    """Convert user text description into an ImagingSystemSpec and
    candidate graph template IDs.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Optional LLM for semantic matching when keywords are ambiguous.
    registry : RegistryBuilder
        Source of truth for modality keys, element chains, and graph
        template definitions.
    top_k : int
        Maximum number of candidate template IDs to return.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        registry: Optional["RegistryBuilder"] = None,
        top_k: int = 3,
    ) -> None:
        super().__init__(llm_client=llm_client, registry=registry)
        self.top_k = top_k

    def run(self, context: AgentContext) -> SystemDiscernResult:
        """Execute the system discernment pipeline.

        ``context.plan_intent`` should be the user's natural-language
        description of their imaging system (str), or an object with a
        ``.user_prompt`` attribute.

        Parameters
        ----------
        context : AgentContext
            Must contain a description in ``plan_intent`` or
            ``modality_key``.

        Returns
        -------
        SystemDiscernResult
        """
        # -- Extract description text --------------------------------------
        description = ""
        if isinstance(context.plan_intent, str):
            description = context.plan_intent
        elif hasattr(context.plan_intent, "user_prompt"):
            description = context.plan_intent.user_prompt
        if not description:
            description = context.modality_key or ""

        if not description.strip():
            return SystemDiscernResult(
                success=False,
                error="No system description provided.",
            )

        # -- Keyword-based matching ----------------------------------------
        try:
            registry = self._require_registry()
        except RuntimeError as exc:
            return SystemDiscernResult(
                success=False,
                error=str(exc),
            )

        scored = self._keyword_score(description, registry)

        # -- Optional LLM refinement --------------------------------------
        if self.llm is not None and (
            not scored or scored[0][1] < 3
        ):
            llm_candidates = self._llm_score(description, registry)
            if llm_candidates:
                # Merge: LLM results get a bonus
                llm_dict = {k: s for k, s in llm_candidates}
                for key, score in scored:
                    if key not in llm_dict:
                        llm_dict[key] = score
                    else:
                        llm_dict[key] = max(llm_dict[key], score)
                scored = sorted(llm_dict.items(), key=lambda t: t[1], reverse=True)

        if not scored:
            return SystemDiscernResult(
                success=False,
                error=(
                    f"Could not match description to any modality. "
                    f"Available: {registry.list_modalities()}"
                ),
                info_suggestions=self._suggest_info(description),
            )

        # -- Select top-K candidates ---------------------------------------
        candidates = [key for key, _score in scored[: self.top_k]]
        primary_key = candidates[0]
        primary_score = scored[0][1]

        # -- Confidence estimation ----------------------------------------
        total_kw = 1
        try:
            mod = registry.get_modality(primary_key)
            total_kw = max(len(mod.keywords), 1)
        except Exception:
            pass
        confidence = min(primary_score / max(total_kw, 1), 1.0)
        confidence = max(confidence, 0.1)

        # -- Build ImagingSystem from registry ----------------------------
        imaging_system: Optional[ImagingSystem] = None
        try:
            from .plan_agent import PlanAgent

            builder = PlanAgent(
                llm_client=self.llm, registry=registry
            )
            imaging_system = builder.build_imaging_system(primary_key)
        except Exception as exc:
            logger.warning(
                "Could not build ImagingSystem for '%s': %s",
                primary_key,
                exc,
            )

        # -- Info suggestions for low confidence --------------------------
        info_suggestions: List[str] = []
        if confidence < 0.5:
            info_suggestions = self._suggest_info(description)

        return SystemDiscernResult(
            success=True,
            imaging_system=imaging_system,
            candidate_template_ids=candidates,
            confidence=round(confidence, 3),
            info_suggestions=info_suggestions,
            raw_data={
                "description": description,
                "scored": scored[: self.top_k * 2],
            },
        )

    # ===================================================================
    # Keyword scoring
    # ===================================================================

    def _keyword_score(
        self,
        description: str,
        registry: "RegistryBuilder",
    ) -> List[tuple]:
        """Score each modality by keyword overlap with the description.

        Returns a list of ``(modality_key, score)`` sorted descending.
        """
        desc_lower = description.lower()
        desc_tokens = set(re.split(r"[\s,;:.\-_/()]+", desc_lower))

        results: List[tuple] = []
        for mod_key in registry.list_modalities():
            try:
                mod = registry.get_modality(mod_key)
            except Exception:
                continue

            score = 0
            # Check modality key itself
            mk_lower = mod_key.lower()
            if mk_lower in desc_tokens or (
                len(mk_lower) > 3 and mk_lower in desc_lower
            ):
                score += 3

            # Check keywords
            for kw in mod.keywords:
                kw_lower = kw.lower()
                if kw_lower in desc_tokens:
                    score += 1
                elif len(kw_lower) > 3 and kw_lower in desc_lower:
                    score += 1

            # Check display name
            display_lower = mod.display_name.lower()
            for token in display_lower.split():
                if len(token) > 3 and token in desc_lower:
                    score += 1

            if score > 0:
                results.append((mod_key, score))

        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # ===================================================================
    # LLM scoring (optional enhancement)
    # ===================================================================

    def _llm_score(
        self,
        description: str,
        registry: "RegistryBuilder",
    ) -> List[tuple]:
        """Use the LLM to select the best modality match.

        Returns a list of ``(modality_key, score)`` with LLM-assigned
        scores.
        """
        if self.llm is None:
            return []

        modality_keys = registry.list_modalities()
        if not modality_keys:
            return []

        try:
            result = self.llm.select(
                (
                    f"The user describes their imaging system as:\n"
                    f'"{description}"\n\n'
                    f"Which imaging modality does this correspond to?"
                ),
                modality_keys,
            )
            selected = result.get("selected", "")
            if selected and selected in modality_keys:
                return [(selected, 5)]
        except Exception as exc:
            logger.warning("LLM scoring failed: %s", exc)

        return []

    # ===================================================================
    # Info suggestions
    # ===================================================================

    @staticmethod
    def _suggest_info(description: str) -> List[str]:
        """Suggest what additional information would help identify the
        system, based on what is missing from the description."""
        suggestions: List[str] = []
        desc_lower = description.lower()

        if not any(
            w in desc_lower
            for w in ("wavelength", "nm", "spectral", "broadband", "narrowband")
        ):
            suggestions.append(
                "Wavelength or spectral range (e.g. 450-650 nm, broadband visible)"
            )

        if not any(
            w in desc_lower
            for w in ("detector", "sensor", "camera", "ccd", "cmos", "spad")
        ):
            suggestions.append(
                "Detector type (e.g. CCD, CMOS, SPAD, spectrometer)"
            )

        if not any(
            w in desc_lower
            for w in ("coded", "mask", "aperture", "dmd", "slm", "grating")
        ):
            suggestions.append(
                "Optical encoding element (e.g. coded aperture, DMD, SLM, grating)"
            )

        if not any(
            w in desc_lower
            for w in ("resolution", "pixel", "spatial", "dimension", "size")
        ):
            suggestions.append(
                "Spatial resolution or image dimensions"
            )

        if not suggestions:
            suggestions.append(
                "More specific modality name or technique description"
            )

        return suggestions
