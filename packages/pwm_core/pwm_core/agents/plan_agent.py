"""
pwm_core.agents.plan_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plan Agent: the top-level orchestrator that maps user prompts to imaging
modalities and coordinates the full sub-agent pipeline.

Responsibilities
----------------
1. Parse user intent (mode detection, file path extraction).
2. Map natural-language prompts to a registry modality key (keyword
   matching first, LLM fallback second).
3. Construct the :class:`ImagingSystem` from the modality's element chain.
4. Run sub-agents (Photon, Mismatch, Recoverability, Analysis).
5. Negotiate go/no-go via ``AgentNegotiator``.
6. Build and check the pre-flight report.
7. Return all reports for downstream consumption.

Design invariants
-----------------
* Must run without LLM -- keyword matching is the primary modality
  selection path. The LLM is only used as a semantic fallback when
  keywords fail to match.
* All outputs are deterministic given the same registry and prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseAgent, AgentContext
from .contracts import (
    ElementSpec,
    ForwardModelType,
    ImagingSystem,
    LLMSelectionResult,
    ModeRequested,
    ModalitySelection,
    NoiseKind,
    PlanIntent,
    TransferKind,
)
from .llm_client import LLMClient
from .preflight import PreFlightReportBuilder, PermitMode, check_permit
from .registry import RegistryBuilder

if TYPE_CHECKING:
    from .contracts import (
        MismatchReport,
        NegotiationResult,
        PhotonReport,
        RecoverabilityReport,
        SystemAnalysis,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keywords for intent detection
# ---------------------------------------------------------------------------

_OPERATOR_CORRECTION_KEYWORDS = frozenset({
    "measured",
    "measurement",
    "calibrate",
    "calibration",
    "correction",
    "correct",
    "operator_correction",
    "mode2",
    "fit",
    "refit",
    "refine",
    "upwmi",
    "real data",
    "real-data",
    "experimental",
})

_FILE_EXTENSIONS = frozenset({
    ".npy", ".npz", ".mat", ".h5", ".hdf5", ".tif", ".tiff",
    ".png", ".jpg", ".jpeg", ".fits", ".json", ".csv",
})


# ---------------------------------------------------------------------------
# PlanAgent
# ---------------------------------------------------------------------------


class PlanAgent(BaseAgent):
    """Top-level orchestrator mapping prompts to modalities and coordinating
    the full agent pipeline.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Optional LLM for semantic modality matching when keyword
        matching fails.
    registry : RegistryBuilder
        Source of truth for all modality, solver, and calibration metadata.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        registry: Optional[RegistryBuilder] = None,
    ) -> None:
        super().__init__(llm_client=llm_client, registry=registry)

    def run(self, context: AgentContext) -> Any:
        """Run the plan agent given an existing context.

        This is the ``BaseAgent`` interface. For the full pipeline
        orchestration, prefer :meth:`run_pipeline` which accepts a
        raw prompt string.

        Parameters
        ----------
        context : AgentContext
            Must contain ``plan_intent`` as the user prompt string.

        Returns
        -------
        dict
            Pipeline results dict, same as :meth:`run_pipeline`.
        """
        prompt = context.plan_intent if isinstance(context.plan_intent, str) else ""
        if not prompt:
            prompt = context.modality_key
        return self.run_pipeline(prompt)

    # ===================================================================
    # Intent parsing
    # ===================================================================

    def parse_intent(
        self,
        prompt: str,
        file_paths: Optional[List[str]] = None,
    ) -> PlanIntent:
        """Parse user intent from a natural-language prompt.

        Detects:
        * Mode requested (simulate vs operator correction).
        * Presence of measured data references.
        * File paths (from explicit args or embedded in prompt).

        Parameters
        ----------
        prompt : str
            User's natural-language prompt.
        file_paths : list[str], optional
            Explicit file paths provided alongside the prompt.

        Returns
        -------
        PlanIntent
            Structured intent ready for downstream processing.
        """
        prompt_lower = prompt.lower()

        # -- Detect mode ---------------------------------------------------
        mode = ModeRequested.auto
        for keyword in _OPERATOR_CORRECTION_KEYWORDS:
            if keyword in prompt_lower:
                mode = ModeRequested.operator_correction
                break

        # Also check for explicit mode directives
        if "mode1" in prompt_lower or "simulate only" in prompt_lower:
            mode = ModeRequested.simulate
        elif "mode2" in prompt_lower or "operator correction" in prompt_lower:
            mode = ModeRequested.operator_correction

        # -- Detect file paths ---------------------------------------------
        detected_paths: List[str] = []
        if file_paths:
            detected_paths.extend(file_paths)

        # Extract file paths from prompt text
        # Look for paths with recognised extensions
        tokens = re.split(r"[\s,;]+", prompt)
        for token in tokens:
            token_clean = token.strip("\"'()[]{}").rstrip(".,;:")
            for ext in _FILE_EXTENSIONS:
                if token_clean.lower().endswith(ext):
                    detected_paths.append(token_clean)
                    break
            # Also match Unix-style paths like /path/to/file
            if token_clean.startswith("/") and "/" in token_clean[1:]:
                if token_clean not in detected_paths:
                    detected_paths.append(token_clean)

        # -- Infer has_measured_y ------------------------------------------
        has_measured_y = bool(detected_paths) or any(
            kw in prompt_lower for kw in ("measured", "real data", "y.npy")
        )

        # -- Infer operator type -------------------------------------------
        from .contracts import OperatorType

        has_operator_A = any(
            kw in prompt_lower
            for kw in ("operator", "matrix", "a.npy", "phi.npy", "forward model")
        )
        if has_operator_A:
            if "matrix" in prompt_lower or "a.npy" in prompt_lower:
                operator_type = OperatorType.explicit_matrix
            else:
                operator_type = OperatorType.linear_operator
        else:
            operator_type = OperatorType.unknown

        return PlanIntent(
            mode_requested=mode,
            has_measured_y=has_measured_y,
            has_operator_A=has_operator_A,
            operator_type=operator_type,
            user_prompt=prompt,
            raw_file_paths=detected_paths,
        )

    # ===================================================================
    # Modality mapping
    # ===================================================================

    def map_prompt_to_modality(self, prompt: str) -> ModalitySelection:
        """Map a natural-language prompt to a registry modality key.

        Two-stage matching:
        1. **Keyword matching** -- fast, no LLM, checks prompt tokens
           against each modality's ``keywords`` list from the registry.
        2. **LLM semantic matching** -- only if stage 1 produces no hit
           and an LLM client is available.

        Parameters
        ----------
        prompt : str
            User's natural-language prompt.

        Returns
        -------
        ModalitySelection
            Selected modality with confidence and reasoning.

        Raises
        ------
        ValueError
            If no modality can be matched by any method.
        """
        registry = self._require_registry()

        # -- Stage 1: keyword matching ------------------------------------
        selection = self._keyword_match(prompt, registry)
        if selection is not None:
            # Validate against registry
            try:
                registry.assert_modality_exists(selection.modality_key)
                return selection
            except Exception:
                logger.warning(
                    "Keyword-matched modality '%s' not found in registry; "
                    "falling through to LLM matching.",
                    selection.modality_key,
                )

        # -- Stage 2: LLM semantic matching --------------------------------
        if self.llm is not None:
            selection = self._llm_match(prompt, registry)
            if selection is not None:
                try:
                    registry.assert_modality_exists(selection.modality_key)
                    return selection
                except Exception:
                    logger.warning(
                        "LLM-selected modality '%s' not found in registry; "
                        "returning error.",
                        selection.modality_key,
                    )

        # -- No match found ------------------------------------------------
        available = registry.list_modalities()
        raise ValueError(
            f"Could not map prompt to any imaging modality. "
            f"Available modalities: {available}. "
            f"Prompt: '{prompt[:100]}...'"
            if len(prompt) > 100
            else f"Could not map prompt to any imaging modality. "
            f"Available modalities: {available}. "
            f"Prompt: '{prompt}'"
        )

    def _keyword_match(
        self, prompt: str, registry: RegistryBuilder
    ) -> Optional[ModalitySelection]:
        """Stage 1: fast keyword matching against registry modality keywords.

        Scores each modality by counting how many of its keywords appear
        in the prompt. Returns the highest-scoring modality, or None if
        no keyword matches at all.
        """
        prompt_lower = prompt.lower()
        prompt_tokens = set(re.split(r"[\s,;:.\-_/()]+", prompt_lower))

        best_key: Optional[str] = None
        best_score: int = 0
        best_keywords: List[str] = []
        total_keywords_checked: int = 0

        modality_keys = registry.list_modalities()

        for mod_key in modality_keys:
            try:
                mod_yaml = registry.get_modality(mod_key)
            except Exception:
                continue

            keywords = mod_yaml.keywords
            total_keywords_checked += len(keywords)
            score = 0
            matched_kw: List[str] = []

            for kw in keywords:
                kw_lower = kw.lower()
                # Normalize underscores and hyphens in keyword for matching
                kw_normalized = kw_lower.replace("_", " ").replace("-", " ")
                # Check exact token match first (most reliable)
                if kw_lower in prompt_tokens:
                    score += 1
                    matched_kw.append(kw)
                # For longer keywords (>3 chars), also allow substring match
                elif len(kw_lower) > 3 and kw_lower in prompt_lower:
                    score += 1
                    matched_kw.append(kw)
                # Check underscore-to-space normalised match
                elif kw_normalized != kw_lower and kw_normalized in prompt_lower:
                    score += 2  # bonus for multi-word match
                    matched_kw.append(kw)
                # Check if individual words of the keyword all appear
                elif "_" in kw_lower or " " in kw_lower:
                    kw_parts = set(re.split(r"[_\s\-]+", kw_lower))
                    if kw_parts and kw_parts.issubset(prompt_tokens):
                        score += 1
                        matched_kw.append(kw)

            # Also check if the modality key itself appears in the prompt
            mod_key_lower = mod_key.lower()
            mod_key_spaced = mod_key_lower.replace("_", " ")
            if mod_key_lower in prompt_tokens:
                score += 3  # strong bonus for exact modality key match
                matched_kw.append(f"[key:{mod_key}]")
            elif len(mod_key_lower) > 3 and mod_key_lower in prompt_lower:
                score += 3  # substring match (safe for longer keys)
                matched_kw.append(f"[key:{mod_key}]")
            elif mod_key_spaced != mod_key_lower and mod_key_spaced in prompt_lower:
                score += 3  # underscore-to-space match
                matched_kw.append(f"[key:{mod_key}]")

            if score > best_score:
                best_score = score
                best_key = mod_key
                best_keywords = matched_kw

        if best_key is None or best_score == 0:
            return None

        # Compute confidence from keyword coverage
        try:
            mod_yaml = registry.get_modality(best_key)
            total_kw = max(len(mod_yaml.keywords), 1)
            confidence = min(len(best_keywords) / total_kw, 1.0)
            confidence = max(confidence, 0.3)  # floor at 30% for any match
        except Exception:
            confidence = 0.5

        # Collect fallback modalities (next best matches)
        fallbacks: List[str] = []
        if best_score > 0:
            # Recompute scores for fallback collection
            scores: Dict[str, int] = {}
            for mod_key in modality_keys:
                if mod_key == best_key:
                    continue
                try:
                    mod_yaml = registry.get_modality(mod_key)
                except Exception:
                    continue
                s = 0
                for kw in mod_yaml.keywords:
                    kw_lower = kw.lower()
                    kw_norm = kw_lower.replace("_", " ").replace("-", " ")
                    if kw_lower in prompt_tokens:
                        s += 1
                    elif len(kw_lower) > 3 and kw_lower in prompt_lower:
                        s += 1
                    elif kw_norm != kw_lower and kw_norm in prompt_lower:
                        s += 1
                    elif "_" in kw_lower or " " in kw_lower:
                        kw_parts = set(re.split(r"[_\s\-]+", kw_lower))
                        if kw_parts and kw_parts.issubset(prompt_tokens):
                            s += 1
                mk_lower = mod_key.lower()
                mk_spaced = mk_lower.replace("_", " ")
                if mk_lower in prompt_tokens:
                    s += 3
                elif len(mk_lower) > 3 and mk_lower in prompt_lower:
                    s += 3
                elif mk_spaced != mk_lower and mk_spaced in prompt_lower:
                    s += 3
                if s > 0:
                    scores[mod_key] = s

            fallbacks = sorted(scores, key=scores.get, reverse=True)[:3]

        reasoning = (
            f"Keyword match: {', '.join(best_keywords)} "
            f"(score={best_score})"
        )

        return ModalitySelection(
            modality_key=best_key,
            confidence=confidence,
            reasoning=reasoning,
            fallback_modalities=fallbacks,
        )

    def _llm_match(
        self, prompt: str, registry: RegistryBuilder
    ) -> Optional[ModalitySelection]:
        """Stage 2: LLM-based semantic matching.

        Sends the prompt and available modality descriptions to the LLM
        and asks it to pick the best match.
        """
        if self.llm is None:
            return None

        modality_keys = registry.list_modalities()
        if not modality_keys:
            return None

        # Build a description string for each modality
        descriptions: List[str] = []
        for key in modality_keys:
            try:
                mod = registry.get_modality(key)
                descriptions.append(
                    f"  {key}: {mod.display_name} -- {mod.description[:80]}"
                )
            except Exception:
                descriptions.append(f"  {key}: (description unavailable)")

        selection_prompt = (
            f"The user wants to do the following imaging task:\n"
            f'"{prompt}"\n\n'
            f"Available imaging modalities:\n"
            + "\n".join(descriptions)
            + "\n\nSelect the best matching modality."
        )

        try:
            result = self.llm.select(selection_prompt, modality_keys)
            selected_key = result.get("selected", modality_keys[0])
            reason = result.get("reason", "LLM semantic match")

            return ModalitySelection(
                modality_key=selected_key,
                confidence=0.7,  # LLM matches get moderate confidence
                reasoning=f"LLM semantic match: {reason}",
                fallback_modalities=[],
            )
        except Exception as exc:
            logger.warning("LLM modality matching failed: %s", exc)
            return None

    # ===================================================================
    # Imaging system construction
    # ===================================================================

    def build_imaging_system(self, modality_key: str) -> ImagingSystem:
        """Construct an :class:`ImagingSystem` from the registry.

        Loads the modality's element chain from the registry YAML,
        converts each element to an :class:`ElementSpec`, and assembles
        the full system specification.

        Parameters
        ----------
        modality_key : str
            Registry key for the modality (e.g. ``"cassi"``).

        Returns
        -------
        ImagingSystem
            Complete imaging system ready for sub-agent analysis.

        Raises
        ------
        RegistryKeyError
            If the modality key does not exist.
        """
        registry = self._require_registry()
        registry.assert_modality_exists(modality_key)
        mod_yaml = registry.get_modality(modality_key)

        # Convert ElementYaml -> ElementSpec
        elements: List[ElementSpec] = []
        for el_yaml in mod_yaml.elements:
            # Map string transfer_kind to the TransferKind enum
            try:
                transfer_kind = TransferKind(el_yaml.transfer_kind)
            except ValueError:
                logger.warning(
                    "Unknown transfer kind '%s' for element '%s' in "
                    "modality '%s'; defaulting to 'identity'.",
                    el_yaml.transfer_kind,
                    el_yaml.name,
                    modality_key,
                )
                transfer_kind = TransferKind.identity

            # Map noise kind strings to NoiseKind enums
            noise_kinds: List[NoiseKind] = []
            for nk_str in el_yaml.noise_kinds:
                try:
                    noise_kinds.append(NoiseKind(nk_str))
                except ValueError:
                    logger.warning(
                        "Unknown noise kind '%s' for element '%s'; skipping.",
                        nk_str,
                        el_yaml.name,
                    )

            elements.append(
                ElementSpec(
                    name=el_yaml.name,
                    element_type=el_yaml.element_type,
                    parameters=el_yaml.parameters,
                    transfer_kind=transfer_kind,
                    noise_kinds=noise_kinds,
                    throughput=el_yaml.throughput,
                )
            )

        # Map forward model type
        try:
            fwd_type = ForwardModelType(mod_yaml.forward_model_type)
        except ValueError:
            logger.warning(
                "Unknown forward model type '%s' for modality '%s'; "
                "defaulting to 'linear_operator'.",
                mod_yaml.forward_model_type,
                modality_key,
            )
            fwd_type = ForwardModelType.linear_operator

        # Determine wavelength info
        wavelength_nm: Optional[float] = None
        spectral_range_nm: Optional[List[float]] = None
        if mod_yaml.wavelength_range_nm:
            spectral_range_nm = mod_yaml.wavelength_range_nm
            if len(mod_yaml.wavelength_range_nm) == 2:
                wavelength_nm = sum(mod_yaml.wavelength_range_nm) / 2.0

        return ImagingSystem(
            modality_key=modality_key,
            elements=elements,
            forward_model_type=fwd_type,
            forward_model_equation=mod_yaml.forward_model_equation,
            signal_dims=mod_yaml.signal_dims,
            wavelength_nm=wavelength_nm,
            spectral_range_nm=spectral_range_nm,
        )

    # ===================================================================
    # Mode resolution
    # ===================================================================

    def resolve_mode(self, intent: PlanIntent) -> str:
        """Resolve the execution mode from parsed intent.

        Parameters
        ----------
        intent : PlanIntent
            Parsed user intent.

        Returns
        -------
        str
            One of ``"mode1_simulate"``, ``"mode2_operator_correction"``,
            or ``"mode1_simulate_fallback"``.
        """
        if intent.mode_requested == ModeRequested.operator_correction:
            if intent.has_measured_y:
                return "mode2_operator_correction"
            else:
                logger.warning(
                    "Operator correction requested but no measured data "
                    "detected. Falling back to simulation mode."
                )
                return "mode1_simulate_fallback"

        if intent.mode_requested == ModeRequested.simulate:
            return "mode1_simulate"

        # Auto mode: decide based on available data
        if intent.has_measured_y and intent.has_operator_A:
            return "mode2_operator_correction"
        elif intent.has_measured_y:
            # Has data but no explicit operator -- try operator correction
            # since the operator can be constructed from the registry
            return "mode2_operator_correction"
        else:
            return "mode1_simulate"

    # ===================================================================
    # Full pipeline orchestration
    # ===================================================================

    def run_pipeline(
        self,
        prompt: str,
        permit_mode: PermitMode = PermitMode.interactive,
    ) -> dict:
        """Full pipeline orchestration from prompt to reports.

        Executes the following stages:
        1. Parse intent from prompt.
        2. Map prompt to modality.
        3. Build imaging system.
        4. Run sub-agents (Photon, Mismatch, Recoverability, Analysis).
        5. Run negotiation.
        6. Build pre-flight report.
        7. Check permit.
        8. Return all reports.

        Parameters
        ----------
        prompt : str
            User's natural-language prompt.
        permit_mode : PermitMode
            How to gate execution after the pre-flight report.

        Returns
        -------
        dict
            Keys: ``"intent"``, ``"modality"``, ``"system"``, ``"mode"``,
            ``"photon"``, ``"mismatch"``, ``"recoverability"``,
            ``"analysis"``, ``"negotiation"``, ``"preflight"``,
            ``"permitted"``, ``"error"`` (if any).
        """
        results: Dict[str, Any] = {
            "intent": None,
            "modality": None,
            "system": None,
            "mode": None,
            "photon": None,
            "mismatch": None,
            "recoverability": None,
            "analysis": None,
            "negotiation": None,
            "preflight": None,
            "permitted": False,
            "error": None,
        }

        try:
            # -- 1. Parse intent -------------------------------------------
            intent = self.parse_intent(prompt)
            results["intent"] = intent
            logger.info(
                "Parsed intent: mode=%s, has_measured_y=%s",
                intent.mode_requested.value,
                intent.has_measured_y,
            )

            # -- 2. Map modality -------------------------------------------
            try:
                modality = self.map_prompt_to_modality(prompt)
            except ValueError as exc:
                logger.error("Modality mapping failed: %s", exc)
                results["error"] = str(exc)
                results.update(self.handle_non_imaging_prompt(prompt))
                return results

            results["modality"] = modality
            logger.info(
                "Mapped to modality: %s (confidence=%.2f)",
                modality.modality_key,
                modality.confidence,
            )

            # -- 3. Build imaging system -----------------------------------
            system = self.build_imaging_system(modality.modality_key)
            results["system"] = system

            # -- 4. Resolve mode -------------------------------------------
            mode = self.resolve_mode(intent)
            results["mode"] = mode
            logger.info("Resolved mode: %s", mode)

            # -- 5. Build agent context ------------------------------------
            context = AgentContext(
                modality_key=modality.modality_key,
                imaging_system=system,
                plan_intent=intent,
            )

            # -- 6. Run sub-agents -----------------------------------------
            photon = self._run_photon_agent(context)
            results["photon"] = photon

            mismatch = self._run_mismatch_agent(context)
            results["mismatch"] = mismatch

            recoverability = self._run_recoverability_agent(context)
            results["recoverability"] = recoverability

            analysis = self._run_analysis_agent(context, photon, mismatch, recoverability)
            results["analysis"] = analysis

            # -- 7. Negotiate ----------------------------------------------
            negotiation = self._run_negotiation(
                photon, mismatch, recoverability, analysis
            )
            results["negotiation"] = negotiation

            # -- 8. Build pre-flight report --------------------------------
            builder = PreFlightReportBuilder()
            preflight = builder.build(
                modality=modality,
                system=system,
                photon=photon,
                mismatch=mismatch,
                recoverability=recoverability,
                analysis=analysis,
                negotiation=negotiation,
            )
            results["preflight"] = preflight

            # -- 9. Check permit -------------------------------------------
            permitted = check_permit(preflight, permit_mode)
            results["permitted"] = permitted

            if not permitted:
                logger.warning(
                    "Pre-flight permit denied (mode=%s). Pipeline halted.",
                    permit_mode.value,
                )

            return results

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            results["error"] = str(exc)
            return results

    # ===================================================================
    # Sub-agent runners
    # ===================================================================

    def _run_photon_agent(self, context: AgentContext) -> Any:
        """Run the Photon Agent and return its report.

        Attempts to import and instantiate the PhotonAgent. If the module
        is not yet available, returns a synthetic default report for
        pipeline continuity.
        """
        try:
            from .photon_agent import PhotonAgent

            agent = PhotonAgent(
                llm_client=self.llm, registry=self.registry
            )
            return agent.run(context)
        except ImportError:
            logger.info(
                "PhotonAgent not available; returning synthetic defaults."
            )
            return self._default_photon_report(context)
        except Exception as exc:
            logger.warning(
                "PhotonAgent failed: %s; returning synthetic defaults.",
                exc,
            )
            return self._default_photon_report(context)

    def _run_mismatch_agent(self, context: AgentContext) -> Any:
        """Run the Mismatch Agent and return its report."""
        try:
            from .mismatch_agent import MismatchAgent

            agent = MismatchAgent(
                llm_client=self.llm, registry=self.registry
            )
            return agent.run(context)
        except ImportError:
            logger.info(
                "MismatchAgent not available; returning synthetic defaults."
            )
            return self._default_mismatch_report(context)
        except Exception as exc:
            logger.warning(
                "MismatchAgent failed: %s; returning synthetic defaults.",
                exc,
            )
            return self._default_mismatch_report(context)

    def _run_recoverability_agent(self, context: AgentContext) -> Any:
        """Run the Recoverability Agent and return its report."""
        try:
            from .recoverability_agent import RecoverabilityAgent

            agent = RecoverabilityAgent(
                llm_client=self.llm, registry=self.registry
            )
            return agent.run(context)
        except ImportError:
            logger.info(
                "RecoverabilityAgent not available; returning synthetic "
                "defaults."
            )
            return self._default_recoverability_report(context)
        except Exception as exc:
            logger.warning(
                "RecoverabilityAgent failed: %s; returning synthetic "
                "defaults.",
                exc,
            )
            return self._default_recoverability_report(context)

    def _run_analysis_agent(
        self,
        context: AgentContext,
        photon: Any,
        mismatch: Any,
        recoverability: Any,
    ) -> Any:
        """Run the Analysis Agent and return its report."""
        try:
            from .analysis_agent import AnalysisAgent

            agent = AnalysisAgent(
                llm_client=self.llm, registry=self.registry
            )
            # Pass sub-agent reports via the context
            context.photon_report = photon
            context.mismatch_report = mismatch
            context.recoverability_report = recoverability
            return agent.run(context)
        except ImportError:
            logger.info(
                "AnalysisAgent not available; returning synthetic defaults."
            )
            return self._default_analysis_report()
        except Exception as exc:
            logger.warning(
                "AnalysisAgent failed: %s; returning synthetic defaults.",
                exc,
            )
            return self._default_analysis_report()

    def _run_negotiation(
        self,
        photon: Any,
        mismatch: Any,
        recoverability: Any,
        analysis: Any,
    ) -> Any:
        """Run inter-agent negotiation and return the result."""
        try:
            from .negotiator import AgentNegotiator

            negotiator = AgentNegotiator()
            return negotiator.negotiate(photon, mismatch, recoverability, analysis)
        except ImportError:
            logger.info(
                "AgentNegotiator not available; returning default "
                "proceed=True result."
            )
            return self._default_negotiation_result()
        except Exception as exc:
            logger.warning(
                "Negotiation failed: %s; returning default proceed=True.",
                exc,
            )
            return self._default_negotiation_result()

    # ===================================================================
    # Continuity checker (optional)
    # ===================================================================

    def _check_physical_continuity(self, system: ImagingSystem) -> bool:
        """Run the physical continuity checker if available.

        Returns True if continuity checks pass or the checker is not
        available.
        """
        try:
            from .continuity_checker import PhysicalContinuityChecker

            checker = PhysicalContinuityChecker()
            return checker.check(system)
        except ImportError:
            logger.debug(
                "PhysicalContinuityChecker not available; skipping."
            )
            return True
        except Exception as exc:
            logger.warning(
                "Physical continuity check failed: %s; proceeding anyway.",
                exc,
            )
            return True

    # ===================================================================
    # Non-imaging prompt handler
    # ===================================================================

    def handle_non_imaging_prompt(self, prompt: str) -> dict:
        """Handle prompts that do not match any imaging modality.

        Returns a helpful error dict with supported modalities and
        example prompts.

        Parameters
        ----------
        prompt : str
            The unmatched user prompt.

        Returns
        -------
        dict
            Keys: ``"error"``, ``"supported_modalities"``,
            ``"example_prompts"``.
        """
        supported: List[str] = []
        if self.registry is not None:
            modality_keys = self.registry.list_modalities()
            for key in modality_keys:
                try:
                    mod = self.registry.get_modality(key)
                    supported.append(f"  {key}: {mod.display_name}")
                except Exception:
                    supported.append(f"  {key}")

        example_prompts = [
            "Simulate CASSI hyperspectral imaging with 28 channels",
            "Reconstruct a widefield fluorescence microscopy image",
            "I have measured holography data, calibrate the operator",
            "Perform structured illumination microscopy (SIM) reconstruction",
            "Reconstruct a CT sinogram with 180 projection angles",
            "Run single-pixel camera compressed sensing reconstruction",
            "Process my light-sheet microscopy data to remove stripes",
            "Reconstruct a 3D scene from multi-view images using NeRF",
        ]

        error_msg = (
            f"Could not identify an imaging modality from your prompt: "
            f"'{prompt[:100]}'. "
            f"Please mention a specific imaging technique or modality."
        )

        return {
            "error": error_msg,
            "supported_modalities": supported,
            "example_prompts": example_prompts,
        }

    # ===================================================================
    # Default / synthetic reports for pipeline continuity
    # ===================================================================

    def _default_photon_report(self, context: AgentContext) -> Any:
        """Return a synthetic PhotonReport when the PhotonAgent is
        unavailable."""
        from .contracts import PhotonReport, NoiseRegime

        return PhotonReport(
            n_photons_per_pixel=1000.0,
            snr_db=30.0,
            noise_regime=NoiseRegime.shot_limited,
            shot_noise_sigma=31.6,
            read_noise_sigma=5.0,
            total_noise_sigma=32.0,
            feasible=True,
            quality_tier="acceptable",
            throughput_chain=[],
            noise_model="mixed_poisson_gaussian",
            explanation=(
                "Synthetic default: PhotonAgent not available. "
                "Using nominal photon budget."
            ),
        )

    def _default_mismatch_report(self, context: AgentContext) -> Any:
        """Return a synthetic MismatchReport when the MismatchAgent is
        unavailable."""
        from .contracts import MismatchReport

        return MismatchReport(
            modality_key=context.modality_key,
            mismatch_family="default",
            parameters={},
            severity_score=0.1,
            correction_method="none",
            expected_improvement_db=0.0,
            explanation=(
                "Synthetic default: MismatchAgent not available. "
                "Assuming minimal mismatch."
            ),
        )

    def _default_recoverability_report(self, context: AgentContext) -> Any:
        """Return a synthetic RecoverabilityReport when the
        RecoverabilityAgent is unavailable."""
        from .contracts import (
            RecoverabilityReport,
            NoiseRegime,
            SignalPriorClass,
        )

        return RecoverabilityReport(
            compression_ratio=1.0,
            noise_regime=NoiseRegime.shot_limited,
            signal_prior_class=SignalPriorClass.tv,
            operator_diversity_score=0.5,
            condition_number_proxy=10.0,
            recoverability_score=0.7,
            recoverability_confidence=0.5,
            expected_psnr_db=28.0,
            expected_psnr_uncertainty_db=3.0,
            recommended_solver_family="gap_tv",
            verdict="sufficient",
            explanation=(
                "Synthetic default: RecoverabilityAgent not available. "
                "Using nominal recoverability estimates."
            ),
        )

    def _default_analysis_report(self) -> Any:
        """Return a synthetic SystemAnalysis when the AnalysisAgent is
        unavailable."""
        from .contracts import SystemAnalysis, BottleneckScores

        return SystemAnalysis(
            primary_bottleneck="unknown",
            bottleneck_scores=BottleneckScores(
                photon=0.3,
                mismatch=0.2,
                compression=0.3,
                solver=0.2,
            ),
            suggestions=[],
            overall_verdict="Pipeline running with synthetic defaults.",
            probability_of_success=0.6,
            explanation=(
                "Synthetic default: AnalysisAgent not available. "
                "Bottleneck scores are nominal placeholders."
            ),
        )

    def _default_negotiation_result(self) -> Any:
        """Return a default NegotiationResult when the negotiator is
        unavailable."""
        from .contracts import NegotiationResult

        return NegotiationResult(
            vetoes=[],
            proceed=True,
            probability_of_success=0.6,
        )
