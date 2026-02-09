"""pwm_core.agents.analysis_agent

Analysis Agent: scores bottlenecks, generates actionable suggestions, and
computes a pre-flight probability of success for the reconstruction pipeline.

Design mantra
-------------
All scoring is **deterministic** -- the LLM is used only for an optional
natural-language explanation appended to the result.  The ``run()`` method
must succeed with ``llm_client=None``.

Bottleneck scoring
------------------
Each subsystem is assigned a severity score in [0, 1] where 0 means "no
issue" and 1 means "showstopper".  The primary bottleneck is the subsystem
with the highest score.  Suggestions are generated for the top bottleneck(s).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .base import BaseAgent, AgentContext
from .contracts import (
    BottleneckScores,
    PhotonReport,
    MismatchReport,
    RecoverabilityReport,
    Suggestion,
    SystemAnalysis,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suggestion templates keyed by bottleneck subsystem
# ---------------------------------------------------------------------------

_SUGGESTION_TEMPLATES: Dict[str, List[dict]] = {
    "photon": [
        {
            "action": "Increase source power or exposure time",
            "priority": "critical",
            "expected_improvement_db": 3.0,
            "parameter_path": "budget.exposure_time_s",
            "parameter_change": {"factor": 4.0},
            "details": (
                "The photon budget is the dominant bottleneck.  Quadrupling "
                "the exposure time adds ~6 dB SNR, which can shift the noise "
                "regime from photon-starved to shot-limited."
            ),
        },
        {
            "action": "Replace low-throughput optical elements",
            "priority": "high",
            "expected_improvement_db": 2.0,
            "parameter_path": "system.elements[*].throughput",
            "parameter_change": {"min_throughput": 0.5},
            "details": (
                "One or more elements in the optical chain have throughput "
                "below 50%.  Replacing or removing them improves the photon "
                "budget without changing integration time."
            ),
        },
        {
            "action": "Use a photon-noise-aware solver (e.g. PnP-ADMM with BM3D)",
            "priority": "medium",
            "expected_improvement_db": 1.5,
            "parameter_path": "solver_family",
            "parameter_change": {"solver": "pnp_admm"},
            "details": (
                "Switching to a solver that explicitly models Poisson "
                "statistics can recover 1-2 dB in the photon-starved regime."
            ),
        },
    ],
    "mismatch": [
        {
            "action": "Apply operator correction before reconstruction",
            "priority": "critical",
            "expected_improvement_db": 4.0,
            "parameter_path": "mismatch.correction_method",
            "parameter_change": {"apply_correction": True},
            "details": (
                "Forward-model mismatch is the dominant bottleneck.  Running "
                "operator correction (UPWMI or similar) before reconstruction "
                "can recover 3-5 dB."
            ),
        },
        {
            "action": "Recalibrate the forward operator from measured data",
            "priority": "high",
            "expected_improvement_db": 3.0,
            "parameter_path": "system.forward_operator",
            "parameter_change": {"recalibrate": True},
            "details": (
                "If calibration data is available, refitting the forward "
                "operator to measured PSFs or system matrices reduces "
                "model-reality mismatch."
            ),
        },
    ],
    "compression": [
        {
            "action": "Reduce compression ratio (acquire more measurements)",
            "priority": "critical",
            "expected_improvement_db": 5.0,
            "parameter_path": "system.signal_dims",
            "parameter_change": {"increase_measurements": True},
            "details": (
                "The compression ratio is too aggressive for reliable "
                "recovery.  Acquiring more measurements (e.g. multiple "
                "shots or coded exposures) improves the conditioning of "
                "the inverse problem."
            ),
        },
        {
            "action": "Switch to a stronger signal prior",
            "priority": "high",
            "expected_improvement_db": 2.5,
            "parameter_path": "recoverability.signal_prior_class",
            "parameter_change": {"prior": "deep_prior"},
            "details": (
                "A learned prior (deep image prior, diffusion model) can "
                "compensate for under-sampling by ~2-3 dB compared to "
                "hand-crafted priors like TV."
            ),
        },
        {
            "action": "Improve mask diversity / operator incoherence",
            "priority": "medium",
            "expected_improvement_db": 1.5,
            "parameter_path": "system.elements[mask].parameters",
            "parameter_change": {"diversity": "high"},
            "details": (
                "Increasing the spatial diversity of the coded aperture "
                "mask improves the restricted isometry property of the "
                "sensing matrix."
            ),
        },
    ],
    "solver": [
        {
            "action": "Try a higher-capacity solver",
            "priority": "high",
            "expected_improvement_db": 2.0,
            "parameter_path": "solver_family",
            "parameter_change": {"upgrade": True},
            "details": (
                "The current solver under-performs on this problem class.  "
                "Switching to a learned unrolling network (e.g. MST, "
                "ISTA-Net) may recover 2-3 dB."
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

_VERDICT_THRESHOLDS = [
    (0.80, "excellent"),
    (0.60, "sufficient"),
    (0.35, "marginal"),
    (0.00, "insufficient"),
]


def _verdict_from_probability(prob: float) -> str:
    """Map probability of success to a human-readable verdict string."""
    for threshold, label in _VERDICT_THRESHOLDS:
        if prob >= threshold:
            return label
    return "insufficient"


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------


class AnalysisAgent(BaseAgent):
    """Scores subsystem bottlenecks, generates suggestions, and estimates the
    pre-flight probability of a successful reconstruction.

    All computation is deterministic.  The optional LLM client is used only
    to append a natural-language explanation to the ``SystemAnalysis`` result.

    Bottleneck scoring
    ------------------
    * **photon**:       ``1.0 - min(snr_db / 40, 1.0)``
    * **mismatch**:     ``severity_score`` from the MismatchReport
    * **compression**:  ``1.0 - recoverability_score`` from RecoverabilityReport
    * **solver**:       ``0.2`` default (updated post-reconstruction)

    Probability of success
    ----------------------
    ``prod(1 - score * 0.5)`` for each bottleneck, clipped to [0, 1].
    Each subsystem contributes an independent "survival" factor; a score
    of 1.0 in any subsystem halves the probability on its own.
    """

    # ----- public interface --------------------------------------------------

    def run(self, context: AgentContext) -> SystemAnalysis:
        """Execute the analysis pipeline.

        Parameters
        ----------
        context : AgentContext
            Must contain ``photon_report``, ``mismatch_report``, and
            ``recoverability_report`` from prior agent stages.

        Returns
        -------
        SystemAnalysis
            Bottleneck scores, suggestions, verdict, and probability of
            success.

        Raises
        ------
        ValueError
            If any of the three required reports is missing from *context*.
        """
        photon = self._require_report(context.photon_report, "photon_report")
        mismatch = self._require_report(context.mismatch_report, "mismatch_report")
        recoverability = self._require_report(
            context.recoverability_report, "recoverability_report"
        )

        # -- Step 1: Deterministic bottleneck scoring -------------------------
        scores = self._compute_bottleneck_scores(photon, mismatch, recoverability)

        # -- Step 2: Identify the primary bottleneck --------------------------
        score_map = {
            "photon": scores.photon,
            "mismatch": scores.mismatch,
            "compression": scores.compression,
            "solver": scores.solver,
        }
        primary_bottleneck = max(score_map, key=score_map.get)  # type: ignore[arg-type]

        # -- Step 3: Probability of success -----------------------------------
        probability_of_success = self._compute_probability_of_success(score_map)

        # -- Step 4: Generate suggestions -------------------------------------
        suggestions = self._generate_suggestions(score_map, primary_bottleneck)

        # -- Step 5: Verdict --------------------------------------------------
        overall_verdict = _verdict_from_probability(probability_of_success)

        # -- Step 6: Optional LLM explanation ---------------------------------
        explanation = self._generate_explanation(
            primary_bottleneck, scores, probability_of_success, suggestions
        )

        return SystemAnalysis(
            primary_bottleneck=primary_bottleneck,
            bottleneck_scores=scores,
            suggestions=suggestions,
            overall_verdict=overall_verdict,
            probability_of_success=probability_of_success,
            explanation=explanation,
        )

    # ----- bottleneck scoring ------------------------------------------------

    def _compute_bottleneck_scores(
        self,
        photon: PhotonReport,
        mismatch: MismatchReport,
        recoverability: RecoverabilityReport,
    ) -> BottleneckScores:
        """Compute normalised bottleneck severity for each subsystem.

        Parameters
        ----------
        photon : PhotonReport
            Output of the Photon Agent.
        mismatch : MismatchReport
            Output of the Mismatch Agent.
        recoverability : RecoverabilityReport
            Output of the Recoverability Agent.

        Returns
        -------
        BottleneckScores
            Normalised scores in [0, 1] for each subsystem.
        """
        # Photon score: high SNR -> low score; 40 dB is "excellent" ceiling
        photon_score = 1.0 - min(photon.snr_db / 40.0, 1.0)
        photon_score = max(0.0, min(1.0, photon_score))

        # Mismatch score: directly from the MismatchReport
        mismatch_score = mismatch.severity_score

        # Compression score: inverse of recoverability
        compression_score = 1.0 - recoverability.recoverability_score

        # Solver score: default placeholder (updated post-reconstruction)
        solver_score = 0.2

        return BottleneckScores(
            photon=round(photon_score, 4),
            mismatch=round(mismatch_score, 4),
            compression=round(compression_score, 4),
            solver=round(solver_score, 4),
        )

    # ----- probability of success --------------------------------------------

    @staticmethod
    def _compute_probability_of_success(score_map: Dict[str, float]) -> float:
        """Compute joint probability of success from bottleneck scores.

        Each subsystem contributes a "survival" factor of ``(1 - score * 0.5)``.
        The joint probability is the product of all survival factors, clipped
        to [0, 1].

        Parameters
        ----------
        score_map : dict[str, float]
            Mapping from subsystem name to bottleneck severity score.

        Returns
        -------
        float
            Probability of success in [0, 1].
        """
        probability = 1.0
        for score in score_map.values():
            probability *= (1.0 - score * 0.5)
        return max(0.0, min(1.0, round(probability, 4)))

    # ----- suggestion generation ---------------------------------------------

    @staticmethod
    def _generate_suggestions(
        score_map: Dict[str, float],
        primary_bottleneck: str,
    ) -> List[Suggestion]:
        """Generate 1-3 actionable suggestions targeting the worst bottlenecks.

        The primary bottleneck always contributes at least one suggestion.
        If the primary bottleneck has multiple template suggestions, up to
        three are included, filtered by severity: only suggestions whose
        priority is commensurate with the bottleneck severity are kept.

        Parameters
        ----------
        score_map : dict[str, float]
            Mapping from subsystem name to bottleneck severity score.
        primary_bottleneck : str
            Key of the worst bottleneck subsystem.

        Returns
        -------
        list[Suggestion]
            Between 1 and 3 suggestions, ordered by priority.
        """
        templates = _SUGGESTION_TEMPLATES.get(primary_bottleneck, [])
        if not templates:
            # Fallback: generic suggestion
            return [
                Suggestion(
                    action=f"Investigate the '{primary_bottleneck}' subsystem",
                    priority="medium",
                    expected_improvement_db=1.0,
                    details=(
                        f"The '{primary_bottleneck}' subsystem has the highest "
                        f"bottleneck score ({score_map.get(primary_bottleneck, 0.0):.2f}). "
                        f"Manual inspection is recommended."
                    ),
                )
            ]

        severity = score_map.get(primary_bottleneck, 0.0)
        suggestions: List[Suggestion] = []

        for tmpl in templates:
            # For low severity, skip "critical" suggestions
            if severity < 0.5 and tmpl["priority"] == "critical":
                continue
            # For very low severity, only include "medium" or "low"
            if severity < 0.3 and tmpl["priority"] in ("critical", "high"):
                continue

            suggestions.append(Suggestion(**tmpl))

            if len(suggestions) >= 3:
                break

        # If severity filtering removed everything, include the first template
        if not suggestions and templates:
            suggestions.append(Suggestion(**templates[0]))

        return suggestions

    # ----- LLM explanation (optional) ----------------------------------------

    def _generate_explanation(
        self,
        primary_bottleneck: str,
        scores: BottleneckScores,
        probability_of_success: float,
        suggestions: List[Suggestion],
    ) -> str:
        """Produce a human-readable explanation of the analysis.

        If an LLM client is available, it narrates the result in plain
        English.  Otherwise a deterministic summary is returned.

        Parameters
        ----------
        primary_bottleneck : str
            Key of the worst bottleneck subsystem.
        scores : BottleneckScores
            All bottleneck severity scores.
        probability_of_success : float
            Joint probability of reconstruction success.
        suggestions : list[Suggestion]
            Generated suggestions.

        Returns
        -------
        str
            Human-readable explanation string.
        """
        # Deterministic fallback
        deterministic = (
            f"Primary bottleneck: {primary_bottleneck} "
            f"(score={getattr(scores, primary_bottleneck if primary_bottleneck != 'compression' else 'compression', 0.0):.2f}). "
            f"Probability of success: {probability_of_success:.1%}. "
            f"{len(suggestions)} suggestion(s) generated."
        )

        if self.llm is None:
            return deterministic

        try:
            prompt = (
                "You are a computational-imaging analyst.  Summarise the "
                "following bottleneck analysis in 2-3 concise sentences for "
                "a researcher:\n\n"
                f"Primary bottleneck: {primary_bottleneck}\n"
                f"Scores: photon={scores.photon:.2f}, mismatch={scores.mismatch:.2f}, "
                f"compression={scores.compression:.2f}, solver={scores.solver:.2f}\n"
                f"Probability of success: {probability_of_success:.1%}\n"
                f"Top suggestion: {suggestions[0].action if suggestions else 'none'}"
            )
            return self.llm.generate(prompt)
        except Exception:  # noqa: BLE001
            logger.warning(
                "LLM explanation generation failed; using deterministic fallback.",
                exc_info=True,
            )
            return deterministic

    # ----- internal helpers --------------------------------------------------

    @staticmethod
    def _require_report(report: object, name: str) -> object:
        """Validate that a required report is present in the context.

        Parameters
        ----------
        report : object
            The report object (or ``None``).
        name : str
            Human-readable field name for the error message.

        Returns
        -------
        object
            The validated report.

        Raises
        ------
        ValueError
            If *report* is ``None``.
        """
        if report is None:
            raise ValueError(
                f"AnalysisAgent requires '{name}' in the AgentContext, "
                f"but it was None.  Run the upstream agent first."
            )
        return report
