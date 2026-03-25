"""Hypothesis generation engine.

Generates and ranks scientific hypotheses about imaging system performance
based on domain diagnostic reports (Triad attribution), cross-modality
transfer suggestions, and the autonomous science loop.

Extends the existing AutonomousScienceLoop with structured hypothesis
generation and test planning.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HypothesisType(str, Enum):
    """Types of scientific hypotheses PWM can generate."""

    solver_improvement = "solver_improvement"
    parameter_tuning = "parameter_tuning"
    cross_modality_transfer = "cross_modality_transfer"
    noise_model_refinement = "noise_model_refinement"
    operator_correction = "operator_correction"
    calibration_improvement = "calibration_improvement"
    architecture_change = "architecture_change"


class HypothesisPriority(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    exploratory = "exploratory"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A structured scientific hypothesis with test plan."""

    hypothesis_id: str
    hypothesis_type: HypothesisType
    priority: HypothesisPriority
    statement: str
    rationale: str
    predicted_effect: str
    test_plan: List[str]
    confidence: float  # 0-1
    source: str  # "triad_diagnostic", "transfer_engine", "autonomous_loop", "manual"
    modality: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisReport:
    """Ranked list of hypotheses for a given modality/run."""

    modality: str
    run_id: Optional[str]
    hypotheses: List[Hypothesis]
    generated_at: str
    methodology: str = "hypothesis_engine_v1"

    @property
    def top_hypothesis(self) -> Optional[Hypothesis]:
        return self.hypotheses[0] if self.hypotheses else None

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Priority weights for ranking
# ---------------------------------------------------------------------------

_PRIORITY_WEIGHTS: Dict[HypothesisPriority, float] = {
    HypothesisPriority.critical: 5.0,
    HypothesisPriority.high: 4.0,
    HypothesisPriority.medium: 3.0,
    HypothesisPriority.low: 2.0,
    HypothesisPriority.exploratory: 1.0,
}


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """Generates ranked hypotheses from domain diagnostics and transfer analysis.

    Combines three signal sources:

    1. **Imaging Triad** (G1/G2/G3) -- which bottleneck dominates?
    2. **R-gate verdicts** (R1-R4) -- which certification gates failed?
    3. **Metric analysis** -- raw numbers (PSNR, SSIM, oracle gap).
    4. **Cross-modality transfer** -- what worked elsewhere?
    """

    def __init__(self) -> None:
        self._transfer_engine = None

    @property
    def transfer_engine(self):
        """Lazy-loaded CrossModalityTransferEngine."""
        if self._transfer_engine is None:
            from pwm_core.science.transfer_engine import CrossModalityTransferEngine

            self._transfer_engine = CrossModalityTransferEngine()
        return self._transfer_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_diagnostics(
        self,
        modality: str,
        domain_flags: Optional[Dict[str, Any]] = None,
        gate_verdicts: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> HypothesisReport:
        """Generate hypotheses from a run's diagnostic output.

        Analyzes *domain_flags* (e.g., imaging Triad G1/G2/G3), *gate_verdicts*
        (R1-R4), and *metrics* to produce ranked hypotheses about what would
        improve results.

        Parameters
        ----------
        modality:
            Imaging modality name (e.g. ``"ct"``, ``"mri"``).
        domain_flags:
            Diagnostic flags from the domain layer; expected to contain an
            ``"imaging"`` sub-dict with ``g1_sampling``, ``g2_noise``,
            ``g3_operator`` keys.
        gate_verdicts:
            R-gate verdict dict (keys: gate names, values: dicts with
            ``"verdict"`` and ``"message"``).
        metrics:
            Scalar metrics dict; recognized keys include ``"psnr_db"``,
            ``"ssim"``, ``"oracle_gap"``.
        run_id:
            Optional identifier for the evaluation run.

        Returns
        -------
        HypothesisReport
            Ranked list of hypotheses sorted by (confidence * priority weight).
        """
        hypotheses: List[Hypothesis] = []

        # 1. Analyze domain flags (imaging Triad)
        if domain_flags:
            imaging = domain_flags.get("imaging", {})
            if imaging:
                hypotheses.extend(
                    self._hypotheses_from_triad(modality, imaging, run_id)
                )

        # 2. Analyze gate verdicts
        if gate_verdicts:
            hypotheses.extend(
                self._hypotheses_from_gates(modality, gate_verdicts, run_id)
            )

        # 3. Analyze metrics for improvement opportunities
        if metrics:
            hypotheses.extend(
                self._hypotheses_from_metrics(modality, metrics, run_id)
            )

        # 4. Cross-modality transfer suggestions
        transfer_suggestions = self.transfer_engine.suggest_transfers(
            modality, top_k=3
        )
        for ts in transfer_suggestions:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"transfer_{ts.source_modality}_{modality}",
                    hypothesis_type=HypothesisType.cross_modality_transfer,
                    priority=(
                        HypothesisPriority.medium
                        if ts.confidence > 0.5
                        else HypothesisPriority.exploratory
                    ),
                    statement=(
                        f"Transfer algorithm from {ts.source_modality} to {modality}"
                    ),
                    rationale=ts.rationale,
                    predicted_effect=ts.expected_benefit,
                    test_plan=[
                        f"Run {ts.algorithm} from {ts.source_modality} on {modality} benchmark",
                        f"Compare PSNR/SSIM with current best for {modality}",
                        "Analyze domain_flags for bottleneck shift",
                    ],
                    confidence=ts.confidence,
                    source="transfer_engine",
                    modality=modality,
                    metadata={
                        "shared_primitives": ts.shared_primitives,
                        "risks": ts.risk_factors,
                    },
                )
            )

        # Sort by confidence * priority weight (descending)
        hypotheses.sort(
            key=lambda h: h.confidence * _PRIORITY_WEIGHTS.get(h.priority, 1.0),
            reverse=True,
        )

        return HypothesisReport(
            modality=modality,
            run_id=run_id,
            hypotheses=hypotheses,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Triad-based hypotheses
    # ------------------------------------------------------------------

    def _hypotheses_from_triad(
        self,
        modality: str,
        imaging_flags: Dict[str, Any],
        run_id: Optional[str],
    ) -> List[Hypothesis]:
        """Generate hypotheses from imaging Triad attribution (G1/G2/G3)."""
        hypotheses: List[Hypothesis] = []

        # -- G1 (recoverability / sampling) --------------------------------
        g1 = imaging_flags.get("g1_sampling", {})
        if isinstance(g1, dict) and g1.get("verdict") == "warn":
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"triad_g1_{run_id or 'unknown'}",
                    hypothesis_type=HypothesisType.solver_improvement,
                    priority=HypothesisPriority.high,
                    statement=(
                        f"G1 (recoverability) is the dominant bottleneck for {modality}"
                    ),
                    rationale=(
                        "Triad analysis shows sampling/carrier limit dominates. "
                        "The forward model's information capacity is the main constraint."
                    ),
                    predicted_effect=(
                        "Improving the reconstruction algorithm or adding "
                        "regularization should help more than fixing calibration."
                    ),
                    test_plan=[
                        "Try a more advanced solver (e.g., deep unrolling, learned regularization)",
                        "Increase measurement diversity if possible",
                        "Compare Scenario I vs Scenario III gaps",
                    ],
                    confidence=float(
                        g1.get("details", {}).get("confidence", 0.6)
                    ),
                    source="triad_diagnostic",
                    modality=modality,
                )
            )

        # -- G2 (noise) ----------------------------------------------------
        g2 = imaging_flags.get("g2_noise", {})
        if isinstance(g2, dict) and g2.get("verdict") == "warn":
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"triad_g2_{run_id or 'unknown'}",
                    hypothesis_type=HypothesisType.noise_model_refinement,
                    priority=HypothesisPriority.high,
                    statement=(
                        f"G2 (noise model) is the dominant bottleneck for {modality}"
                    ),
                    rationale=(
                        "Triad analysis shows noise model mismatch dominates. "
                        "The assumed noise model differs significantly from actual noise."
                    ),
                    predicted_effect=(
                        "Refining the noise model or using noise-aware reconstruction "
                        "should yield significant improvement."
                    ),
                    test_plan=[
                        "Characterize actual noise distribution (Gaussian vs Poisson vs mixed)",
                        "Try noise-aware solver (BM3D-prox, noise2noise)",
                        "Compare Scenario II residuals with Scenario I",
                    ],
                    confidence=float(
                        g2.get("details", {}).get("confidence", 0.6)
                    ),
                    source="triad_diagnostic",
                    modality=modality,
                )
            )

        # -- G3 (operator mismatch) ----------------------------------------
        g3 = imaging_flags.get("g3_operator", {})
        if isinstance(g3, dict) and g3.get("verdict") == "warn":
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"triad_g3_{run_id or 'unknown'}",
                    hypothesis_type=HypothesisType.operator_correction,
                    priority=HypothesisPriority.critical,
                    statement=(
                        f"G3 (operator mismatch) is the dominant bottleneck for {modality}"
                    ),
                    rationale=(
                        "Triad analysis shows forward-model mismatch dominates. "
                        "The assumed operator differs from the true physics."
                    ),
                    predicted_effect=(
                        "Calibrating or correcting the forward model should yield "
                        "the largest improvement."
                    ),
                    test_plan=[
                        "Run calibration (Scenario III) and measure improvement",
                        "Investigate specific mismatch sources (PSF, geometry, material properties)",
                        "Try self-calibrating reconstruction algorithms",
                    ],
                    confidence=float(
                        g3.get("details", {}).get("confidence", 0.6)
                    ),
                    source="triad_diagnostic",
                    modality=modality,
                )
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Gate-verdict-based hypotheses
    # ------------------------------------------------------------------

    def _hypotheses_from_gates(
        self,
        modality: str,
        gate_verdicts: Dict[str, Any],
        run_id: Optional[str],
    ) -> List[Hypothesis]:
        """Generate hypotheses from R-gate failures."""
        hypotheses: List[Hypothesis] = []

        for gate_name, verdict_data in gate_verdicts.items():
            if isinstance(verdict_data, dict) and verdict_data.get("verdict") == "fail":
                hypotheses.append(
                    Hypothesis(
                        hypothesis_id=f"gate_{gate_name}_{run_id or 'unknown'}",
                        hypothesis_type=HypothesisType.parameter_tuning,
                        priority=HypothesisPriority.critical,
                        statement=(
                            f"Gate {gate_name} failed -- must be resolved before certification"
                        ),
                        rationale=verdict_data.get("message", "Gate check failed"),
                        predicted_effect=(
                            "Fixing this gate failure is required for any trust "
                            "tier above Draft."
                        ),
                        test_plan=[
                            f"Investigate {gate_name} failure: "
                            f"{verdict_data.get('message', '')}"
                        ],
                        confidence=1.0,
                        source="gate_verdict",
                        modality=modality,
                    )
                )

        return hypotheses

    # ------------------------------------------------------------------
    # Metric-based hypotheses
    # ------------------------------------------------------------------

    def _hypotheses_from_metrics(
        self,
        modality: str,
        metrics: Dict[str, Any],
        run_id: Optional[str],
    ) -> List[Hypothesis]:
        """Generate hypotheses from metric analysis."""
        hypotheses: List[Hypothesis] = []

        psnr = metrics.get("psnr_db")
        oracle_gap = metrics.get("oracle_gap")

        if oracle_gap is not None and oracle_gap > 5.0:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"oracle_gap_{run_id or 'unknown'}",
                    hypothesis_type=HypothesisType.solver_improvement,
                    priority=HypothesisPriority.high,
                    statement=(
                        f"Large oracle gap ({oracle_gap:.1f} dB) suggests "
                        f"significant room for improvement"
                    ),
                    rationale=(
                        "The gap between current performance and oracle "
                        "reconstruction indicates the solver is far from optimal."
                    ),
                    predicted_effect=(
                        f"Up to {oracle_gap:.1f} dB PSNR improvement possible "
                        f"with better solver."
                    ),
                    test_plan=[
                        "Try more iterations or better initialization",
                        "Switch to a more powerful solver family",
                        "Analyze which scenarios show the largest gaps",
                    ],
                    confidence=min(oracle_gap / 10.0, 0.95),
                    source="metric_analysis",
                    modality=modality,
                )
            )

        if psnr is not None and psnr < 20.0:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"low_psnr_{run_id or 'unknown'}",
                    hypothesis_type=HypothesisType.architecture_change,
                    priority=HypothesisPriority.high,
                    statement=(
                        f"Low PSNR ({psnr:.1f} dB) may indicate fundamental "
                        f"solver mismatch"
                    ),
                    rationale=(
                        "PSNR below 20 dB often indicates the solver architecture "
                        "is not well-suited to this forward model."
                    ),
                    predicted_effect=(
                        "Architecture change may yield 5-15 dB improvement."
                    ),
                    test_plan=[
                        "Check if the solver handles this forward model family",
                        "Try a solver designed for this primitive chain",
                        "Verify data loading and preprocessing",
                    ],
                    confidence=0.7,
                    source="metric_analysis",
                    modality=modality,
                )
            )

        return hypotheses
