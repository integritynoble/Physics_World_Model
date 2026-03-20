"""pwm_core.agents.self_improvement

Design alternative proposals with proxy simulation predictions.

Deterministic self-improvement loop: analyses bottleneck scores and proposes
concrete parameter changes with predicted PSNR improvements. No LLM required.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from .contracts import StrictBaseModel, SystemAnalysis
from .base import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class DesignAlternative(StrictBaseModel):
    """One proposed design change with predicted impact."""

    change: str
    predicted_psnr_improvement_db: float
    cost: str
    parameter_path: str
    parameter_change: Dict[str, Any]


# ---------------------------------------------------------------------------
# Self-improvement loop
# ---------------------------------------------------------------------------

class SelfImprovementLoop:
    """Deterministic self-improvement: propose design alternatives based on
    bottleneck analysis.

    Analyses SystemAnalysis bottleneck scores and proposes concrete parameter
    changes (exposure, compression ratio) with predicted PSNR improvements
    from physics-based scaling laws.

    No LLM required — entirely deterministic.
    """

    MAX_ITERATIONS: int = 3
    PROXY_RESOLUTION_FACTOR: float = 0.25

    def run(
        self,
        analysis: SystemAnalysis,
        context: AgentContext,
    ) -> List[DesignAlternative]:
        """Generate design alternative proposals.

        Parameters
        ----------
        analysis : SystemAnalysis
            Output from the Analysis Agent with bottleneck scores.
        context : AgentContext
            Pipeline context with modality and budget information.

        Returns
        -------
        List[DesignAlternative]
            Ranked list of proposed design changes.
        """
        proposals: List[DesignAlternative] = []

        # Photon-limited: propose exposure increases
        if analysis.bottleneck_scores.photon > 0.4:
            for factor in (2.0, 4.0):
                improvement = self._predict_exposure_improvement(context, factor)
                proposals.append(DesignAlternative(
                    change=f"Increase exposure by {factor:.0f}x",
                    predicted_psnr_improvement_db=improvement,
                    cost=f"Acquisition time x{factor:.0f}",
                    parameter_path="states.budget.photon_budget.exposure_time",
                    parameter_change={"multiply": factor},
                ))

        # Compression-limited: propose CR boosts
        if analysis.bottleneck_scores.compression > 0.4:
            for cr_boost in (1.5, 2.0):
                improvement = self._predict_cr_improvement(context, cr_boost)
                proposals.append(DesignAlternative(
                    change=f"Boost compression ratio by {cr_boost:.1f}x",
                    predicted_psnr_improvement_db=improvement,
                    cost=f"Measurement count x{cr_boost:.1f}",
                    parameter_path="states.budget.measurement_budget.sampling_rate",
                    parameter_change={"multiply": cr_boost},
                ))

        # Sort by predicted improvement (best first)
        proposals.sort(key=lambda p: p.predicted_psnr_improvement_db, reverse=True)
        return proposals

    @staticmethod
    def _predict_exposure_improvement(context: AgentContext, factor: float) -> float:
        """Predict PSNR improvement from exposure increase.

        Physics-based scaling:
        - Shot-limited: SNR scales as sqrt(N) -> 10*log10(factor) dB
        - Photon-starved: diminishing returns -> 5*log10(factor) dB
        """
        # Check noise regime from photon report if available
        photon_starved = False
        if context.photon_report is not None:
            regime = getattr(context.photon_report, "noise_regime", None)
            if regime is not None:
                photon_starved = str(regime) == "photon_starved"

        if photon_starved:
            return 5.0 * math.log10(max(factor, 1.0))
        return 10.0 * math.log10(max(factor, 1.0))

    @staticmethod
    def _predict_cr_improvement(context: AgentContext, cr_boost: float) -> float:
        """Predict PSNR improvement from compression ratio boost.

        Empirical scaling: 10*log10(ratio), saturating above CR=0.5.
        """
        # Get current CR from context budget if available
        current_cr = 0.25  # default assumption
        if context.budget and "compression_ratio" in context.budget:
            current_cr = float(context.budget["compression_ratio"])

        new_cr = min(current_cr * cr_boost, 1.0)
        ratio = new_cr / max(current_cr, 1e-6)

        # Saturating above CR=0.5: diminishing returns
        if new_cr > 0.5:
            # Above 0.5 — reduced scaling
            ratio_effective = 1.0 + (ratio - 1.0) * 0.5
            return 10.0 * math.log10(max(ratio_effective, 1.0))

        return 10.0 * math.log10(max(ratio, 1.0))
