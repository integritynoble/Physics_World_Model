"""pwm_core.agents.what_if_precomputer

Pre-compute sensitivity curves for the interactive viewer.

Generates what-if data: "what happens to PSNR if I change photon budget
by X?" or "what happens if I change compression ratio by Y?"  Results
are written to the RunBundle ``what_if/`` directory as JSON files that
the viewer can load without re-running the pipeline.

Entirely deterministic — no LLM required.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .contracts import StrictBaseModel, SystemAnalysis
from .base import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SensitivityPoint(StrictBaseModel):
    """One point on a sensitivity curve."""

    parameter_value: float
    predicted_psnr_db: float
    predicted_snr_db: float
    regime: str


class SensitivityCurve(StrictBaseModel):
    """Complete sensitivity curve for one parameter sweep."""

    parameter_name: str
    parameter_unit: str
    points: List[SensitivityPoint]
    baseline_value: float
    baseline_psnr_db: float


# ---------------------------------------------------------------------------
# Precomputer
# ---------------------------------------------------------------------------

class WhatIfPrecomputer:
    """Pre-compute sensitivity curves for interactive viewer.

    Generates two standard curves:
    1. PSNR vs photon budget (exposure multiplier)
    2. PSNR vs compression ratio

    Physics-based models (no simulation needed):
    - Photon: SNR scales as sqrt(N), PSNR ~ 20*log10(SNR)
    - Compression: empirical PSNR ~ -10*log10(1 - CR) for sparse signals
    """

    # Standard sweep parameters
    PHOTON_MULTIPLIERS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]
    CR_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]

    def run(
        self,
        analysis: SystemAnalysis,
        context: AgentContext,
        output_dir: Optional[str] = None,
    ) -> Dict[str, SensitivityCurve]:
        """Pre-compute sensitivity curves.

        Parameters
        ----------
        analysis : SystemAnalysis
            Analysis results with baseline PSNR and bottleneck scores.
        context : AgentContext
            Pipeline context with photon and budget information.
        output_dir : str, optional
            Directory to write JSON files. If None, returns data only.

        Returns
        -------
        Dict[str, SensitivityCurve]
            Mapping from curve name to sensitivity data.
        """
        curves: Dict[str, SensitivityCurve] = {}

        # Extract baseline info
        baseline_psnr = self._get_baseline_psnr(context)
        baseline_snr = self._get_baseline_snr(context)

        # 1. Photon sensitivity
        photon_curve = self._compute_photon_sensitivity(
            baseline_psnr, baseline_snr, context
        )
        curves["photon_sensitivity"] = photon_curve

        # 2. Compression ratio sensitivity
        cr_curve = self._compute_cr_sensitivity(baseline_psnr, context)
        curves["cr_sensitivity"] = cr_curve

        # Write to disk if output_dir specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for name, curve in curves.items():
                path = os.path.join(output_dir, f"{name}.json")
                with open(path, "w") as f:
                    json.dump(curve.model_dump(), f, indent=2)
                logger.info(f"Wrote sensitivity curve: {path}")

        return curves

    def _compute_photon_sensitivity(
        self,
        baseline_psnr: float,
        baseline_snr: float,
        context: AgentContext,
    ) -> SensitivityCurve:
        """Compute PSNR vs photon budget multiplier."""
        is_photon_starved = False
        if context.photon_report is not None:
            regime = getattr(context.photon_report, "noise_regime", None)
            if regime is not None:
                is_photon_starved = str(regime) == "photon_starved"

        points: List[SensitivityPoint] = []
        for mult in self.PHOTON_MULTIPLIERS:
            # SNR scaling: shot-limited -> sqrt(N), read-limited -> linear
            if is_photon_starved:
                snr_gain_db = 5.0 * math.log10(max(mult, 1e-6))
            else:
                snr_gain_db = 10.0 * math.log10(max(mult, 1e-6))

            pred_snr = baseline_snr + snr_gain_db
            pred_psnr = baseline_psnr + snr_gain_db

            # Determine regime
            if mult < 0.25:
                regime = "photon_starved"
            elif mult < 1.0:
                regime = "marginal"
            elif mult < 4.0:
                regime = "shot_limited"
            else:
                regime = "read_limited"

            points.append(SensitivityPoint(
                parameter_value=mult,
                predicted_psnr_db=pred_psnr,
                predicted_snr_db=pred_snr,
                regime=regime,
            ))

        return SensitivityCurve(
            parameter_name="photon_budget_multiplier",
            parameter_unit="x",
            points=points,
            baseline_value=1.0,
            baseline_psnr_db=baseline_psnr,
        )

    def _compute_cr_sensitivity(
        self,
        baseline_psnr: float,
        context: AgentContext,
    ) -> SensitivityCurve:
        """Compute PSNR vs compression ratio."""
        current_cr = 0.25
        if context.budget and "compression_ratio" in context.budget:
            current_cr = float(context.budget["compression_ratio"])

        points: List[SensitivityPoint] = []
        for cr in self.CR_VALUES:
            # Empirical model: PSNR improves with CR, saturating above 0.5
            ratio = cr / max(current_cr, 1e-6)
            if cr > 0.5:
                # Saturating regime
                psnr_change = 10.0 * math.log10(max(ratio, 1e-6)) * 0.5
            else:
                psnr_change = 10.0 * math.log10(max(ratio, 1e-6))

            pred_psnr = baseline_psnr + psnr_change

            # Determine regime
            if cr < 0.1:
                regime = "severely_undersampled"
            elif cr < 0.25:
                regime = "undersampled"
            elif cr < 0.5:
                regime = "moderate"
            else:
                regime = "well_sampled"

            points.append(SensitivityPoint(
                parameter_value=cr,
                predicted_psnr_db=pred_psnr,
                predicted_snr_db=pred_psnr,  # Approximate
                regime=regime,
            ))

        return SensitivityCurve(
            parameter_name="compression_ratio",
            parameter_unit="ratio",
            points=points,
            baseline_value=current_cr,
            baseline_psnr_db=baseline_psnr,
        )

    @staticmethod
    def _get_baseline_psnr(context: AgentContext) -> float:
        """Extract baseline PSNR from context."""
        if context.recoverability_report is not None:
            psnr = getattr(context.recoverability_report, "expected_psnr_db", None)
            if psnr is not None:
                return float(psnr)
        return 25.0  # Default assumption

    @staticmethod
    def _get_baseline_snr(context: AgentContext) -> float:
        """Extract baseline SNR from context."""
        if context.photon_report is not None:
            snr = getattr(context.photon_report, "snr_db", None)
            if snr is not None:
                return float(snr)
        return 20.0  # Default assumption
