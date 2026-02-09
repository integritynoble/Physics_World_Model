"""pwm_core.agents.recoverability_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Practical Recoverability Agent with interpolation and confidence reporting.

Renamed from ``CompressedAgent`` to reflect its actual role: it estimates
whether a computational-imaging reconstruction is likely to succeed given
the system's compression ratio, noise regime, signal prior, and operator
diversity.

The agent uses **empirical calibration tables** (``compression_db.yaml``)
rather than synthetic RIP/sparsity formulas.  When the exact
(compression-ratio, noise-regime, solver) triple is not in the table,
linear interpolation between the two nearest entries is performed and a
confidence band is reported so downstream agents know how trustworthy
the estimate is.

Design mantra
-------------
The ``run()`` method succeeds without an LLM and produces fully
deterministic outputs.  Every code path returns a validated
``RecoverabilityReport``.
"""

from __future__ import annotations

import logging
from math import prod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseAgent, AgentContext
from .contracts import (
    InterpolatedResult,
    NoiseRegime,
    RecoverabilityReport,
    SignalPriorClass,
)

logger = logging.getLogger(__name__)

# Default calibration entry used when the registry has no table for a
# modality.  Provides conservative estimates so the pipeline never
# hard-fails on missing data.
_FALLBACK_RECOVERABILITY = 0.50
_FALLBACK_PSNR_DB = 25.0
_FALLBACK_CONFIDENCE = 0.3
_FALLBACK_UNCERTAINTY_DB = 5.0


class RecoverabilityAgent(BaseAgent):
    """Estimate reconstruction recoverability from system parameters.

    The agent computes a four-factor scoring stack:

    1. **Signal prior class** -- loaded from the calibration registry.
    2. **Operator diversity** -- modality-specific heuristic measuring
       incoherence / pattern diversity.
    3. **Noise regime** -- taken from the upstream ``PhotonReport``.
    4. **Calibration-table lookup** -- interpolated from
       ``compression_db.yaml`` entries indexed by compression ratio,
       noise regime, and solver family.

    The output includes both a point estimate and an uncertainty band,
    allowing the Analysis Agent and Negotiator to make risk-aware
    decisions.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Not used by this agent (reserved for future narrative).
    registry : RegistryBuilder, optional
        Source of truth for calibration tables.  Required.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> RecoverabilityReport:
        """Execute the recoverability analysis pipeline.

        Parameters
        ----------
        context : AgentContext
            Must have ``modality_key`` set.  ``imaging_system`` is used
            for compression-ratio computation and operator diversity.
            ``photon_report`` supplies the noise regime.
            ``solver_family`` narrows the calibration-table lookup.

        Returns
        -------
        RecoverabilityReport
            Validated pydantic model with recoverability score,
            confidence, PSNR estimate, uncertainty, verdict, and
            recommended solver.

        Raises
        ------
        RuntimeError
            If the registry was not provided at construction time.
        """
        registry = self._require_registry()
        modality_key = context.modality_key

        # --- Factor 1: Compression ratio ------------------------------
        cr = self._compute_compression_ratio(context)

        # --- Factor 2: Signal prior class -----------------------------
        signal_prior_class = self._get_signal_prior_class(
            registry, modality_key,
        )

        # --- Factor 3: Operator diversity & condition proxy -----------
        diversity_score = self._compute_operator_diversity(context)
        condition_proxy = self._compute_condition_number_proxy(
            diversity_score,
        )

        # --- Factor 4: Noise regime -----------------------------------
        noise_regime = self._resolve_noise_regime(context)

        # --- Calibration-table interpolation --------------------------
        solver_family = context.solver_family
        lookup = self._interpolated_lookup(
            registry, modality_key, cr, noise_regime, solver_family,
        )

        recoverability = lookup.recoverability
        expected_psnr_db = lookup.expected_psnr_db
        confidence = lookup.confidence
        uncertainty_db = lookup.uncertainty_db

        # --- Verdict --------------------------------------------------
        verdict = self._compute_verdict(recoverability)

        # --- Recommended solver ---------------------------------------
        recommended_solver = self._find_best_solver(
            registry, modality_key, cr, noise_regime,
        )

        # --- Explanation ----------------------------------------------
        explanation = self._build_explanation(
            modality_key, cr, noise_regime, signal_prior_class,
            diversity_score, condition_proxy, recoverability,
            confidence, expected_psnr_db, uncertainty_db, verdict,
            recommended_solver,
        )

        return RecoverabilityReport(
            compression_ratio=float(cr),
            noise_regime=noise_regime,
            signal_prior_class=signal_prior_class,
            operator_diversity_score=float(diversity_score),
            condition_number_proxy=float(condition_proxy),
            recoverability_score=float(np.clip(recoverability, 0.0, 1.0)),
            recoverability_confidence=float(np.clip(confidence, 0.0, 1.0)),
            expected_psnr_db=float(expected_psnr_db),
            expected_psnr_uncertainty_db=float(max(uncertainty_db, 0.0)),
            recommended_solver_family=recommended_solver,
            verdict=verdict,
            calibration_table_entry=lookup.raw_entry,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_compression_ratio(context: AgentContext) -> float:
        """Compute the compression ratio from the imaging system dims.

        .. math::

            \\text{CR} = \\frac{\\prod(y\\_shape)}{\\prod(x\\_shape)}

        If the imaging system or its ``signal_dims`` is unavailable a
        conservative default of 0.5 is returned.

        Parameters
        ----------
        context : AgentContext
            Pipeline context containing the imaging system.

        Returns
        -------
        float
            Compression ratio (typically in (0, 1] for under-determined
            systems).
        """
        system = context.imaging_system
        if system is None:
            logger.warning(
                "No imaging_system in context; using default CR=0.50."
            )
            return 0.50

        signal_dims = getattr(system, "signal_dims", None)
        if signal_dims is None or not isinstance(signal_dims, dict):
            logger.warning(
                "imaging_system has no valid signal_dims; using default "
                "CR=0.50."
            )
            return 0.50

        # signal_dims is a dict like {"y": [H, W], "x": [C, H, W]}
        y_shape = signal_dims.get("y") or signal_dims.get("measurement")
        x_shape = signal_dims.get("x") or signal_dims.get("spectral_cube")

        # Fallback: try first and second keys if standard names not found
        if y_shape is None or x_shape is None:
            keys = list(signal_dims.keys())
            if len(keys) >= 2:
                y_shape = y_shape or signal_dims[keys[0]]
                x_shape = x_shape or signal_dims[keys[1]]
            elif len(keys) == 1:
                # Single entry: assume CR=1.0 (no compression)
                return 1.0
            else:
                return 0.50

        y_prod = prod(y_shape) if y_shape else 1
        x_prod = prod(x_shape) if x_shape else 1

        if x_prod == 0:
            logger.warning("x_shape product is zero; using default CR=0.50.")
            return 0.50

        cr = float(y_prod) / float(x_prod)

        # Sanity: clip to reasonable range
        if cr <= 0.0:
            logger.warning("Computed CR <= 0 (%.4f); clipping to 0.01.", cr)
            cr = 0.01

        return float(cr)

    # ------------------------------------------------------------------
    # Signal prior class
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signal_prior_class(
        registry: Any,
        modality_key: str,
    ) -> SignalPriorClass:
        """Look up the signal prior class from the calibration table.

        Falls back to ``SignalPriorClass.deep_prior`` if the modality
        has no entry.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.
        modality_key : str
            Imaging modality key.

        Returns
        -------
        SignalPriorClass
            Enum member representing the signal prior family.
        """
        compression_db = registry._compression
        cal_tables = compression_db.calibration_tables

        if modality_key in cal_tables:
            raw_prior = cal_tables[modality_key].signal_prior_class
            try:
                return SignalPriorClass(raw_prior)
            except ValueError:
                logger.warning(
                    "Unknown signal_prior_class '%s' in calibration table "
                    "for '%s'; falling back to deep_prior.",
                    raw_prior, modality_key,
                )

        return SignalPriorClass.deep_prior

    # ------------------------------------------------------------------
    # Operator diversity
    # ------------------------------------------------------------------

    def _compute_operator_diversity(
        self, context: AgentContext,
    ) -> float:
        """Compute a modality-specific operator diversity score.

        This is a deterministic proxy for measurement incoherence /
        pattern diversity.  The heuristics are calibrated to produce
        values in [0, 1] where higher means better diversity.

        Parameters
        ----------
        context : AgentContext
            Pipeline context with imaging system.

        Returns
        -------
        float
            Diversity score in [0, 1].
        """
        modality = context.modality_key
        system = context.imaging_system

        if modality == "spc":
            return self._spc_diversity(system)
        elif modality == "cassi":
            return self._cassi_diversity(system)
        elif modality == "mri":
            return self._mri_diversity(system)

        # Default: moderate diversity assumption
        return 0.70

    @staticmethod
    def _spc_diversity(system: Any) -> float:
        """Single-pixel camera diversity based on pattern type.

        Parameters
        ----------
        system : ImagingSystem or None
            The imaging system.

        Returns
        -------
        float
            Diversity score: 0.95 for Hadamard, 0.90 for Gaussian,
            0.70 otherwise.
        """
        if system is None:
            return 0.70

        # Search for mask element in the system
        pattern_type = _get_element_param(
            system, "mask", "pattern", default="gaussian",
        )

        if pattern_type == "hadamard":
            return 0.95  # Orthogonal basis, best incoherence
        elif pattern_type == "gaussian":
            return 0.90  # Near-optimal RIP
        else:
            return 0.70  # Binary or unknown, suboptimal

    @staticmethod
    def _cassi_diversity(system: Any) -> float:
        """CASSI diversity based on coded aperture mask density.

        Uses the Bernoulli variance heuristic: diversity is maximised
        at density=0.5 and falls off symmetrically.

        .. math::

            D = 4 \\cdot p \\cdot (1 - p)

        Parameters
        ----------
        system : ImagingSystem or None
            The imaging system.

        Returns
        -------
        float
            Diversity score in [0, 1].
        """
        if system is None:
            return 0.70

        density = _get_element_param(
            system, "mask", "density", default=0.5,
        )

        try:
            density = float(density)
        except (TypeError, ValueError):
            density = 0.5

        return float(np.clip(4.0 * density * (1.0 - density), 0.0, 1.0))

    @staticmethod
    def _mri_diversity(system: Any) -> float:
        """MRI diversity based on acceleration factor.

        Higher acceleration means fewer samples and lower diversity.

        Parameters
        ----------
        system : ImagingSystem or None
            The imaging system.

        Returns
        -------
        float
            Diversity score in [0.2, 1.0].
        """
        if system is None:
            return 0.70

        # Try system.signal_dims first, then element parameters
        acceleration = 4  # default acceleration factor
        signal_dims = getattr(system, "signal_dims", None)
        if isinstance(signal_dims, dict) and "acceleration" in signal_dims:
            acceleration = signal_dims["acceleration"]
        else:
            acceleration = _get_element_param(
                system, "modulator", "acceleration", default=4,
            )

        try:
            acceleration = float(acceleration)
        except (TypeError, ValueError):
            acceleration = 4.0

        if acceleration <= 0:
            acceleration = 1.0

        return float(np.clip(1.0 / acceleration * 4.0, 0.2, 1.0))

    # ------------------------------------------------------------------
    # Condition number proxy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_condition_number_proxy(diversity_score: float) -> float:
        """Simplified condition number proxy from diversity.

        .. math::

            \\kappa_{\\text{proxy}} = \\frac{1}{1 + D}

        where *D* is the diversity score.  Lower is better (well-
        conditioned).

        Parameters
        ----------
        diversity_score : float
            Operator diversity in [0, 1].

        Returns
        -------
        float
            Condition number proxy (approximately in [0.5, 1.0]).
        """
        return 1.0 / (1.0 + diversity_score)

    # ------------------------------------------------------------------
    # Noise regime
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_noise_regime(context: AgentContext) -> NoiseRegime:
        """Extract noise regime from the photon report or use a default.

        Parameters
        ----------
        context : AgentContext
            Pipeline context.

        Returns
        -------
        NoiseRegime
            The noise regime for the current system.
        """
        if context.photon_report is not None:
            nr = getattr(context.photon_report, "noise_regime", None)
            if nr is not None:
                if isinstance(nr, NoiseRegime):
                    return nr
                try:
                    return NoiseRegime(nr)
                except ValueError:
                    pass

        return NoiseRegime.shot_limited

    # ------------------------------------------------------------------
    # Interpolated calibration-table lookup
    # ------------------------------------------------------------------

    def _interpolated_lookup(
        self,
        registry: Any,
        modality_key: str,
        cr: float,
        noise_regime: NoiseRegime,
        solver_family: Optional[str],
    ) -> InterpolatedResult:
        """Perform interpolated calibration-table lookup.

        Strategy:

        1. Get all calibration entries for the modality.
        2. Filter by noise regime and solver (if available).
        3. Fall back progressively: noise-only, then all entries.
        4. Sort remaining entries by distance to the query CR.
        5. If exact match: confidence=1.0, uncertainty=0.5 dB.
        6. If 2+ entries: linear interpolation, confidence from distance,
           uncertainty from PSNR spread between bracket entries.
        7. If 1 entry (non-exact): confidence=0.8, uncertainty=0.5 dB.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.
        modality_key : str
            Imaging modality key.
        cr : float
            Query compression ratio.
        noise_regime : NoiseRegime
            Dominant noise regime.
        solver_family : str or None
            Solver family key for filtering (optional).

        Returns
        -------
        InterpolatedResult
            Interpolation result with confidence and uncertainty.
        """
        entries = self._get_calibration_entries(registry, modality_key)

        if not entries:
            logger.warning(
                "No calibration entries for modality '%s'; using fallback.",
                modality_key,
            )
            return InterpolatedResult(
                recoverability=_FALLBACK_RECOVERABILITY,
                expected_psnr_db=_FALLBACK_PSNR_DB,
                confidence=_FALLBACK_CONFIDENCE,
                uncertainty_db=_FALLBACK_UNCERTAINTY_DB,
                raw_entry={"source": "fallback_no_entries"},
            )

        noise_str = noise_regime.value

        # Progressive filtering
        matched = self._filter_entries(entries, noise_str, solver_family)

        # Sort by CR distance
        matched.sort(key=lambda e: abs(e.cr - cr))

        # --- Case 1: Exact or near-exact match -----------------------
        if abs(matched[0].cr - cr) < 1e-6:
            e = matched[0]
            return InterpolatedResult(
                recoverability=e.recoverability,
                expected_psnr_db=e.expected_psnr_db,
                confidence=1.0,
                uncertainty_db=0.5,
                raw_entry=e.model_dump(),
            )

        # --- Case 2: Single non-exact entry ---------------------------
        if len(matched) == 1:
            e = matched[0]
            return InterpolatedResult(
                recoverability=e.recoverability,
                expected_psnr_db=e.expected_psnr_db,
                confidence=0.8,
                uncertainty_db=0.5,
                raw_entry=e.model_dump(),
            )

        # --- Case 3: 2+ entries -- linear interpolation ---------------
        e1, e2 = matched[0], matched[1]
        return self._linear_interpolation(e1, e2, cr)

    @staticmethod
    def _get_calibration_entries(
        registry: Any, modality_key: str,
    ) -> List[Any]:
        """Retrieve calibration entries for a modality.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.
        modality_key : str
            Imaging modality key.

        Returns
        -------
        list[CalibrationEntry]
            Calibration entries, or empty list if not available.
        """
        compression_db = registry._compression
        cal_tables = compression_db.calibration_tables

        if modality_key not in cal_tables:
            return []

        return list(cal_tables[modality_key].entries)

    @staticmethod
    def _filter_entries(
        entries: List[Any],
        noise_str: str,
        solver_family: Optional[str],
    ) -> List[Any]:
        """Filter calibration entries by noise regime and solver.

        Falls back progressively:
        1. Match by noise + solver
        2. Match by noise only
        3. Use all entries

        Parameters
        ----------
        entries : list
            All calibration entries for the modality.
        noise_str : str
            Noise regime string value.
        solver_family : str or None
            Solver family key.

        Returns
        -------
        list
            Filtered (or unfiltered) entries, guaranteed non-empty.
        """
        if solver_family is not None:
            matched = [
                e for e in entries
                if e.noise == noise_str and e.solver == solver_family
            ]
            if matched:
                return matched

        # Noise-only match
        matched = [e for e in entries if e.noise == noise_str]
        if matched:
            return matched

        # Last resort: all entries
        return list(entries)

    @staticmethod
    def _linear_interpolation(
        e1: Any, e2: Any, cr: float,
    ) -> InterpolatedResult:
        """Linearly interpolate between two calibration entries.

        Parameters
        ----------
        e1 : CalibrationEntry
            Nearest entry to the query CR.
        e2 : CalibrationEntry
            Second-nearest entry.
        cr : float
            Query compression ratio.

        Returns
        -------
        InterpolatedResult
            Interpolated result with confidence from distance and
            uncertainty from PSNR spread.
        """
        cr_span = e2.cr - e1.cr

        if abs(cr_span) < 1e-10:
            alpha = 0.5
        else:
            alpha = (cr - e1.cr) / cr_span
            alpha = float(np.clip(alpha, 0.0, 1.0))

        rec = e1.recoverability * (1.0 - alpha) + e2.recoverability * alpha
        psnr = e1.expected_psnr_db * (1.0 - alpha) + e2.expected_psnr_db * alpha

        # Confidence: degrades with distance from nearest table point
        dist = min(abs(cr - e1.cr), abs(cr - e2.cr))
        conf = float(np.clip(1.0 - dist * 5.0, 0.3, 1.0))

        # Uncertainty: proportional to PSNR spread between bracket entries
        psnr_spread = abs(e2.expected_psnr_db - e1.expected_psnr_db)
        uncertainty = float(psnr_spread * 0.3)

        # Ensure uncertainty has a minimum floor
        uncertainty = max(uncertainty, 0.3)

        return InterpolatedResult(
            recoverability=float(np.clip(rec, 0.0, 1.0)),
            expected_psnr_db=float(psnr),
            confidence=conf,
            uncertainty_db=uncertainty,
            raw_entry={
                "interpolated_from": [e1.model_dump(), e2.model_dump()],
                "alpha": alpha,
            },
        )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_verdict(
        recoverability: float,
    ) -> str:
        """Determine the recoverability verdict from the score.

        Thresholds:

        - >= 0.85 : ``"excellent"``
        - >= 0.60 : ``"sufficient"``
        - >= 0.35 : ``"marginal"``
        - < 0.35  : ``"insufficient"``

        Parameters
        ----------
        recoverability : float
            Recoverability score in [0, 1].

        Returns
        -------
        str
            One of ``"excellent"``, ``"sufficient"``, ``"marginal"``,
            ``"insufficient"``.
        """
        if recoverability >= 0.85:
            return "excellent"
        elif recoverability >= 0.60:
            return "sufficient"
        elif recoverability >= 0.35:
            return "marginal"
        else:
            return "insufficient"

    # ------------------------------------------------------------------
    # Recommended solver
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_solver(
        registry: Any,
        modality_key: str,
        cr: float,
        noise_regime: NoiseRegime,
    ) -> str:
        """Find the best solver from the calibration table.

        Selects the solver with the highest expected PSNR among
        entries closest to the query CR and matching the noise regime.
        Falls back to the modality's default solver.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.
        modality_key : str
            Imaging modality key.
        cr : float
            Query compression ratio.
        noise_regime : NoiseRegime
            Dominant noise regime.

        Returns
        -------
        str
            Solver family key.
        """
        # Try to get default solver from modalities registry
        default_solver = "gap_tv"
        try:
            modality_spec = registry._modalities.modalities.get(modality_key)
            if modality_spec is not None:
                default_solver = modality_spec.default_solver
        except (AttributeError, KeyError):
            pass

        # Search calibration table for best solver
        compression_db = registry._compression
        cal_tables = compression_db.calibration_tables

        if modality_key not in cal_tables:
            return default_solver

        entries = cal_tables[modality_key].entries
        noise_str = noise_regime.value

        # Filter by noise regime
        noise_matched = [e for e in entries if e.noise == noise_str]
        if not noise_matched:
            noise_matched = list(entries)

        if not noise_matched:
            return default_solver

        # Sort by CR distance, then by PSNR (descending)
        noise_matched.sort(
            key=lambda e: (abs(e.cr - cr), -e.expected_psnr_db),
        )

        return noise_matched[0].solver

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        modality_key: str,
        cr: float,
        noise_regime: NoiseRegime,
        signal_prior_class: SignalPriorClass,
        diversity_score: float,
        condition_proxy: float,
        recoverability: float,
        confidence: float,
        expected_psnr_db: float,
        uncertainty_db: float,
        verdict: str,
        recommended_solver: str,
    ) -> str:
        """Build a deterministic human-readable explanation.

        Parameters
        ----------
        modality_key : str
            Imaging modality.
        cr : float
            Compression ratio.
        noise_regime : NoiseRegime
            Noise regime enum.
        signal_prior_class : SignalPriorClass
            Signal prior family.
        diversity_score : float
            Operator diversity score.
        condition_proxy : float
            Condition number proxy.
        recoverability : float
            Recoverability score.
        confidence : float
            Lookup confidence.
        expected_psnr_db : float
            Expected PSNR.
        uncertainty_db : float
            PSNR uncertainty band.
        verdict : str
            Recoverability verdict.
        recommended_solver : str
            Best solver family.

        Returns
        -------
        str
            Multi-line explanation string.
        """
        return (
            f"Recoverability analysis for modality '{modality_key}':\n"
            f"  Compression ratio: {cr:.4f}\n"
            f"  Noise regime: {noise_regime.value}\n"
            f"  Signal prior: {signal_prior_class.value}\n"
            f"  Operator diversity: {diversity_score:.3f}\n"
            f"  Condition proxy: {condition_proxy:.3f}\n"
            f"  Recoverability: {recoverability:.3f} "
            f"(confidence: {confidence:.2f})\n"
            f"  Expected PSNR: {expected_psnr_db:.1f} +/- "
            f"{uncertainty_db:.1f} dB\n"
            f"  Verdict: {verdict}\n"
            f"  Recommended solver: {recommended_solver}"
        )


# ======================================================================
# Module-level helpers
# ======================================================================


def _get_element_param(
    system: Any,
    element_type: str,
    param_name: str,
    default: Any = None,
) -> Any:
    """Extract a parameter from the first element of a given type.

    Searches the imaging system's element list for the first element
    whose ``element_type`` matches, then returns
    ``element.parameters[param_name]``.

    Parameters
    ----------
    system : ImagingSystem or similar
        Object with an ``elements`` attribute (list of ``ElementSpec``).
    element_type : str
        Element type to search for (e.g. ``"mask"``, ``"modulator"``).
    param_name : str
        Parameter key to extract.
    default : Any
        Value to return if the element or parameter is not found.

    Returns
    -------
    Any
        The parameter value, or *default*.
    """
    elements = getattr(system, "elements", None)
    if elements is None:
        return default

    for elem in elements:
        etype = getattr(elem, "element_type", None)
        if etype == element_type:
            params = getattr(elem, "parameters", {})
            if isinstance(params, dict):
                return params.get(param_name, default)
            return default

    return default
