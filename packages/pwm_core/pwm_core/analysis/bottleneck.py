"""pwm_core.analysis.bottleneck

Ranked bottleneck diagnosis across four subsystems:

    Photon x Recoverability x Mismatch x Solver Fit

Each factor receives a severity score in [0, 1] and an estimated gain
(in dB) that would be realised by fixing it.  The output includes
a ranked list and a plain-language "what to change first" recommendation.

The original ``classify_bottleneck()`` function is preserved for backward
compatibility, but new callers should prefer ``rank_bottlenecks()`` which
returns the richer ``BottleneckDiagnosis`` dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Legacy classifier (kept for backward compatibility)
# ---------------------------------------------------------------------------


def classify_bottleneck(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy bottleneck classifier (backward-compatible).

    Parameters
    ----------
    diag : dict
        Must contain an ``"evidence"`` sub-dict with optional keys
        ``residual_rms``, ``saturation_risk``, ``sampling_rate``,
        ``drift_detected``.

    Returns
    -------
    dict
        Keys: ``verdict``, ``confidence``, ``evidence``.
    """
    e = diag.get("evidence", {}) if isinstance(diag, dict) else {}
    residual_rms = float(e.get("residual_rms", 0.0))
    saturation_risk = float(e.get("saturation_risk", 0.0))
    sampling_rate = float(e.get("sampling_rate", 1.0))
    drift = float(e.get("drift_detected", 0.0))

    verdict = "Unknown"
    if sampling_rate < 0.15:
        verdict = "Sampling Limited"
    elif residual_rms > 0.2:
        verdict = "Calibration/Mismatch Limited"
    elif drift > 0.5:
        verdict = "Drift Limited"
    else:
        verdict = (
            "Dose Limited"
            if saturation_risk < 0.1 and residual_rms > 0.05
            else "OK"
        )

    return {
        "verdict": verdict,
        "confidence": 0.7,
        "evidence": e,
    }


# ---------------------------------------------------------------------------
# Enhanced ranked bottleneck diagnosis
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))


def _severity_to_expected_gain(severity: float, max_gain_db: float) -> float:
    """Map a severity score in [0, 1] to an expected dB gain.

    A quadratic model: gain = max_gain_db * severity^2.  Fixing a
    severity-1.0 factor yields the full ``max_gain_db``; fixing a 0.0
    factor yields nothing.
    """
    return max_gain_db * severity * severity


def rank_bottlenecks(
    *,
    photon_severity: float = 0.0,
    recoverability_severity: float = 0.0,
    mismatch_severity: float = 0.0,
    solver_fit_severity: float = 0.0,
    # Optional context for richer recommendations
    snr_db: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    mismatch_family: Optional[str] = None,
    solver_family: Optional[str] = None,
    correction_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rank the four imaging-pipeline bottleneck factors and produce
    actionable recommendations.

    Parameters
    ----------
    photon_severity : float
        Severity of photon-budget limitations (0 = no issue, 1 = severe).
    recoverability_severity : float
        Severity of recoverability / compressive-sensing limitations.
    mismatch_severity : float
        Severity of operator mismatch / calibration error.
    solver_fit_severity : float
        Severity of solver-data mismatch (wrong solver family, too few
        iterations, etc.).
    snr_db : float, optional
        System SNR in dB, used for photon recommendation detail.
    compression_ratio : float, optional
        System compression ratio, used for recoverability detail.
    mismatch_family : str, optional
        Active mismatch family name (e.g. ``"dispersion_step"``).
    solver_family : str, optional
        Active solver family name (e.g. ``"gap_tv"``).
    correction_result : dict, optional
        If a CorrectionResult (or dict equivalent) is available, pass
        it here so the recommendation can reference uncertainty bands.

    Returns
    -------
    dict
        Keys:

        ``ranked`` : list[dict]
            Factors sorted from most to least severe.  Each entry has
            ``factor``, ``severity``, ``expected_gain_db``, and
            ``recommendation``.
        ``primary_bottleneck`` : str
            Name of the most severe factor.
        ``what_to_change_first`` : str
            Plain-language recommendation for the single highest-impact
            action.
        ``scores`` : dict[str, float]
            Raw severity scores for all four factors.
    """
    # -- Clamp inputs --------------------------------------------------------
    factors: Dict[str, float] = {
        "photon": _clamp01(photon_severity),
        "recoverability": _clamp01(recoverability_severity),
        "mismatch": _clamp01(mismatch_severity),
        "solver_fit": _clamp01(solver_fit_severity),
    }

    # Maximum achievable gains per factor (empirical ceiling in dB).
    # These are domain-typical ceilings observed across the 26 modalities.
    max_gains: Dict[str, float] = {
        "photon": 6.0,
        "recoverability": 8.0,
        "mismatch": 12.0,
        "solver_fit": 5.0,
    }

    # -- Build ranked list ---------------------------------------------------
    ranked: List[Dict[str, Any]] = []
    for factor, severity in factors.items():
        expected_gain = _severity_to_expected_gain(severity, max_gains[factor])
        recommendation = _make_recommendation(
            factor=factor,
            severity=severity,
            snr_db=snr_db,
            compression_ratio=compression_ratio,
            mismatch_family=mismatch_family,
            solver_family=solver_family,
            correction_result=correction_result,
        )
        ranked.append(
            {
                "factor": factor,
                "severity": round(severity, 4),
                "expected_gain_db": round(expected_gain, 2),
                "recommendation": recommendation,
            }
        )

    # Sort by expected gain descending (highest impact first)
    ranked.sort(key=lambda d: d["expected_gain_db"], reverse=True)

    primary = ranked[0]["factor"] if ranked else "unknown"
    what_first = ranked[0]["recommendation"] if ranked else "No actionable recommendation."

    return {
        "ranked": ranked,
        "primary_bottleneck": primary,
        "what_to_change_first": what_first,
        "scores": {f: round(s, 4) for f, s in factors.items()},
    }


# ---------------------------------------------------------------------------
# Recommendation text generation (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _make_recommendation(
    *,
    factor: str,
    severity: float,
    snr_db: Optional[float],
    compression_ratio: Optional[float],
    mismatch_family: Optional[str],
    solver_family: Optional[str],
    correction_result: Optional[Dict[str, Any]],
) -> str:
    """Generate a plain-language recommendation for one factor.

    Returns a 1-2 sentence actionable suggestion.  No LLM is involved.
    """
    if severity < 0.1:
        return f"{factor.replace('_', ' ').title()} is not a significant bottleneck."

    if factor == "photon":
        base = "Increase photon budget."
        if snr_db is not None and snr_db < 15.0:
            base += (
                f" Current SNR ({snr_db:.1f} dB) is below the 15 dB "
                "marginal threshold. Consider longer exposure time, "
                "higher source power, or reduced background."
            )
        elif snr_db is not None and snr_db < 25.0:
            base += (
                f" SNR ({snr_db:.1f} dB) is acceptable but could benefit "
                "from improved collection efficiency or averaging."
            )
        else:
            base += (
                " Review throughput chain for lossy elements or "
                "consider binning to trade resolution for SNR."
            )
        return base

    if factor == "recoverability":
        base = "Improve measurement diversity or prior strength."
        if compression_ratio is not None and compression_ratio > 8.0:
            base += (
                f" Compression ratio ({compression_ratio:.1f}x) is high. "
                "Capture more snapshots, add coded aperture diversity, "
                "or switch to a stronger learned prior."
            )
        elif compression_ratio is not None and compression_ratio > 4.0:
            base += (
                f" Compression ratio ({compression_ratio:.1f}x) is "
                "moderate. Consider adding temporal frames or using a "
                "joint spatio-spectral prior."
            )
        else:
            base += (
                " Check operator condition number and consider using "
                "a regularisation-aware solver."
            )
        return base

    if factor == "mismatch":
        base = "Reduce operator-model mismatch through calibration."
        if correction_result is not None:
            theta_unc = correction_result.get("theta_uncertainty", {})
            wide_params = [
                p for p, ci in theta_unc.items()
                if isinstance(ci, (list, tuple))
                and len(ci) == 2
                and (ci[1] - ci[0]) > 0.5
            ]
            if wide_params:
                base += (
                    f" Parameters with wide uncertainty: "
                    f"{', '.join(wide_params)}. Capture additional "
                    f"calibration data targeting these parameters."
                )
            else:
                base += (
                    " Run bootstrap_correction() to quantify parameter "
                    "uncertainty, then use capture_advisor for next steps."
                )
        elif mismatch_family:
            base += (
                f" Active mismatch family: {mismatch_family}. "
                "Run the correction loop and check convergence."
            )
        else:
            base += (
                " Identify the dominant mismatch source (alignment, "
                "dispersion, PSF drift) and run operator correction."
            )
        return base

    if factor == "solver_fit":
        base = "Improve solver choice or tuning."
        if solver_family:
            base += (
                f" Current solver: {solver_family}. "
                "Try increasing iterations, tuning regularisation "
                "weight, or switching to a solver better matched to "
                "the noise regime and signal prior."
            )
        else:
            base += (
                " Consult the solver registry for alternatives. "
                "Ensure the solver's assumptions (noise model, prior "
                "class) match the actual imaging conditions."
            )
        return base

    return f"Address {factor.replace('_', ' ')} (severity={severity:.2f})."
