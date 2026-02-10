"""pwm_core.mismatch.capture_advisor

When calibration is underdetermined (wide uncertainty bands), suggest
what additional measurements to capture, recommended geometry/parameters,
and expected uncertainty reduction.

All advice is deterministic (no LLM).  Recommendations are based on the
``CorrectionResult`` uncertainty bands and, optionally, the imaging system
specification from the registry.

Usage
-----
::

    from pwm_core.mismatch.capture_advisor import suggest_next_capture

    advice = suggest_next_capture(correction_result, system_spec)
    for s in advice.suggestions:
        print(s["parameter"], s["action"], s["expected_reduction_pct"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold for "wide" uncertainty band per parameter family
# ---------------------------------------------------------------------------

# Default CI width threshold (in parameter units) above which we consider
# the parameter underdetermined.  Callers can override via system_spec.
_DEFAULT_WIDTH_THRESHOLD = 0.5

# Known parameter families and their domain-specific thresholds + advice.
_PARAM_ADVICE: Dict[str, Dict[str, Any]] = {
    "gain": {
        "threshold": 0.3,
        "capture_action": (
            "Capture a flat-field image (uniform illumination) to "
            "constrain the gain parameter."
        ),
        "geometry": "uniform_illumination",
        "expected_reduction_pct": 60,
    },
    "bias": {
        "threshold": 0.1,
        "capture_action": (
            "Capture a dark frame (no illumination, shutter closed) "
            "to constrain the bias/offset parameter."
        ),
        "geometry": "dark_frame",
        "expected_reduction_pct": 70,
    },
    "dx": {
        "threshold": 1.0,
        "capture_action": (
            "Capture a calibration target with known horizontal "
            "features (e.g. USAF resolution chart) to constrain "
            "the x-shift parameter."
        ),
        "geometry": "resolution_target_horizontal",
        "expected_reduction_pct": 50,
    },
    "dy": {
        "threshold": 1.0,
        "capture_action": (
            "Capture a calibration target with known vertical "
            "features to constrain the y-shift parameter."
        ),
        "geometry": "resolution_target_vertical",
        "expected_reduction_pct": 50,
    },
    "dx0": {
        "threshold": 1.0,
        "capture_action": (
            "Capture a point source or slit at a known position to "
            "constrain the global x-offset."
        ),
        "geometry": "point_source_centered",
        "expected_reduction_pct": 55,
    },
    "dy0": {
        "threshold": 1.0,
        "capture_action": (
            "Capture a point source or slit at a known position to "
            "constrain the global y-offset."
        ),
        "geometry": "point_source_centered",
        "expected_reduction_pct": 55,
    },
    "cor_shift": {
        "threshold": 1.0,
        "capture_action": (
            "Acquire a sinogram of a known phantom (e.g. wire phantom "
            "or pin) at 0 and 180 degrees to determine the center of "
            "rotation offset."
        ),
        "geometry": "opposing_projections_0_180",
        "expected_reduction_pct": 70,
    },
    "disp_step": {
        "threshold": 0.5,
        "capture_action": (
            "Capture a narrowband source at 3+ known wavelengths to "
            "constrain the dispersion polynomial. Space wavelengths "
            "across the full spectral range."
        ),
        "geometry": "narrowband_wavelength_sweep",
        "expected_reduction_pct": 65,
    },
    "disp_poly_x_0": {
        "threshold": 2.0,
        "capture_action": (
            "Capture a monochromatic source at the center wavelength "
            "to constrain the zero-order dispersion offset."
        ),
        "geometry": "monochromatic_center_wavelength",
        "expected_reduction_pct": 60,
    },
    "disp_poly_x_1": {
        "threshold": 1.0,
        "capture_action": (
            "Capture two narrowband sources at the spectral extremes "
            "to constrain the linear dispersion coefficient."
        ),
        "geometry": "two_wavelength_extremes",
        "expected_reduction_pct": 55,
    },
    "timing_offset": {
        "threshold": 0.5,
        "capture_action": (
            "Capture a high-contrast temporal calibration pattern "
            "(e.g. LED flash sequence) to pin mask timing offset."
        ),
        "geometry": "temporal_flash_sequence",
        "expected_reduction_pct": 60,
    },
    "position_offset": {
        "threshold": 0.5,
        "capture_action": (
            "Capture a known grid pattern and use cross-correlation "
            "to determine probe position offset."
        ),
        "geometry": "calibration_grid",
        "expected_reduction_pct": 55,
    },
    "psf_shift": {
        "threshold": 0.5,
        "capture_action": (
            "Capture a point source (pinhole) at a known position "
            "to measure the actual PSF center and constrain the shift."
        ),
        "geometry": "pinhole_point_source",
        "expected_reduction_pct": 65,
    },
}


@dataclass
class CaptureAdvice:
    """Actionable next-capture advice produced by ``suggest_next_capture``.

    Attributes
    ----------
    suggestions : list[dict]
        One entry per underdetermined parameter.  Each dict has keys:
        ``parameter``, ``ci_width``, ``threshold``, ``action``,
        ``geometry``, ``expected_reduction_pct``.
    summary : str
        One-paragraph summary of all suggestions.
    n_underdetermined : int
        Number of parameters with uncertainty wider than threshold.
    all_parameters_constrained : bool
        True if every parameter's CI is below its threshold.
    """

    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    n_underdetermined: int = 0
    all_parameters_constrained: bool = True


def suggest_next_capture(
    correction_result: Any,
    system_spec: Optional[Dict[str, Any]] = None,
) -> CaptureAdvice:
    """Suggest what additional measurements to capture to reduce
    calibration uncertainty.

    Parameters
    ----------
    correction_result : CorrectionResult or dict
        Must contain ``theta_corrected`` and ``theta_uncertainty`` fields.
        Can be a pydantic CorrectionResult or a plain dict with the same
        keys.
    system_spec : dict, optional
        Imaging system specification.  If provided, may contain
        ``"width_thresholds"`` (dict mapping parameter names to custom
        thresholds) and ``"modality_key"`` (str) for modality-aware
        advice.

    Returns
    -------
    CaptureAdvice
        Actionable suggestions for next captures.
    """
    # -- Extract fields (handle both pydantic model and dict) ---------------
    if hasattr(correction_result, "theta_corrected"):
        theta_corrected = correction_result.theta_corrected
        theta_uncertainty = correction_result.theta_uncertainty
    elif isinstance(correction_result, dict):
        theta_corrected = correction_result.get("theta_corrected", {})
        theta_uncertainty = correction_result.get("theta_uncertainty", {})
    else:
        logger.warning(
            "correction_result has unexpected type %s; returning empty advice.",
            type(correction_result).__name__,
        )
        return CaptureAdvice(summary="Unable to parse correction result.")

    if not theta_uncertainty:
        return CaptureAdvice(
            summary="No uncertainty data available. Run bootstrap_correction() first.",
        )

    # -- Custom thresholds from system_spec ---------------------------------
    custom_thresholds: Dict[str, float] = {}
    if system_spec and isinstance(system_spec, dict):
        custom_thresholds = system_spec.get("width_thresholds", {})

    # -- Evaluate each parameter --------------------------------------------
    suggestions: List[Dict[str, Any]] = []

    for param, ci in theta_uncertainty.items():
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
        else:
            continue

        ci_width = ci_hi - ci_lo

        # Determine threshold for this parameter
        if param in custom_thresholds:
            threshold = float(custom_thresholds[param])
        elif param in _PARAM_ADVICE:
            threshold = _PARAM_ADVICE[param]["threshold"]
        else:
            threshold = _DEFAULT_WIDTH_THRESHOLD

        if ci_width <= threshold:
            continue  # parameter is sufficiently constrained

        # Build suggestion
        advice_entry = _PARAM_ADVICE.get(param, {})
        action = advice_entry.get(
            "capture_action",
            f"Capture additional calibration data that exercises the "
            f"'{param}' parameter across its expected operating range "
            f"({ci_lo:.3f} to {ci_hi:.3f}).",
        )
        geometry = advice_entry.get("geometry", "general_calibration")
        expected_reduction = advice_entry.get("expected_reduction_pct", 40)

        suggestions.append(
            {
                "parameter": param,
                "ci_width": round(ci_width, 4),
                "threshold": round(threshold, 4),
                "current_ci": [round(ci_lo, 4), round(ci_hi, 4)],
                "corrected_value": round(
                    float(theta_corrected.get(param, 0.0)), 4
                ),
                "action": action,
                "geometry": geometry,
                "expected_reduction_pct": expected_reduction,
            }
        )

    # Sort by CI width descending (widest = most benefit from new data)
    suggestions.sort(key=lambda s: s["ci_width"], reverse=True)

    n_underdetermined = len(suggestions)
    all_constrained = n_underdetermined == 0

    # -- Build summary -------------------------------------------------------
    if all_constrained:
        summary = (
            "All calibration parameters are well-constrained within their "
            "thresholds. No additional captures are needed."
        )
    else:
        param_names = [s["parameter"] for s in suggestions]
        summary = (
            f"{n_underdetermined} parameter(s) have uncertainty bands "
            f"exceeding their thresholds: {', '.join(param_names)}. "
        )
        if suggestions:
            top = suggestions[0]
            summary += (
                f"Highest priority: capture data for '{top['parameter']}' "
                f"(CI width = {top['ci_width']:.3f}, threshold = "
                f"{top['threshold']:.3f}). "
                f"Recommended geometry: {top['geometry']}. "
                f"Expected uncertainty reduction: "
                f"~{top['expected_reduction_pct']}%."
            )

    return CaptureAdvice(
        suggestions=suggestions,
        summary=summary,
        n_underdetermined=n_underdetermined,
        all_parameters_constrained=all_constrained,
    )
