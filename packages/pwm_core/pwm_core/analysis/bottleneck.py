"""pwm_core.analysis.bottleneck

A simple bottleneck classifier:
- dose-limited
- sampling-limited
- drift-limited
- calibration/mismatch-limited
"""

from __future__ import annotations

from typing import Any, Dict


def classify_bottleneck(diag: Dict[str, Any]) -> Dict[str, Any]:
    # Expect caller to provide evidence keys like residual_rms, saturation_risk, sampling_rate, etc.
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
        verdict = "Dose Limited" if saturation_risk < 0.1 and residual_rms > 0.05 else "OK"

    return {
        "verdict": verdict,
        "confidence": 0.7,
        "evidence": e,
    }
