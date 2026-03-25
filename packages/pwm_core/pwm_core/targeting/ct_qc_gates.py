"""pwm_core.targeting.ct_qc_gates

CT QC Copilot — Domain Judge gates for CT scanner quality control.

These gates extend the universal S1-S4 kernel with CT-specific QC checks.
They are loaded only when the active DomainProfile is ct_qc/v1.

Public API
----------
    run_ct_qc_gates(qc_metrics: dict) -> list[GateResult]
    check_drift(current_metrics: dict, baseline_metrics: dict,
                threshold_sigma: float = 2.5) -> list[DriftAlert]
    classify_artifacts(image: np.ndarray) -> list[ArtifactFlag]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# QC gate thresholds (mirrors ct_qc/v1/profile.yaml)
_QC_THRESHOLDS = {
    "HU_noise_sd":               {"warn": 8.0,  "fail": 12.0,  "direction": "high"},
    "HU_uniformity_max_deviation": {"warn": 4.0,  "fail": 8.0,   "direction": "high"},
    "CNR":                         {"warn": 3.0,  "fail": 1.5,   "direction": "low"},
    "MTF10_lpmm":                  {"warn": 4.5,  "fail": 3.5,   "direction": "low"},
    "HU_accuracy_max_error":       {"warn": 7.0,  "fail": 15.0,  "direction": "high"},
}


@dataclass
class QCGateResult:
    gate_id: str
    metric: str
    value: float
    verdict: str        # "pass", "warn", "fail"
    message: str
    threshold_warn: float
    threshold_fail: float


@dataclass
class DriftAlert:
    metric: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    verdict: str        # "ok", "warn", "alert"
    message: str


@dataclass
class ArtifactFlag:
    artifact_id: str
    display_name: str
    detected: bool
    severity: str       # "info", "warn", "fail"
    confidence: float   # 0-1


def run_ct_qc_gates(qc_metrics: Dict[str, float]) -> List[QCGateResult]:
    """Run all CT QC gates against the provided metric dict.

    Parameters
    ----------
    qc_metrics:
        Dict with keys matching _QC_THRESHOLDS (e.g., HU_noise_sd, CNR, ...).
        Missing keys produce a warn verdict.

    Returns
    -------
    List of QCGateResult, one per defined threshold metric.
    """
    results = []
    for metric, thresholds in _QC_THRESHOLDS.items():
        if metric not in qc_metrics:
            results.append(QCGateResult(
                gate_id=f"qc_{metric.lower()}",
                metric=metric,
                value=float("nan"),
                verdict="warn",
                message=f"Metric '{metric}' not provided — cannot evaluate gate.",
                threshold_warn=thresholds["warn"],
                threshold_fail=thresholds["fail"],
            ))
            continue

        value = float(qc_metrics[metric])
        direction = thresholds["direction"]

        if direction == "high":
            # High value = bad
            if value >= thresholds["fail"]:
                verdict, msg = "fail", f"{metric}={value:.2f} exceeds fail threshold {thresholds['fail']}"
            elif value >= thresholds["warn"]:
                verdict, msg = "warn", f"{metric}={value:.2f} exceeds warn threshold {thresholds['warn']}"
            else:
                verdict, msg = "pass", f"{metric}={value:.2f} within normal range"
        else:
            # Low value = bad
            if value <= thresholds["fail"]:
                verdict, msg = "fail", f"{metric}={value:.2f} below fail threshold {thresholds['fail']}"
            elif value <= thresholds["warn"]:
                verdict, msg = "warn", f"{metric}={value:.2f} below warn threshold {thresholds['warn']}"
            else:
                verdict, msg = "pass", f"{metric}={value:.2f} within normal range"

        results.append(QCGateResult(
            gate_id=f"qc_{metric.lower()}",
            metric=metric,
            value=value,
            verdict=verdict,
            message=msg,
            threshold_warn=thresholds["warn"],
            threshold_fail=thresholds["fail"],
        ))

    return results


def check_drift(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, List[float]],
    threshold_sigma: float = 2.5,
) -> List[DriftAlert]:
    """Detect metric drift against a rolling baseline.

    Parameters
    ----------
    current_metrics:
        Current QC measurement dict.
    baseline_metrics:
        Dict mapping metric name -> list of historical values (last 30 days).
    threshold_sigma:
        Z-score threshold for drift alert. Default 2.5 sigma.

    Returns
    -------
    List of DriftAlert, one per metric with sufficient baseline history.
    """
    alerts = []
    for metric, current_value in current_metrics.items():
        history = baseline_metrics.get(metric, [])
        if len(history) < 5:
            continue  # Not enough history for meaningful drift detection

        baseline_mean = float(np.mean(history))
        baseline_std = float(np.std(history, ddof=1))
        if baseline_std < 1e-9:
            z_score = 0.0
        else:
            z_score = abs(current_value - baseline_mean) / baseline_std

        if z_score >= threshold_sigma:
            verdict = "alert"
            msg = (f"DRIFT DETECTED: {metric}={current_value:.2f} deviates "
                   f"{z_score:.1f}sigma from baseline mean={baseline_mean:.2f}")
        elif z_score >= threshold_sigma * 0.7:
            verdict = "warn"
            msg = (f"Drift warning: {metric}={current_value:.2f}, "
                   f"z-score={z_score:.1f} (threshold={threshold_sigma}sigma)")
        else:
            verdict = "ok"
            msg = f"{metric}={current_value:.2f} within baseline range (z={z_score:.1f})"

        alerts.append(DriftAlert(
            metric=metric,
            current_value=current_value,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            z_score=z_score,
            verdict=verdict,
            message=msg,
        ))

    return alerts


def classify_artifacts(image: np.ndarray) -> List[ArtifactFlag]:
    """Lightweight artifact classifier for CT images.

    Uses frequency-domain analysis for ring detection and spatial statistics
    for cupping. Metal/motion streaks use directional edge statistics.

    Parameters
    ----------
    image:
        2D CT image array, float32, HU-calibrated or normalized.

    Returns
    -------
    List of ArtifactFlag (one per artifact type).
    """
    flags = []

    # Ring artifact: detect circular periodicity in frequency domain
    ring_confidence = _detect_ring_artifact(image)
    flags.append(ArtifactFlag(
        artifact_id="ring_artifact",
        display_name="Ring Artifact",
        detected=ring_confidence > 0.5,
        severity="warn" if ring_confidence > 0.5 else "info",
        confidence=ring_confidence,
    ))

    # Cupping artifact: center vs. periphery HU gradient
    cupping_confidence = _detect_cupping(image)
    flags.append(ArtifactFlag(
        artifact_id="cupping_artifact",
        display_name="Beam Hardening / Cupping",
        detected=cupping_confidence > 0.4,
        severity="warn" if cupping_confidence > 0.4 else "info",
        confidence=cupping_confidence,
    ))

    # Streak artifact: high directional edge energy
    streak_confidence = _detect_streaks(image)
    flags.append(ArtifactFlag(
        artifact_id="streak_artifact",
        display_name="Streak Artifact",
        detected=streak_confidence > 0.6,
        severity="warn" if streak_confidence > 0.6 else "info",
        confidence=streak_confidence,
    ))

    return flags


# ---------------------------------------------------------------------------
# Artifact detection helpers (lightweight, no external dependencies)
# ---------------------------------------------------------------------------

def _detect_ring_artifact(image: np.ndarray) -> float:
    """Estimate ring artifact confidence via radial frequency analysis."""
    if image.ndim != 2:
        return 0.0
    h, w = image.shape
    cx, cy = h // 2, w // 2
    # Convert to polar coordinates (simplified via FFT of radial profiles)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    # High energy in narrow annular frequency bands -> ring artifact
    ring_region = fft_mag[cx - 5:cx + 5, :]
    outer_region = fft_mag[cx - 50:cx - 10, :]
    if outer_region.size == 0 or ring_region.mean() == 0:
        return 0.0
    ratio = float(ring_region.max() / (outer_region.mean() + 1e-9))
    return float(min(1.0, ratio / 20.0))


def _detect_cupping(image: np.ndarray) -> float:
    """Estimate cupping artifact by comparing center vs peripheral mean."""
    if image.ndim != 2:
        return 0.0
    h, w = image.shape
    cx, cy = h // 2, w // 2
    r_center = max(1, h // 8)
    r_outer_start = max(r_center + 1, h // 4)
    r_outer_end = max(r_outer_start + 1, h // 3)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    center_mask = r < r_center
    outer_mask = (r >= r_outer_start) & (r < r_outer_end)

    if center_mask.sum() == 0 or outer_mask.sum() == 0:
        return 0.0

    center_mean = float(image[center_mask].mean())
    outer_mean = float(image[outer_mask].mean())
    diff = abs(center_mean - outer_mean)
    # Normalize: +-15 HU difference -> high cupping confidence
    return float(min(1.0, diff / 15.0))


def _detect_streaks(image: np.ndarray) -> float:
    """Estimate streak artifact via anisotropic edge energy."""
    if image.ndim != 2:
        return 0.0
    # Horizontal vs vertical gradient energy ratio
    gy = np.diff(image.astype(np.float32), axis=0)
    gx = np.diff(image.astype(np.float32), axis=1)
    gy_energy = float(np.mean(gy ** 2))
    gx_energy = float(np.mean(gx ** 2))
    total = gy_energy + gx_energy
    if total < 1e-9:
        return 0.0
    anisotropy = abs(gy_energy - gx_energy) / total
    return float(min(1.0, anisotropy * 2.0))
