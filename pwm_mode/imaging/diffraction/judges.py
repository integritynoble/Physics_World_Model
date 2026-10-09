"""Independent diffraction judge functions; no solver/library imports."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

CRITERION_VERSION = "1.0.0"


def judge_saxs_simulate(submission: Any, key: dict[str, Any], criterion: dict[str, Any]) -> dict[str, Any]:
    """Score a predicted one-dimensional SAXS reference curve against its hidden key."""
    if not isinstance(submission, dict) or "curve" not in submission:
        return {"pass": False, "metrics": {}, "reasons": ["submission must contain curve"]}
    try:
        curve = np.asarray(submission["curve"], dtype=np.float64)
    except (TypeError, ValueError):
        return {"pass": False, "metrics": {}, "reasons": ["curve must be numeric"]}
    ref = np.asarray(key["reference_curve"], dtype=np.float64)
    if curve.shape != ref.shape or curve.ndim != 1 or not np.all(np.isfinite(curve)):
        return {"pass": False, "metrics": {}, "reasons": ["curve has wrong shape or non-finite values"]}
    digest = hashlib.sha256(np.ascontiguousarray(curve.astype("<f8")).tobytes()).hexdigest()
    if digest in set(key.get("forbidden_public_curve_sha256", [])):
        return {"pass": False, "metrics": {"rel_l2": 0.0, "copied_public_curve": True},
                "reasons": ["exact copy of a public development curve"]}
    scale = max(float(np.linalg.norm(ref)), 1e-12)
    rel = float(np.linalg.norm(curve - ref) / scale)
    passed = math.isfinite(rel) and rel <= float(criterion["relative_l2_max"])
    return {"pass": passed, "metrics": {"rel_l2": rel, "copied_public_curve": False},
            "reasons": [] if passed else [f"relative L2 {rel:.5f} exceeds {criterion['relative_l2_max']:.5f}"]}


def judge_atom_probe_simulate(submission: Any, key: dict[str, Any], criterion: dict[str, Any]) -> dict[str, Any]:
    """Score a predicted APT mass-to-charge spectrum against its hidden reference counts."""
    if not isinstance(submission, dict) or "spectrum" not in submission:
        return {"pass": False, "metrics": {}, "reasons": ["submission must contain spectrum"]}
    try:
        curve = np.asarray(submission["spectrum"], dtype=np.float64)
    except (TypeError, ValueError):
        return {"pass": False, "metrics": {}, "reasons": ["spectrum must be numeric"]}
    ref = np.asarray(key["reference_spectrum"], dtype=np.float64)
    if curve.shape != ref.shape or curve.ndim != 1 or not np.all(np.isfinite(curve)) or np.any(curve < 0):
        return {"pass": False, "metrics": {}, "reasons": ["spectrum has wrong shape, negative counts, or non-finite values"]}
    digest = hashlib.sha256(np.ascontiguousarray(curve.astype("<f8")).tobytes()).hexdigest()
    if digest in set(key.get("forbidden_public_spectrum_sha256", [])):
        return {"pass": False, "metrics": {"rel_l2": 0.0, "copied_public_spectrum": True},
                "reasons": ["exact copy of a public development spectrum"]}
    rel = float(np.linalg.norm(curve - ref) / max(float(np.linalg.norm(ref)), 1e-12))
    passed = math.isfinite(rel) and rel <= float(criterion["relative_l2_max"])
    return {"pass": passed, "metrics": {"rel_l2": rel, "copied_public_spectrum": False},
            "reasons": [] if passed else [f"relative L2 {rel:.5f} exceeds {criterion['relative_l2_max']:.5f}"]}


def judge_cathodoluminescence_simulate(submission: Any, key: dict[str, Any], criterion: dict[str, Any]) -> dict[str, Any]:
    """Score a predicted CL spectral curve against a hidden measured reference."""
    if not isinstance(submission, dict) or "curve" not in submission:
        return {"pass": False, "metrics": {}, "reasons": ["submission must contain curve"]}
    try:
        curve = np.asarray(submission["curve"], dtype=np.float64)
    except (TypeError, ValueError):
        return {"pass": False, "metrics": {}, "reasons": ["curve must be numeric"]}
    ref = np.asarray(key["reference_curve"], dtype=np.float64)
    if curve.shape != ref.shape or curve.ndim != 1 or not np.all(np.isfinite(curve)):
        return {"pass": False, "metrics": {}, "reasons": ["curve has wrong shape or non-finite values"]}
    digest = hashlib.sha256(np.ascontiguousarray(curve.astype("<f8")).tobytes()).hexdigest()
    if digest in set(key.get("forbidden_public_curve_sha256", [])):
        return {"pass": False, "metrics": {"rel_l2": 0.0, "copied_public_curve": True},
                "reasons": ["exact copy of a public development curve"]}
    rel = float(np.linalg.norm(curve - ref) / max(float(np.linalg.norm(ref)), 1e-12))
    passed = math.isfinite(rel) and rel <= float(criterion["relative_l2_max"])
    return {"pass": passed, "metrics": {"rel_l2": rel, "copied_public_curve": False},
            "reasons": [] if passed else [f"relative L2 {rel:.5f} exceeds {criterion['relative_l2_max']:.5f}"]}
