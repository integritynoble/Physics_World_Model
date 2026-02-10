"""PWM Flagship -- Holography breadth anchor.

Demonstrates the PWM framework on off-axis digital holography:

1. Compile holography template from graph_templates.yaml to OperatorGraph.
2. Serialize + reproduce (RunBundle + hashes).
3. Adjoint check (has non-linear magnitude_sq element).
4. One mismatch (propagation distance error) + calibration improvement.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_holo --out_dir results/flagship_holo
    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_holo --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec

logger = logging.getLogger(__name__)

PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
TEMPLATES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "packages" / "pwm_core" / "contrib" / "graph_templates.yaml"
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest()


def _compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(np.abs(x).max()), float(np.abs(y).max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def _load_holo_template() -> Dict[str, Any]:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    templates = data.get("templates", {})
    if "holography_graph_v1" not in templates:
        raise RuntimeError("holography_graph_v1 not found")
    return templates["holography_graph_v1"]


def run_holo_breadth(
    out_dir: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run holography breadth anchor experiment."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4200)
    compiler = GraphCompiler()

    # -- Step 1: Compile holography template ---------------------------------
    template = _load_holo_template()
    spec = OperatorGraphSpec(
        graph_id="holography_graph_v1",
        nodes=template["nodes"],
        edges=template["edges"],
        metadata=template.get("metadata", {}),
    )
    graph_op = compiler.compile(spec)
    logger.info("Compiled holography_graph_v1: %d nodes, linear=%s",
                len(graph_op.forward_plan), graph_op.all_linear)

    # -- Step 2: Serialize + hash -------------------------------------------
    serialized = graph_op.serialize()
    serial_json = json.dumps(serialized, sort_keys=True, default=str)
    serial_hash = hashlib.sha256(serial_json.encode()).hexdigest()
    logger.info("Holography graph hash: %s", serial_hash)

    # -- Step 3: Adjoint check -----------------------------------------------
    # Holography has magnitude_sq (non-linear), so full adjoint check fails.
    # We test the linear subgraph (fresnel_prop) separately.
    full_adjoint = graph_op.check_adjoint(n_trials=2, rtol=1e-3, seed=42)
    logger.info("Full graph adjoint (expected fail due to magnitude_sq): %s",
                full_adjoint.summary())

    # Test linear subgraph: Fresnel propagation
    from pwm_core.graph.primitives import FresnelProp
    prop = FresnelProp(params={
        "wavelength": 0.633e-6,
        "distance": 5.0e-3,
        "pixel_size": 3.45e-6,
    })
    x_test = rng.standard_normal((64, 64)).astype(np.complex128)
    y_test = rng.standard_normal((64, 64)).astype(np.complex128)
    Px = prop.forward(x_test)
    PTy = prop.adjoint(y_test)
    inner1 = float(np.real(np.sum(Px * np.conj(y_test))))
    inner2 = float(np.real(np.sum(x_test * np.conj(PTy))))
    rel_err = abs(inner1 - inner2) / max(abs(inner1), abs(inner2), 1e-30)
    prop_adjoint_pass = rel_err < 1e-3
    logger.info("Fresnel prop adjoint check: rel_err=%.2e, pass=%s",
                rel_err, prop_adjoint_pass)

    # -- Step 4: Mismatch (distance error) + calibration --------------------
    # Create a complex-valued sample (amplitude + phase)
    amplitude = gaussian_filter(rng.random((64, 64)).astype(np.float64), sigma=4.0)
    amplitude -= amplitude.min()
    amplitude /= amplitude.max() + 1e-8
    phase = gaussian_filter(rng.random((64, 64)).astype(np.float64), sigma=6.0) * np.pi
    sample = amplitude * np.exp(1j * phase)

    # True propagation distance
    true_distance = 5.0e-3
    true_prop = FresnelProp(params={
        "wavelength": 0.633e-6,
        "distance": true_distance,
        "pixel_size": 3.45e-6,
    })

    # True hologram (intensity after propagation)
    field_true = true_prop.forward(sample)
    hologram = np.abs(field_true) ** 2

    # Mismatched distance (10% error)
    wrong_distance = 5.5e-3
    wrong_prop = FresnelProp(params={
        "wavelength": 0.633e-6,
        "distance": wrong_distance,
        "pixel_size": 3.45e-6,
    })

    # Reconstruct with wrong distance
    x_wrong = wrong_prop.adjoint(hologram.astype(np.complex128))
    x_wrong_amp = np.abs(x_wrong)

    # Calibrate: search for best distance
    best_distance = wrong_distance
    best_score = float("inf")
    n_search = 5 if smoke else 21
    for d_trial in np.linspace(4.0e-3, 6.0e-3, n_search):
        trial_prop = FresnelProp(params={
            "wavelength": 0.633e-6,
            "distance": d_trial,
            "pixel_size": 3.45e-6,
        })
        x_trial = trial_prop.adjoint(hologram.astype(np.complex128))
        # Self-consistency: re-propagate and compare intensity
        h_trial = np.abs(trial_prop.forward(x_trial)) ** 2
        score = float(np.sum((h_trial - hologram) ** 2))
        if score < best_score:
            best_score = score
            best_distance = d_trial

    # Reconstruct with calibrated distance
    cal_prop = FresnelProp(params={
        "wavelength": 0.633e-6,
        "distance": best_distance,
        "pixel_size": 3.45e-6,
    })
    x_cal = cal_prop.adjoint(hologram.astype(np.complex128))
    x_cal_amp = np.abs(x_cal)

    # Oracle reconstruction
    x_oracle = true_prop.adjoint(hologram.astype(np.complex128))
    x_oracle_amp = np.abs(x_oracle)

    psnr_oracle = _compute_psnr(amplitude, x_oracle_amp)
    psnr_wrong = _compute_psnr(amplitude, x_wrong_amp)
    psnr_cal = _compute_psnr(amplitude, x_cal_amp)

    logger.info("Holo PSNR: oracle=%.2f, wrong=%.2f, cal=%.2f (d_cal=%.4fmm)",
                psnr_oracle, psnr_wrong, psnr_cal, best_distance * 1e3)

    # -- Save ----------------------------------------------------------------
    np.save(os.path.join(out_dir, "amplitude.npy"), amplitude.astype(np.float32))
    np.save(os.path.join(out_dir, "x_cal_amp.npy"), x_cal_amp.astype(np.float32))

    metrics = {
        "compiled": True,
        "schema_valid": True,
        "serializable": True,
        "adjoint_prop_pass": prop_adjoint_pass,
        "adjoint_prop_rel_err": rel_err,
        "adjoint_full_pass": full_adjoint.passed,
        "psnr_oracle_db": psnr_oracle,
        "psnr_wrong_db": psnr_wrong,
        "psnr_cal_db": psnr_cal,
        "psnr_gain_db": psnr_cal - psnr_wrong,
        "true_distance_mm": true_distance * 1e3,
        "wrong_distance_mm": wrong_distance * 1e3,
        "cal_distance_mm": best_distance * 1e3,
        "n_nodes": len(graph_op.forward_plan),
        "all_linear": graph_op.all_linear,
    }

    artifacts = {"amplitude": "amplitude.npy", "x_cal_amp": "x_cal_amp.npy"}
    hashes = {
        k: _sha256(np.load(os.path.join(out_dir, v)))
        for k, v in artifacts.items()
    }

    bundle = {
        "version": BUNDLE_VERSION,
        "spec_id": "holo_breadth_anchor",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": [4200],
            "platform": platform.platform(),
            "pwm_version": PWM_VERSION,
        },
        "metrics": metrics,
        "artifacts": artifacts,
        "hashes": hashes,
        "graph_serialized": serialized,
    }

    with open(os.path.join(out_dir, "runbundle_manifest.json"), "w") as f:
        json.dump(bundle, f, indent=2, default=str)

    summary = {
        "modality": "holography",
        "compiled": True,
        "adjoint_linear_pass": prop_adjoint_pass,
        "has_nonlinear": not graph_op.all_linear,
        "mismatch_type": "propagation_distance_error",
        "calibration_gain_db": psnr_cal - psnr_wrong,
        "metrics": metrics,
    }

    with open(os.path.join(out_dir, "holo_breadth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Holography breadth anchor complete -> %s", out_dir)
    return summary


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: Holography breadth anchor"
    )
    parser.add_argument("--out_dir", default="results/flagship_holo")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_holo_breadth(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
