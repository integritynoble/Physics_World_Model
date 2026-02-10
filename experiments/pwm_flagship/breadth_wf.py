"""PWM Flagship -- Widefield microscopy breadth anchor.

Demonstrates the PWM framework on widefield fluorescence microscopy:

1. Compile widefield template from graph_templates.yaml to OperatorGraph.
2. Serialize + reproduce (RunBundle + hashes).
3. Adjoint check (linear).
4. One mismatch (PSF blur sigma error) + calibration improvement.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_wf --out_dir results/flagship_wf
    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_wf --smoke
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


def _load_wf_template() -> Dict[str, Any]:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    templates = data.get("templates", {})
    if "widefield_graph_v1" not in templates:
        raise RuntimeError("widefield_graph_v1 not found")
    return templates["widefield_graph_v1"]


def run_wf_breadth(
    out_dir: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run widefield breadth anchor experiment."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4100)
    compiler = GraphCompiler()

    # -- Step 1: Compile widefield template ----------------------------------
    template = _load_wf_template()
    spec = OperatorGraphSpec(
        graph_id="widefield_graph_v1",
        nodes=template["nodes"],
        edges=template["edges"],
        metadata=template.get("metadata", {}),
    )
    graph_op = compiler.compile(spec)
    logger.info("Compiled widefield_graph_v1: %d nodes, linear=%s",
                len(graph_op.forward_plan), graph_op.all_linear)

    # -- Step 2: Serialize + hash -------------------------------------------
    serialized = graph_op.serialize()
    serial_json = json.dumps(serialized, sort_keys=True, default=str)
    serial_hash = hashlib.sha256(serial_json.encode()).hexdigest()
    logger.info("Widefield graph hash: %s", serial_hash)

    # -- Step 3: Adjoint check -----------------------------------------------
    # Widefield has blur (self-adjoint) + noise (non-linear => no full adjoint)
    # The blur primitive is linear and self-adjoint, noise is not
    # For the full graph adjoint check, it will fail because of the noise node
    # Instead we test the linear subgraph (just the blur node)
    from pwm_core.graph.primitives import Conv2d
    blur_prim = Conv2d(params={"sigma": 2.0, "mode": "reflect"})

    # Manual adjoint check for the blur primitive
    x_test = rng.standard_normal((64, 64)).astype(np.float64)
    y_test = rng.standard_normal((64, 64)).astype(np.float64)
    Ax = blur_prim.forward(x_test)
    ATy = blur_prim.adjoint(y_test)
    inner1 = float(np.sum(Ax * y_test))
    inner2 = float(np.sum(x_test * ATy))
    rel_err = abs(inner1 - inner2) / max(abs(inner1), abs(inner2), 1e-30)
    adjoint_pass = rel_err < 1e-4
    logger.info("Widefield blur adjoint check: rel_err=%.2e, pass=%s",
                rel_err, adjoint_pass)

    # Full graph adjoint check (expected: fail due to noise)
    full_adjoint = graph_op.check_adjoint(n_trials=2, rtol=1e-3, seed=42)
    logger.info("Full graph adjoint (expected fail due to noise): %s",
                full_adjoint.summary())

    # -- Step 4: PSF mismatch + calibration demonstration -------------------
    # Ground truth sample
    sample = gaussian_filter(rng.random((64, 64)).astype(np.float32), sigma=3.0)
    sample -= sample.min()
    sample /= sample.max() + 1e-8

    # True PSF blur: sigma=2.0
    true_sigma = 2.0
    blurred_true = gaussian_filter(sample, sigma=true_sigma)

    # Mismatched PSF blur: sigma=3.5 (system thinks PSF is wider)
    wrong_sigma = 3.5
    blurred_wrong = gaussian_filter(sample, sigma=wrong_sigma)

    # Deconvolution using Wiener-like filter (approximate)
    def _wiener_deconv(blurred: np.ndarray, sigma: float) -> np.ndarray:
        """Approximate deconvolution via iterative Landweber."""
        x = blurred.copy().astype(np.float64)
        for _ in range(20 if not smoke else 5):
            Ax = gaussian_filter(x, sigma=sigma)
            grad = gaussian_filter(Ax - blurred.astype(np.float64), sigma=sigma)
            x = x - 0.5 * grad
            x = np.clip(x, 0, 1)
        return x.astype(np.float32)

    # Reconstruct with wrong sigma
    x_wrong = _wiener_deconv(blurred_true, wrong_sigma)

    # Calibrated sigma: search for best sigma
    best_sigma = wrong_sigma
    best_score = float("inf")
    for trial_sigma in np.linspace(1.0, 5.0, 17):
        x_trial = _wiener_deconv(blurred_true, trial_sigma)
        # Self-consistency: re-blur and compare
        re_blurred = gaussian_filter(x_trial, sigma=trial_sigma)
        score = float(np.sum((re_blurred - blurred_true) ** 2))
        if score < best_score:
            best_score = score
            best_sigma = trial_sigma

    # Reconstruct with calibrated sigma
    x_cal = _wiener_deconv(blurred_true, best_sigma)

    # Oracle: deconv with true sigma
    x_oracle = _wiener_deconv(blurred_true, true_sigma)

    psnr_oracle = _compute_psnr(sample, x_oracle)
    psnr_wrong = _compute_psnr(sample, x_wrong)
    psnr_cal = _compute_psnr(sample, x_cal)

    logger.info("WF PSNR: oracle=%.2f, wrong=%.2f, cal=%.2f (best_sigma=%.2f)",
                psnr_oracle, psnr_wrong, psnr_cal, best_sigma)

    # -- Save ----------------------------------------------------------------
    np.save(os.path.join(out_dir, "sample.npy"), sample)
    np.save(os.path.join(out_dir, "x_cal.npy"), x_cal)

    metrics = {
        "compiled": True,
        "schema_valid": True,
        "serializable": True,
        "adjoint_blur_pass": adjoint_pass,
        "adjoint_blur_rel_err": rel_err,
        "adjoint_full_pass": full_adjoint.passed,
        "psnr_oracle_db": psnr_oracle,
        "psnr_wrong_db": psnr_wrong,
        "psnr_cal_db": psnr_cal,
        "psnr_gain_db": psnr_cal - psnr_wrong,
        "true_sigma": true_sigma,
        "wrong_sigma": wrong_sigma,
        "calibrated_sigma": best_sigma,
        "n_nodes": len(graph_op.forward_plan),
    }

    artifacts = {"sample": "sample.npy", "x_cal": "x_cal.npy"}
    hashes = {
        k: _sha256(np.load(os.path.join(out_dir, v)))
        for k, v in artifacts.items()
    }

    bundle = {
        "version": BUNDLE_VERSION,
        "spec_id": "wf_breadth_anchor",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": [4100],
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
        "modality": "widefield",
        "compiled": True,
        "adjoint_linear_pass": adjoint_pass,
        "mismatch_type": "psf_blur_sigma",
        "calibration_gain_db": psnr_cal - psnr_wrong,
        "metrics": metrics,
    }

    with open(os.path.join(out_dir, "wf_breadth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Widefield breadth anchor complete -> %s", out_dir)
    return summary


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: Widefield breadth anchor"
    )
    parser.add_argument("--out_dir", default="results/flagship_wf")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_wf_breadth(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
