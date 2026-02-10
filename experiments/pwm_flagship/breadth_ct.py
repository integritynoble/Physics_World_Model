"""PWM Flagship -- CT breadth anchor.

Demonstrates the PWM framework on CT (Computed Tomography):

1. Compile CT template from graph_templates.yaml to OperatorGraph.
2. Serialize + reproduce (RunBundle + hashes).
3. Pass adjoint check (CT is linear).
4. One mismatch (angular error) + one calibration improvement.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_ct --out_dir results/flagship_ct
    PYTHONPATH=. python -m experiments.pwm_flagship.breadth_ct --smoke
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
from typing import Any, Dict, List

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


def _load_ct_template() -> Dict[str, Any]:
    """Load the ct_graph_v1 template from YAML."""
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    templates = data.get("templates", {})
    if "ct_graph_v1" not in templates:
        raise RuntimeError("ct_graph_v1 template not found in graph_templates.yaml")
    return templates["ct_graph_v1"]


def run_ct_breadth(
    out_dir: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run CT breadth anchor experiment."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4000)
    compiler = GraphCompiler()

    # -- Step 1: Compile CT template -----------------------------------------
    template = _load_ct_template()
    spec = OperatorGraphSpec(
        graph_id="ct_graph_v1",
        nodes=template["nodes"],
        edges=template["edges"],
        metadata=template.get("metadata", {}),
    )
    graph_op = compiler.compile(spec)
    logger.info("Compiled ct_graph_v1: %d nodes, linear=%s",
                len(graph_op.forward_plan), graph_op.all_linear)

    # -- Step 2: Serialize + hash for reproducibility ------------------------
    serialized = graph_op.serialize()
    serial_json = json.dumps(serialized, sort_keys=True, default=str)
    serial_hash = hashlib.sha256(serial_json.encode()).hexdigest()
    logger.info("CT graph hash: %s", serial_hash)

    # -- Step 3: Adjoint check -----------------------------------------------
    # Full graph includes noise (non-linear), so test full graph and
    # also test the Radon primitive (linear) separately.
    full_adjoint = graph_op.check_adjoint(n_trials=2, rtol=1e-3, seed=42)
    logger.info("Full graph adjoint (expected fail due to noise): %s",
                full_adjoint.summary())

    # Test the Radon primitive alone (linear, exact adjoint for small sizes)
    from pwm_core.graph.primitives import CTRadon as CTRadonCheck
    radon_check = CTRadonCheck(params={"n_angles": 180, "H": 32, "W": 32})
    x_t = rng.standard_normal((32, 32)).astype(np.float64)
    y_t = rng.standard_normal((180, 32)).astype(np.float64)
    Ax = radon_check.forward(x_t)
    ATy = radon_check.adjoint(y_t)
    inner1 = float(np.sum(Ax * y_t))
    inner2 = float(np.sum(x_t * ATy))
    radon_rel_err = abs(inner1 - inner2) / max(abs(inner1), abs(inner2), 1e-30)
    radon_adjoint_pass = radon_rel_err < 1e-3
    logger.info("Radon adjoint check: rel_err=%.2e, pass=%s",
                radon_rel_err, radon_adjoint_pass)
    adjoint_report = full_adjoint

    # -- Step 4: Mismatch + calibration demonstration -----------------------
    # Generate synthetic phantom (small for speed)
    IMG_SZ = 32  # Small enough for explicit matrix (N=1024)
    phantom = gaussian_filter(
        rng.random((IMG_SZ, IMG_SZ)).astype(np.float64), sigma=3.0
    )
    phantom -= phantom.min()
    phantom /= phantom.max() + 1e-8

    from pwm_core.graph.primitives import CTRadon

    # True operator: 180 angles
    true_radon = CTRadon(params={"n_angles": 180, "H": IMG_SZ, "W": IMG_SZ})
    sino_true = true_radon.forward(phantom)

    # Mismatched operator: simulate angular error by adding a systematic
    # shift to the sinogram (center-of-rotation offset)
    cor_shift = 3  # pixels shift simulating misaligned COR
    sino_mm = np.roll(sino_true, cor_shift, axis=1)

    # Reconstruction via Landweber iteration: x += step * AT(y - Ax)
    def _iterative_recon(radon_op, sino, n_iter=50):
        # Estimate step size from operator norm
        x_probe = rng.standard_normal((IMG_SZ, IMG_SZ))
        Ax = radon_op.forward(x_probe)
        ATAx = radon_op.adjoint(Ax)
        op_norm = float(np.sqrt(np.sum(ATAx ** 2) / (np.sum(x_probe ** 2) + 1e-10)))
        step = 1.0 / max(op_norm, 1e-6)

        x = radon_op.adjoint(sino) * step  # Warm start
        for _ in range(n_iter):
            residual = sino - radon_op.forward(x)
            update = radon_op.adjoint(residual)
            x = x + step * update
            x = np.clip(x, 0, None)
        return x

    n_recon = 20 if smoke else 80

    # Oracle reconstruction (true sinogram, true operator)
    x_bp_true = _iterative_recon(true_radon, sino_true, n_iter=n_recon)

    # Wrong reconstruction (mismatched sinogram, true operator)
    x_bp_wrong = _iterative_recon(true_radon, sino_mm, n_iter=n_recon)

    # Calibration: search for COR shift
    best_shift = 0
    best_res = float("inf")
    for trial_shift in range(-5, 6):
        sino_trial = np.roll(sino_mm, -trial_shift, axis=1)
        x_trial = _iterative_recon(true_radon, sino_trial, n_iter=max(n_recon // 3, 3))
        sino_check = true_radon.forward(x_trial)
        res = float(np.sum((sino_trial - sino_check) ** 2))
        if res < best_res:
            best_res = res
            best_shift = trial_shift

    sino_cal = np.roll(sino_mm, -best_shift, axis=1)
    x_bp_cal = _iterative_recon(true_radon, sino_cal, n_iter=n_recon)

    psnr_true = _compute_psnr(phantom, x_bp_true)
    psnr_wrong = _compute_psnr(phantom, x_bp_wrong)
    psnr_cal = _compute_psnr(phantom, x_bp_cal)

    logger.info("CT PSNR: true=%.2f, wrong=%.2f, cal=%.2f",
                psnr_true, psnr_wrong, psnr_cal)

    # -- Save results --------------------------------------------------------
    np.save(os.path.join(out_dir, "phantom.npy"), phantom)
    np.save(os.path.join(out_dir, "x_bp_cal.npy"), x_bp_cal)

    metrics = {
        "compiled": True,
        "schema_valid": True,
        "serializable": True,
        "adjoint_radon_pass": radon_adjoint_pass,
        "adjoint_radon_rel_err": radon_rel_err,
        "adjoint_full_pass": full_adjoint.passed,
        "psnr_true_db": psnr_true,
        "psnr_wrong_db": psnr_wrong,
        "psnr_cal_db": psnr_cal,
        "psnr_gain_db": psnr_cal - psnr_wrong,
        "cor_shift_true": cor_shift,
        "cor_shift_est": best_shift,
        "n_nodes": len(graph_op.forward_plan),
        "all_linear": graph_op.all_linear,
    }

    artifacts = {"phantom": "phantom.npy", "x_bp_cal": "x_bp_cal.npy"}
    hashes = {
        k: _sha256(np.load(os.path.join(out_dir, v)))
        for k, v in artifacts.items()
    }

    bundle = {
        "version": BUNDLE_VERSION,
        "spec_id": "ct_breadth_anchor",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": [4000],
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
        "modality": "ct",
        "compiled": True,
        "adjoint_linear_pass": radon_adjoint_pass,
        "mismatch_type": "center_of_rotation_shift",
        "calibration_gain_db": psnr_cal - psnr_wrong,
        "metrics": metrics,
    }

    with open(os.path.join(out_dir, "ct_breadth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("CT breadth anchor complete -> %s", out_dir)
    return summary


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: CT breadth anchor"
    )
    parser.add_argument("--out_dir", default="results/flagship_ct")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ct_breadth(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
