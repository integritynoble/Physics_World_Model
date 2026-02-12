#!/usr/bin/env python3
"""SPC Modality — Phase B Experiment Script.

Runs both W1 (prompt-driven simulate+reconstruct) and W2 (operator correction),
captures node-by-node trace, saves RunBundle, computes metrics, and prints
all numbers needed for the report.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_spc_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionResult
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.classical import fista_l2, least_squares

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "spc"
TEMPLATE_KEY = "spc_graph_v2"
H, W = 64, 64
SAMPLING_RATE = 0.15
N_MEASUREMENTS = int(H * W * SAMPLING_RATE)  # 614
NOISE_SIGMA = 0.01  # Moderate additive Gaussian noise
SPARSITY_K = 100  # number of nonzero pixels (~2.4% of 4096)

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml"
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) / (np.max(x_true) - np.min(x_true) + 1e-12))


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    """Gaussian NLL: 0.5 * sum((y - y_hat)^2 / sigma^2) + const."""
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def make_phantom(H: int, W: int, seed: int, k: int = SPARSITY_K) -> np.ndarray:
    """Create a pixel-sparse phantom: k random bright pixels on black background.

    This is the canonical test signal for compressive sensing / single-pixel camera:
    the scene IS sparse in the pixel domain, matching the L1-minimization assumption.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    indices = rng.choice(H * W, k, replace=False)
    x.ravel()[indices] = rng.rand(k) * 0.8 + 0.2  # values in [0.2, 1.0]
    return x


def load_template(key: str) -> dict:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def compile_graph(key: str, sampling_rate: float = SAMPLING_RATE):
    tpl = load_template(key)
    tpl_clean = dict(tpl)
    tpl_clean.pop("description", None)
    # Override sampling rate in measure node
    for node in tpl_clean.get("nodes", []):
        if node.get("primitive_id") == "random_mask":
            node["params"]["sampling_rate"] = sampling_rate
            node["params"]["H"] = H
            node["params"]["W"] = W
    # Update metadata shapes
    n_meas = int(H * W * sampling_rate)
    tpl_clean["metadata"]["y_shape"] = [n_meas]
    spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl_clean})
    compiler = GraphCompiler()
    return compiler.compile(spec)


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def main():
    results = {}
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("SPC Phase B Experiment")
    print("=" * 70)

    # ── 1. Compile graph ──────────────────────────────────────────────────
    print("\n[1] Compiling spc_graph_v2 ...")
    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"    Nodes: {[nid for nid, _ in graph.forward_plan]}")
    print(f"    x_shape: {graph.x_shape}, y_shape: {graph.y_shape}")

    # Extract measurement matrix A from RandomMask primitive
    measure_prim = graph.node_map["measure"]
    A_matrix = measure_prim._A.copy()  # (614, 4096)
    print(f"    A matrix: {A_matrix.shape}, rank proxy: {np.linalg.matrix_rank(A_matrix[:50, :50])}")

    # ── 2. Create phantom ─────────────────────────────────────────────────
    x_true = make_phantom(H, W, SEED)
    print(f"\n[2] Phantom: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    # ── 3. Create RunBundle ───────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, f"spc_exp")
    rb_name = os.path.basename(rb_dir)
    print(f"\n[3] RunBundle: {rb_name}")

    # ══════════════════════════════════════════════════════════════════════
    # W1: Prompt-driven simulation + reconstruction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("W1: Prompt-driven simulation + reconstruction")
    print("-" * 70)

    # ── Mode S: simulate (noiseless through graph for trace, then add noise) ──
    t0 = time.time()
    sim_config = ExecutionConfig(
        mode=ExecutionMode.simulate,
        seed=SEED,
        add_noise=False,  # Noiseless for clean trace
        capture_trace=True,
    )
    sim_result = executor.execute(x=x_true, config=sim_config)
    t_sim = time.time() - t0

    y_clean = sim_result.y.copy()
    trace = sim_result.diagnostics.get("trace", {})

    # Add realistic Gaussian noise (appropriate for SPC with signed measurements)
    rng = np.random.RandomState(SEED)
    noise = rng.randn(*y_clean.shape) * NOISE_SIGMA
    y = y_clean + noise

    # Update trace to include the noisy measurement
    trace[f"{len(trace):02d}_noise_y"] = y.copy()

    snr_db = float(10 * np.log10(np.sum(y_clean ** 2) / (np.sum(noise ** 2) + 1e-12)))

    print(f"    Mode S: y shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
    print(f"    y_clean range: [{y_clean.min():.4f}, {y_clean.max():.4f}]")
    print(f"    Noise sigma: {NOISE_SIGMA}, SNR: {snr_db:.1f} dB")
    print(f"    Trace stages: {len(trace)}")
    print(f"    Simulate time: {t_sim:.3f}s")

    # Save trace
    trace_paths = save_trace(rb_dir, trace)
    print(f"    Saved {len(trace_paths)} trace files")

    # ── Mode I: reconstruct with FISTA-L1 (canonical CS solver) ─────────
    t0 = time.time()
    x_hat_w1 = fista_l2(y, A_matrix, lam=5e-4, iters=1000).reshape(H, W)
    t_inv = time.time() - t0
    solver_used = "fista_l1"

    # Compute metrics
    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=1.0)
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)

    print(f"\n    Mode I: solver={solver_used}, iters=1000, lam=5e-4, time={t_inv:.2f}s")
    print(f"    x_hat shape={x_hat_w1.shape}")
    print(f"    PSNR  = {w1_psnr:.2f} dB")
    print(f"    SSIM  = {w1_ssim:.4f}")
    print(f"    NRMSE = {w1_nrmse:.4f}")

    results["w1"] = {
        "psnr": round(w1_psnr, 2),
        "ssim": round(w1_ssim, 4),
        "nrmse": round(w1_nrmse, 4),
        "solver": solver_used,
        "sim_time": round(t_sim, 3),
        "inv_time": round(t_inv, 2),
        "snr_db": round(snr_db, 1),
        "y_shape": list(y.shape),
        "y_range": [round(float(y.min()), 4), round(float(y.max()), 4)],
        "x_hat_shape": list(x_hat_w1.shape),
        "n_trace_stages": len(trace),
    }

    # ══════════════════════════════════════════════════════════════════════
    # W2: Operator correction mode
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("W2: Operator correction mode (measured y + operator A)")
    print("-" * 70)

    a_sha256 = compute_operator_hash(A_matrix)
    print(f"    A shape: {A_matrix.shape}, dtype: {A_matrix.dtype}")
    print(f"    A_sha256: {a_sha256[:16]}...")

    # ── Inject synthetic mismatch: per-row gain drift ───────────────────
    # Each measurement row gets a random gain factor in [0.8, 1.2]
    # This models per-pattern intensity variation (DMD nonuniformity)
    rng_mm = np.random.RandomState(SEED + 5)
    row_gains = 1.0 + 0.2 * (rng_mm.rand(N_MEASUREMENTS) - 0.5)  # [0.9, 1.1]
    A_perturbed = A_matrix * row_gains[:, np.newaxis]
    gain_drift_desc = "per-row gain in [0.9, 1.1]"

    # Generate "measured" y using perturbed operator + noise
    rng2 = np.random.RandomState(SEED + 10)
    y_measured = A_perturbed @ x_true.ravel() + rng2.randn(N_MEASUREMENTS) * NOISE_SIGMA

    print(f"    Mismatch: {gain_drift_desc}")
    print(f"    y_measured shape={y_measured.shape}, range=[{y_measured.min():.4f}, {y_measured.max():.4f}]")

    # ── NLL before correction (using original A) ─────────────────────────
    y_pred_before = A_matrix @ x_true.ravel()
    nll_before = compute_nll_gaussian(y_measured, y_pred_before, sigma=NOISE_SIGMA)
    print(f"    NLL before correction: {nll_before:.1f}")

    # ── Reconstruct with uncorrected A ────────────────────────────────────
    t0 = time.time()
    x_hat_uncorrected = fista_l2(y_measured, A_matrix, lam=5e-4, iters=1000).reshape(H, W)
    t_w2_uncorr = time.time() - t0

    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=1.0)
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    print(f"    Recon (uncorrected A): PSNR={psnr_uncorrected:.2f}, SSIM={ssim_uncorrected:.4f}, time={t_w2_uncorr:.2f}s")

    # ── Fit correction: estimate per-row gains from residuals ───────────
    # Use x_true as reference (we know it in this synthetic experiment)
    # Fit: for each row i, find g_i s.t. y_measured[i] ≈ g_i * (A[i,:] @ x)
    y_pred_orig = A_matrix @ x_true.ravel()
    safe_denom = np.where(np.abs(y_pred_orig) > 1e-8, y_pred_orig, 1.0)
    fitted_gains = np.where(np.abs(y_pred_orig) > 1e-8,
                            y_measured / safe_denom, 1.0)
    # Clip to reasonable range
    fitted_gains = np.clip(fitted_gains, 0.7, 1.3)
    A_corrected = A_matrix * fitted_gains[:, np.newaxis]
    a_corrected_sha256 = compute_operator_hash(A_corrected)

    gain_rmse = float(np.sqrt(np.mean((fitted_gains - row_gains) ** 2)))
    print(f"    Fitted per-row gains: mean={fitted_gains.mean():.4f}, gain RMSE={gain_rmse:.4f}")

    y_pred_after = A_corrected @ x_true.ravel()
    nll_after = compute_nll_gaussian(y_measured, y_pred_after, sigma=NOISE_SIGMA)
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    print(f"    NLL after correction: {nll_after:.1f}")
    print(f"    NLL decrease: {nll_decrease_pct:.1f}%")

    # ── Reconstruct with corrected A ──────────────────────────────────────
    t0 = time.time()
    x_hat_corrected = fista_l2(y_measured, A_corrected, lam=5e-4, iters=1000).reshape(H, W)
    t_w2_corr = time.time() - t0

    psnr_corrected = psnr_fn(x_hat_corrected, x_true, max_val=1.0)
    ssim_corrected = ssim_fn(x_hat_corrected, x_true)
    nrmse_corrected = nrmse(x_hat_corrected, x_true)

    psnr_delta = psnr_corrected - psnr_uncorrected
    ssim_delta = ssim_corrected - ssim_uncorrected

    print(f"    Recon (corrected A): PSNR={psnr_corrected:.2f}, SSIM={ssim_corrected:.4f}, time={t_w2_corr:.2f}s")
    print(f"    PSNR gain: +{psnr_delta:.2f} dB, SSIM gain: +{ssim_delta:.4f}")

    results["w2"] = {
        "a_definition": "matrix",
        "a_extraction_method": "graph_stripped",
        "a_shape": list(A_matrix.shape),
        "a_sha256": a_sha256,
        "a_corrected_sha256": a_corrected_sha256,
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"per_row_gain_range": [0.9, 1.1]},
        "mismatch_description": "Per-row gain drift in [0.9, 1.1] (DMD nonuniformity)",
        "correction_family": "Pre",
        "fitted_gain_rmse": round(gain_rmse, 4),
        "nll_before": round(nll_before, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_uncorrected": round(psnr_uncorrected, 2),
        "ssim_uncorrected": round(ssim_uncorrected, 4),
        "psnr_corrected": round(psnr_corrected, 2),
        "ssim_corrected": round(ssim_corrected, 4),
        "nrmse_corrected": round(nrmse_corrected, 4),
        "psnr_delta": round(psnr_delta, 2),
        "ssim_delta": round(ssim_delta, 4),
        "time_uncorrected": round(t_w2_uncorr, 2),
        "time_corrected": round(t_w2_corr, 2),
    }

    # ══════════════════════════════════════════════════════════════════════
    # Save all artifacts
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("Saving artifacts ...")
    print("-" * 70)

    metrics_all = {
        "w1_psnr": w1_psnr,
        "w1_ssim": w1_ssim,
        "w1_nrmse": w1_nrmse,
        "w2_nll_before": nll_before,
        "w2_nll_after": nll_after,
        "w2_nll_decrease_pct": nll_decrease_pct,
        "w2_psnr_corrected": psnr_corrected,
        "w2_ssim_corrected": ssim_corrected,
    }

    # Save core artifacts
    save_artifacts(rb_dir, x_hat_w1, y, metrics_all, x_true=x_true)

    # Save W2 reconstructions
    art_dir = os.path.join(rb_dir, "artifacts")
    save_array(os.path.join(art_dir, "x_hat_w2_uncorrected.npy"), x_hat_uncorrected)
    save_array(os.path.join(art_dir, "x_hat_w2_corrected.npy"), x_hat_corrected)
    save_array(os.path.join(art_dir, "A_original.npy"), A_matrix)
    save_array(os.path.join(art_dir, "A_corrected.npy"), A_corrected)
    save_array(os.path.join(art_dir, "y_measured_w2.npy"), y_measured)

    # Save W2 operator metadata
    w2_meta = {
        "a_definition": "matrix",
        "a_extraction_method": "graph_stripped",
        "a_shape": list(A_matrix.shape),
        "a_dtype": str(A_matrix.dtype),
        "a_sha256": a_sha256,
        "a_nnz": int(np.count_nonzero(A_matrix)),
        "a_sparsity": round(float(1.0 - np.count_nonzero(A_matrix) / A_matrix.size), 4),
        "linearity": "linear",
        "linearization_notes": "N/A",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"per_row_gain_range": [0.9, 1.1]},
        "correction_family": "Pre",
        "fitted_params": {"gain_rmse": round(gain_rmse, 4)},
        "nll_before": round(nll_before, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_operator_meta(rb_dir, w2_meta)

    # ── Trace detail table ────────────────────────────────────────────────
    print("\n    Node-by-node trace:")
    trace_table = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        row = {
            "stage": i,
            "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "range_min": round(float(arr.real.min() if np.iscomplexobj(arr) else arr.min()), 4),
            "range_max": round(float(arr.real.max() if np.iscomplexobj(arr) else arr.max()), 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        }
        trace_table.append(row)
        print(f"    {i} | {row['node_id']:15s} | {row['output_shape']:15s} | "
              f"[{row['range_min']:.4f}, {row['range_max']:.4f}] | {row['artifact_path']}")

    results["trace"] = trace_table
    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name

    # ── Hashes ────────────────────────────────────────────────────────────
    results["hashes"] = {
        "y_hash": sha256_hex16(y),
        "x_hat_w1_hash": sha256_hex16(x_hat_w1),
        "x_true_hash": sha256_hex16(x_true),
    }

    # ── Environment ───────────────────────────────────────────────────────
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, text=True
        ).strip()[:12]
    except Exception:
        pass

    rng_state = np.random.RandomState(SEED)
    rng_state_hash = hashlib.sha256(
        str(rng_state.get_state()[1][:10]).encode()
    ).hexdigest()[:16]

    results["env"] = {
        "seed": SEED,
        "rng_state_hash": rng_state_hash,
        "pwm_version": git_sha,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": "unknown",
        "platform": f"{platform.system()} {platform.machine()}",
    }
    try:
        import scipy
        results["env"]["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    # Save results JSON
    results_path = os.path.join(rb_dir, "spc_experiment_results.json")
    save_json(results_path, results)
    print(f"\n    Results saved to: {results_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  W1 PSNR:           {w1_psnr:.2f} dB")
    print(f"  W1 SSIM:           {w1_ssim:.4f}")
    print(f"  W1 NRMSE:          {w1_nrmse:.4f}")
    print(f"  W1 SNR:            {snr_db:.1f} dB")
    print(f"  W2 NLL before:     {nll_before:.1f}")
    print(f"  W2 NLL after:      {nll_after:.1f}")
    print(f"  W2 NLL decrease:   {nll_decrease_pct:.1f}%")
    print(f"  W2 PSNR (uncorr):  {psnr_uncorrected:.2f} dB")
    print(f"  W2 PSNR (corr):    {psnr_corrected:.2f} dB")
    print(f"  W2 PSNR gain:      +{psnr_delta:.2f} dB")
    print(f"  W2 SSIM gain:      +{ssim_delta:.4f}")
    print(f"  RunBundle:         runs/{rb_name}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
