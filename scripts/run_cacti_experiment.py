#!/usr/bin/env python3
"""CACTI Modality — Phase B Experiment Script.

Runs both W1 (prompt-driven simulate+reconstruct) and W2 (operator correction),
captures node-by-node trace, saves RunBundle, computes metrics, and prints
all numbers needed for the report.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_experiment.py
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
from pwm_core.graph.primitives import TemporalMask, PhotonSource, PhotonSensor
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.gap_tv import gap_tv_cacti

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "cacti"
TEMPLATE_KEY = "cacti_graph_v2"
H, W, T = 64, 64, 8  # Spatial height, width, temporal frames
NOISE_SIGMA = 0.01
GAP_TV_ITERS = 50
GAP_TV_LAM = 0.001

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml",
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) / (np.max(x_true) - np.min(x_true) + 1e-12))


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    """Gaussian NLL: 0.5 * sum((y - y_hat)^2 / sigma^2) + const."""
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def make_video_phantom(H: int, W: int, T: int, seed: int) -> np.ndarray:
    """Create a 3D video phantom (H, W, T) with smooth temporal motion.

    Uses Gaussian blobs that move across frames to create a realistic
    video test cube with spatial and temporal structure.
    """
    rng = np.random.RandomState(seed)

    n_objects = 5
    x = np.zeros((H, W, T), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')

    for i in range(n_objects):
        cy0 = rng.rand() * 0.6 + 0.2
        cx0 = rng.rand() * 0.6 + 0.2
        vy = (rng.rand() - 0.5) * 0.05  # velocity per frame
        vx = (rng.rand() - 0.5) * 0.05
        sigma_s = rng.rand() * 0.06 + 0.05
        brightness = rng.rand() * 0.5 + 0.3

        for t in range(T):
            cy = cy0 + vy * t
            cx = cx0 + vx * t
            blob = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma_s ** 2)
            x[:, :, t] += brightness * blob

    # Scale to [0.05, 0.95]
    x = x / (x.max() + 1e-12) * 0.9 + 0.05
    return x.astype(np.float64)


def load_template(key: str) -> dict:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def compile_graph(key: str):
    tpl = load_template(key)
    tpl_clean = dict(tpl)
    tpl_clean.pop("description", None)
    spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl_clean})
    compiler = GraphCompiler()
    return compiler.compile(spec)


def build_cacti_operators(seed: int, H: int, W: int, T: int):
    """Build CACTI forward/adjoint callables from individual primitives.

    Returns (forward_fn, adjoint_fn, mask_op, masks_array).
    """
    source = PhotonSource(params={"strength": 1.0})
    mask_op = TemporalMask(params={"H": H, "W": W, "T": T, "seed": seed})
    sensor = PhotonSensor(params={"quantum_efficiency": 0.9, "gain": 1.0})

    def forward_fn(x):
        out = source.forward(x)
        out = mask_op.forward(out)
        out = sensor.forward(out)
        return out

    def adjoint_fn(y):
        out = sensor.adjoint(y)
        out = mask_op.adjoint(out)
        out = source.adjoint(out)
        return out

    return forward_fn, adjoint_fn, mask_op, mask_op._masks.copy()


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def main():
    results = {}
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("CACTI Phase B Experiment")
    print("=" * 70)

    # ── 1. Compile graph ──────────────────────────────────────────────────
    print("\n[1] Compiling cacti_graph_v2 ...")
    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"    Nodes: {[nid for nid, _ in graph.forward_plan]}")
    print(f"    x_shape: {graph.x_shape}, y_shape: {graph.y_shape}")

    # ── 2. Create phantom ─────────────────────────────────────────────────
    x_true = make_video_phantom(H, W, T, SEED)
    print(f"\n[2] Phantom: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    # ── 3. Create RunBundle ───────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "cacti_exp")
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
        add_noise=False,
        capture_trace=True,
    )
    sim_result = executor.execute(x=x_true, config=sim_config)
    t_sim = time.time() - t0

    y_clean = sim_result.y.copy()
    trace = sim_result.diagnostics.get("trace", {})

    # Add realistic Gaussian noise
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

    # ── Build callable operator for reconstruction ────────────────────────
    fwd_fn, adj_fn, mask_op, masks = build_cacti_operators(SEED, H, W, T)

    # ── Mode I: reconstruct with GAP-TV (CACTI-specific) ──────────────────
    t0 = time.time()
    x_hat_w1 = gap_tv_cacti(
        y, masks,
        iterations=GAP_TV_ITERS, lam=GAP_TV_LAM,
    ).astype(np.float64)
    t_inv = time.time() - t0
    solver_used = "gap_tv_cacti"

    # Compute metrics
    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=1.0)
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)

    print(f"\n    Mode I: solver={solver_used}, iters={GAP_TV_ITERS}, lam={GAP_TV_LAM}, time={t_inv:.2f}s")
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
    print("W2: Operator correction mode (measured y + callable operator A)")
    print("-" * 70)

    # Hash the masks as the operator fingerprint
    a_sha256 = compute_operator_hash(masks)
    print(f"    Masks shape: {masks.shape}, dtype: {masks.dtype}")
    print(f"    A_sha256 (masks): {a_sha256[:16]}...")

    # ── Inject synthetic mismatch: timing jitter (different seed) ──────────
    perturbed_seed = 99
    fwd_pert, adj_pert, mask_op_pert, masks_pert = build_cacti_operators(
        perturbed_seed, H, W, T,
    )

    # Generate "measured" y using perturbed operator + noise
    rng2 = np.random.RandomState(SEED + 10)
    y_measured = fwd_pert(x_true) + rng2.randn(*y.shape) * NOISE_SIGMA

    print(f"    Mismatch: timing jitter (seed {SEED} → {perturbed_seed})")
    print(f"    y_measured shape={y_measured.shape}, range=[{y_measured.min():.4f}, {y_measured.max():.4f}]")

    # ── NLL before correction (using nominal operator) ────────────────────
    y_pred_before = fwd_fn(x_true)
    nll_before = compute_nll_gaussian(y_measured, y_pred_before, sigma=NOISE_SIGMA)
    print(f"    NLL before correction: {nll_before:.1f}")

    # ── Reconstruct with uncorrected (nominal) operator ───────────────────
    t0 = time.time()
    x_hat_uncorrected = gap_tv_cacti(
        y_measured, masks,
        iterations=GAP_TV_ITERS, lam=GAP_TV_LAM,
    ).astype(np.float64)
    t_w2_uncorr = time.time() - t0

    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=1.0)
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    print(f"    Recon (uncorrected): PSNR={psnr_uncorrected:.2f}, SSIM={ssim_uncorrected:.4f}, time={t_w2_uncorr:.2f}s")

    # ── Fit correction: grid search over mask seeds ─────────────────────
    # For CACTI, the mismatch is the mask seed (timing jitter).
    # We search over a range of seeds to find the best match.
    print("    Grid search over mask seeds [30..120] ...")
    search_seeds = list(range(30, 121, 3))  # ~30 candidates
    best_nll = np.inf
    best_seed = SEED
    for trial_seed in search_seeds:
        fwd_trial, _, _, _ = build_cacti_operators(trial_seed, H, W, T)
        y_pred_trial = fwd_trial(x_true)
        nll_trial = compute_nll_gaussian(y_measured, y_pred_trial, sigma=NOISE_SIGMA)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_seed = trial_seed

    print(f"    Best seed: {best_seed} (NLL={best_nll:.1f})")

    # Build corrected operator
    fwd_corrected, adj_corrected, mask_op_corr, masks_corrected = build_cacti_operators(
        best_seed, H, W, T,
    )
    a_corrected_sha256 = compute_operator_hash(masks_corrected)

    y_pred_after = fwd_corrected(x_true)
    nll_after = compute_nll_gaussian(y_measured, y_pred_after, sigma=NOISE_SIGMA)
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    print(f"    NLL after correction: {nll_after:.1f}")
    print(f"    NLL decrease: {nll_decrease_pct:.1f}%")

    # ── Reconstruct with corrected operator ───────────────────────────────
    t0 = time.time()
    x_hat_corrected = gap_tv_cacti(
        y_measured, masks_corrected,
        iterations=GAP_TV_ITERS, lam=GAP_TV_LAM,
    ).astype(np.float64)
    t_w2_corr = time.time() - t0

    psnr_corrected = psnr_fn(x_hat_corrected, x_true, max_val=1.0)
    ssim_corrected = ssim_fn(x_hat_corrected, x_true)
    nrmse_corrected = nrmse(x_hat_corrected, x_true)

    psnr_delta = psnr_corrected - psnr_uncorrected
    ssim_delta = ssim_corrected - ssim_uncorrected

    print(f"    Recon (corrected): PSNR={psnr_corrected:.2f}, SSIM={ssim_corrected:.4f}, time={t_w2_corr:.2f}s")
    print(f"    PSNR gain: {psnr_delta:+.2f} dB, SSIM gain: {ssim_delta:+.4f}")

    results["w2"] = {
        "a_definition": "callable",
        "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256,
        "a_corrected_sha256": a_corrected_sha256,
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"seed_nominal": SEED, "seed_perturbed": perturbed_seed},
        "mismatch_description": f"Timing jitter: mask seed {SEED} → {perturbed_seed}",
        "correction_family": "Pre",
        "fitted_seed": best_seed,
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
    save_array(os.path.join(art_dir, "y_measured_w2.npy"), y_measured)
    save_array(os.path.join(art_dir, "masks_nominal.npy"), masks)
    save_array(os.path.join(art_dir, "masks_corrected.npy"), masks_corrected)

    # Save W2 operator metadata
    w2_meta = {
        "a_definition": "callable",
        "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256,
        "linearity": "linear",
        "linearization_notes": "N/A",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"seed_nominal": SEED, "seed_perturbed": perturbed_seed},
        "correction_family": "Pre",
        "fitted_params": {"seed": best_seed},
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
        print(f"    {i} | {row['node_id']:20s} | {row['output_shape']:20s} | "
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
    results_path = os.path.join(rb_dir, "cacti_experiment_results.json")
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
    print(f"  W2 PSNR gain:      {psnr_delta:+.2f} dB")
    print(f"  W2 SSIM gain:      {ssim_delta:+.4f}")
    print(f"  W2 fitted seed:    {best_seed}")
    print(f"  RunBundle:         runs/{rb_name}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
