#!/usr/bin/env python3
"""MRI Modality — Phase B Experiment Script.

Runs both W1 (prompt-driven simulate+reconstruct) and W2 (operator correction),
captures node-by-node trace, saves RunBundle, computes metrics, and prints
all numbers needed for the report.

MRI uses COMPLEX data: y = Mask * FFT(x) + complex_noise.
Reconstruction: zero-filled (iFFT) or CS-MRI with wavelet sparsity.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_mri_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionResult
from pwm_core.graph.primitives import MRIKspace, CoilSensor, SpinSource
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.mri_solvers import cs_mri_wavelet, zero_filled_reconstruction

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "mri"
TEMPLATE_KEY = "mri_graph_v2"
H, W = 64, 64
SAMPLING_RATE = 0.25
NOISE_SIGMA = 0.005
CS_ITERS = 50
CS_LAM = 0.005

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml",
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) / (np.max(x_true) - np.min(x_true) + 1e-12))


def compute_nll_complex_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    """Complex Gaussian NLL: sum(|y - y_hat|^2 / sigma^2)."""
    r = (y.ravel() - y_hat.ravel()).astype(np.complex128)
    return float(np.sum(np.abs(r) ** 2 / sigma ** 2).real)


def make_mri_phantom(H: int, W: int, seed: int) -> np.ndarray:
    """Create a brain-like MRI phantom (real-valued, 2D).

    Uses smooth Gaussian blobs simulating brain tissue contrast.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')

    # Outer skull ellipse
    mask = ((xx / 0.85) ** 2 + (yy / 0.95) ** 2) <= 1.0
    x[mask] = 0.3

    # Brain matter with varying intensity
    structures = [
        (0.0, 0.0, 0.6, 0.7, 0.7),     # white matter
        (0.15, 0.1, 0.25, 0.3, 0.9),    # gray matter region 1
        (-0.15, -0.1, 0.2, 0.25, 0.85), # gray matter region 2
        (0.0, 0.25, 0.15, 0.12, 0.5),   # CSF-like region
        (0.0, -0.3, 0.18, 0.15, 0.45),  # ventricle-like
        (0.3, 0.0, 0.08, 0.1, 0.95),    # bright lesion
    ]
    for cx, cy, rx, ry, val in structures:
        region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        x[region] = val

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


def build_mri_operators(H: int, W: int, sampling_rate: float, seed: int):
    """Build MRI forward/adjoint callables.

    Returns (forward_fn, adjoint_fn, kspace_op, mask_array).
    """
    source = SpinSource(params={"strength": 1.0})
    kspace = MRIKspace(params={"H": H, "W": W, "sampling_rate": sampling_rate, "seed": seed})
    sensor = CoilSensor(params={"sensitivity": 1.0})

    def forward_fn(x):
        out = source.forward(x)
        out = kspace.forward(out)
        out = sensor.forward(out)
        return out

    def adjoint_fn(y):
        out = sensor.adjoint(y)
        out = kspace.adjoint(out)
        out = source.adjoint(out)
        return out

    return forward_fn, adjoint_fn, kspace, kspace._mask.copy()


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def main():
    results = {}
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("MRI Phase B Experiment")
    print("=" * 70)

    # ── 1. Compile graph ──────────────────────────────────────────────────
    print("\n[1] Compiling mri_graph_v2 ...")
    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"    Nodes: {[nid for nid, _ in graph.forward_plan]}")
    print(f"    x_shape: {graph.x_shape}, y_shape: {graph.y_shape}")

    # ── 2. Create phantom ─────────────────────────────────────────────────
    x_true = make_mri_phantom(H, W, SEED)
    print(f"\n[2] Phantom: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    # ── 3. Create RunBundle ───────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "mri_exp")
    rb_name = os.path.basename(rb_dir)
    print(f"\n[3] RunBundle: {rb_name}")

    # ══════════════════════════════════════════════════════════════════════
    # W1: Prompt-driven simulation + reconstruction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("W1: Prompt-driven simulation + reconstruction")
    print("-" * 70)

    # ── Mode S: simulate ──
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

    # Add complex Gaussian noise
    rng = np.random.RandomState(SEED)
    noise = (rng.randn(*y_clean.shape) + 1j * rng.randn(*y_clean.shape)) * NOISE_SIGMA
    y = y_clean + noise

    trace[f"{len(trace):02d}_noise_y"] = y.copy()

    y_clean_mag = np.abs(y_clean)
    noise_mag = np.abs(noise)
    snr_db = float(10 * np.log10(np.sum(y_clean_mag ** 2) / (np.sum(noise_mag ** 2) + 1e-12)))

    print(f"    Mode S: y shape={y.shape}, dtype={y.dtype}")
    print(f"    |y| range: [{np.abs(y).min():.4f}, {np.abs(y).max():.4f}]")
    print(f"    SNR: {snr_db:.1f} dB")
    print(f"    Trace stages: {len(trace)}")
    print(f"    Simulate time: {t_sim:.3f}s")

    trace_paths = save_trace(rb_dir, trace)
    print(f"    Saved {len(trace_paths)} trace files")

    # ── Build operators ──
    fwd_fn, adj_fn, kspace_op, mask = build_mri_operators(H, W, SAMPLING_RATE, SEED)

    # ── Mode I: reconstruct with CS-MRI ──
    t0 = time.time()
    x_hat_w1 = cs_mri_wavelet(
        y, mask, lam=CS_LAM, iterations=CS_ITERS,
    )
    t_inv = time.time() - t0
    # Take magnitude for real-valued comparison
    if np.iscomplexobj(x_hat_w1):
        x_hat_w1 = np.abs(x_hat_w1)
    x_hat_w1 = x_hat_w1.astype(np.float64)
    solver_used = "cs_mri_wavelet"

    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)

    print(f"\n    Mode I: solver={solver_used}, iters={CS_ITERS}, lam={CS_LAM}, time={t_inv:.2f}s")
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
        "y_dtype": str(y.dtype),
        "x_hat_shape": list(x_hat_w1.shape),
        "n_trace_stages": len(trace),
    }

    # ══════════════════════════════════════════════════════════════════════
    # W2: Operator correction mode
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("W2: Operator correction mode")
    print("-" * 70)

    a_sha256 = compute_operator_hash(mask)
    print(f"    Mask shape: {mask.shape}, density: {mask.mean():.3f}")
    print(f"    A_sha256 (mask): {a_sha256[:16]}...")

    # ── Inject mismatch: different undersampling mask (wrong seed) ──
    perturbed_seed = 99
    fwd_pert, adj_pert, kspace_pert, mask_pert = build_mri_operators(
        H, W, SAMPLING_RATE, perturbed_seed,
    )

    rng2 = np.random.RandomState(SEED + 10)
    noise_w2 = (rng2.randn(*y.shape) + 1j * rng2.randn(*y.shape)) * NOISE_SIGMA
    y_measured = fwd_pert(x_true) + noise_w2

    print(f"    Mismatch: mask seed {SEED} → {perturbed_seed}")
    print(f"    y_measured shape={y_measured.shape}")

    # ── NLL before correction ──
    y_pred_before = fwd_fn(x_true)
    nll_before = compute_nll_complex_gaussian(y_measured, y_pred_before, sigma=NOISE_SIGMA)
    print(f"    NLL before correction: {nll_before:.1f}")

    # ── Reconstruct with uncorrected mask ──
    t0 = time.time()
    x_hat_uncorrected = cs_mri_wavelet(y_measured, mask, lam=CS_LAM, iterations=CS_ITERS)
    t_w2_uncorr = time.time() - t0
    if np.iscomplexobj(x_hat_uncorrected):
        x_hat_uncorrected = np.abs(x_hat_uncorrected)
    x_hat_uncorrected = x_hat_uncorrected.astype(np.float64)

    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=float(x_true.max()))
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    print(f"    Recon (uncorrected): PSNR={psnr_uncorrected:.2f}, SSIM={ssim_uncorrected:.4f}, time={t_w2_uncorr:.2f}s")

    # ── Fit correction: grid search over mask seeds ──
    print("    Grid search over mask seeds [30..120] ...")
    search_seeds = list(range(30, 121, 3))
    best_nll = np.inf
    best_seed = SEED
    for trial_seed in search_seeds:
        fwd_trial, _, _, _ = build_mri_operators(H, W, SAMPLING_RATE, trial_seed)
        y_pred_trial = fwd_trial(x_true)
        nll_trial = compute_nll_complex_gaussian(y_measured, y_pred_trial, sigma=NOISE_SIGMA)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_seed = trial_seed

    print(f"    Best seed: {best_seed} (NLL={best_nll:.1f})")

    fwd_corrected, adj_corrected, kspace_corr, mask_corrected = build_mri_operators(
        H, W, SAMPLING_RATE, best_seed,
    )
    a_corrected_sha256 = compute_operator_hash(mask_corrected)

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100
    print(f"    NLL after correction: {nll_after:.1f}")
    print(f"    NLL decrease: {nll_decrease_pct:.1f}%")

    # ── Reconstruct with corrected mask ──
    t0 = time.time()
    x_hat_corrected = cs_mri_wavelet(y_measured, mask_corrected, lam=CS_LAM, iterations=CS_ITERS)
    t_w2_corr = time.time() - t0
    if np.iscomplexobj(x_hat_corrected):
        x_hat_corrected = np.abs(x_hat_corrected)
    x_hat_corrected = x_hat_corrected.astype(np.float64)

    psnr_corrected = psnr_fn(x_hat_corrected, x_true, max_val=float(x_true.max()))
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
        "mismatch_description": f"K-space mask seed {SEED} → {perturbed_seed}",
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

    save_artifacts(rb_dir, x_hat_w1, np.abs(y), metrics_all, x_true=x_true)

    art_dir = os.path.join(rb_dir, "artifacts")
    save_array(os.path.join(art_dir, "x_hat_w2_uncorrected.npy"), x_hat_uncorrected)
    save_array(os.path.join(art_dir, "x_hat_w2_corrected.npy"), x_hat_corrected)
    save_array(os.path.join(art_dir, "y_measured_w2.npy"), np.abs(y_measured))
    save_array(os.path.join(art_dir, "mask_nominal.npy"), mask)
    save_array(os.path.join(art_dir, "mask_corrected.npy"), mask_corrected)

    w2_meta = {
        "a_definition": "callable",
        "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256,
        "linearity": "linear",
        "linearization_notes": "N/A (linear Fourier + masking)",
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

    # ── Trace detail table ──
    print("\n    Node-by-node trace:")
    trace_table = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        if np.iscomplexobj(arr):
            rmin, rmax = float(np.abs(arr).min()), float(np.abs(arr).max())
        else:
            rmin, rmax = float(arr.min()), float(arr.max())
        row = {
            "stage": i,
            "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "range_min": round(rmin, 4),
            "range_max": round(rmax, 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        }
        trace_table.append(row)
        print(f"    {i} | {row['node_id']:20s} | {row['output_shape']:20s} | "
              f"[{row['range_min']:.4f}, {row['range_max']:.4f}] | {row['artifact_path']}")

    results["trace"] = trace_table
    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name

    results["hashes"] = {
        "y_hash": sha256_hex16(np.abs(y)),
        "x_hat_w1_hash": sha256_hex16(x_hat_w1),
        "x_true_hash": sha256_hex16(x_true),
    }

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

    results_path = os.path.join(rb_dir, "mri_experiment_results.json")
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
