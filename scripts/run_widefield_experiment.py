#!/usr/bin/env python3
"""Widefield Fluorescence Microscopy — Phase B Experiment Script.

Runs both W1 (prompt-driven simulate+reconstruct) and W2 (operator correction),
captures node-by-node trace, saves RunBundle, computes metrics, and prints
all numbers needed for the report.

Widefield microscopy: y = Poisson(peak * PSF ** x) + read_noise
Reconstruction: Richardson-Lucy deconvolution.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_widefield_experiment.py
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
from scipy import ndimage
from scipy.signal import fftconvolve

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionResult
from pwm_core.graph.primitives import get_primitive
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.richardson_lucy import richardson_lucy_2d

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "widefield"
TEMPLATE_KEY = "widefield_graph_v2"
H, W = 64, 64
PSF_SIGMA = 2.0
PEAK_PHOTONS = 10000.0
READ_SIGMA = 0.01
RL_ITERS = 50

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml",
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) / (np.max(x_true) - np.min(x_true) + 1e-12))


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    """Gaussian NLL: 0.5 * sum((y - y_hat)^2 / sigma^2)."""
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def make_gaussian_psf(sigma: float, size: int = 15) -> np.ndarray:
    """Create a normalized 2D Gaussian PSF."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def make_widefield_phantom(H: int, W: int, seed: int) -> np.ndarray:
    """Create a fluorescence-like phantom with smooth Gaussian blobs.

    Simulates fluorescent structures (cells, filaments) typical
    of widefield microscopy of biological samples.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')

    # Background fluorescence
    x += 0.05

    # Cell body (large Gaussian blob)
    x += 0.4 * np.exp(-((xx - 0.1)**2 + (yy + 0.05)**2) / (2 * 0.35**2))

    # Nucleus (bright, smaller)
    x += 0.5 * np.exp(-((xx - 0.05)**2 + (yy - 0.0)**2) / (2 * 0.12**2))

    # Filament structures (elongated Gaussians)
    x += 0.3 * np.exp(-((xx + 0.3)**2 / (2 * 0.05**2) + (yy - 0.2)**2 / (2 * 0.3**2)))
    x += 0.25 * np.exp(-((xx - 0.4)**2 / (2 * 0.3**2) + (yy + 0.3)**2 / (2 * 0.04**2)))

    # Small bright puncta (vesicles)
    for _ in range(8):
        cx, cy = rng.uniform(-0.7, 0.7, 2)
        brightness = rng.uniform(0.3, 0.7)
        x += brightness * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 0.03**2))

    # Clip to [0, 1]
    x = np.clip(x, 0, 1)
    return x.astype(np.float64)


def load_template(key: str) -> dict:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def compile_graph(key: str):
    tpl = load_template(key)
    tpl_clean = dict(tpl)
    tpl_clean.pop("description", None)
    # Override shapes
    for node in tpl_clean.get("nodes", []):
        if node.get("primitive_id") == "conv2d":
            node["params"]["sigma"] = PSF_SIGMA
    tpl_clean["metadata"]["x_shape"] = [H, W]
    tpl_clean["metadata"]["y_shape"] = [H, W]
    spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl_clean})
    compiler = GraphCompiler()
    return compiler.compile(spec)


def build_widefield_operators(sigma: float):
    """Build widefield forward/adjoint callables.

    Returns (forward_fn, adjoint_fn).
    """
    source = get_primitive("photon_source", {"strength": 1.0})
    blur = get_primitive("conv2d", {"sigma": sigma, "mode": "reflect"})
    sensor = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})

    def forward_fn(x):
        out = source.forward(x)
        out = blur.forward(out)
        out = sensor.forward(out)
        return out

    def adjoint_fn(y):
        out = sensor.adjoint(y)
        out = blur.adjoint(out)
        out = source.adjoint(out)
        return out

    return forward_fn, adjoint_fn


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def main():
    results = {}
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("Widefield Fluorescence Microscopy — Phase B Experiment")
    print("=" * 70)

    # ── 1. Compile graph ──────────────────────────────────────────────────
    print("\n[1] Compiling widefield_graph_v2 ...")
    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"    Nodes: {[nid for nid, _ in graph.forward_plan]}")
    print(f"    x_shape: {graph.x_shape}, y_shape: {graph.y_shape}")

    # ── 2. Create phantom ─────────────────────────────────────────────────
    x_true = make_widefield_phantom(H, W, SEED)
    print(f"\n[2] Phantom: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    # ── 3. Create RunBundle ───────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "widefield_exp")
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

    # Add Poisson-Gaussian noise (widefield noise model)
    rng = np.random.RandomState(SEED)
    scaled = np.maximum(y_clean * PEAK_PHOTONS, 0.0)
    y_shot = rng.poisson(scaled).astype(np.float64) / PEAK_PHOTONS
    y = y_shot + rng.normal(0, READ_SIGMA, size=y_clean.shape)

    trace[f"{len(trace):02d}_noise_y"] = y.copy()

    noise_total = y - y_clean
    snr_db = float(10 * np.log10(np.sum(y_clean ** 2) / (np.sum(noise_total ** 2) + 1e-12)))

    print(f"    Mode S: y shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
    print(f"    y_clean range: [{y_clean.min():.4f}, {y_clean.max():.4f}]")
    print(f"    SNR: {snr_db:.1f} dB")
    print(f"    Trace stages: {len(trace)}")
    print(f"    Simulate time: {t_sim:.3f}s")

    trace_paths = save_trace(rb_dir, trace)
    print(f"    Saved {len(trace_paths)} trace files")

    # ── Mode I: reconstruct with Richardson-Lucy ──
    t0 = time.time()
    psf = make_gaussian_psf(PSF_SIGMA)
    x_hat_w1 = richardson_lucy_2d(y, psf, iterations=RL_ITERS, clip=True)
    t_inv = time.time() - t0
    x_hat_w1 = x_hat_w1.astype(np.float64)
    solver_used = "richardson_lucy_2d"

    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)

    print(f"\n    Mode I: solver={solver_used}, iters={RL_ITERS}, time={t_inv:.2f}s")
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
        "x_hat_shape": list(x_hat_w1.shape),
        "n_trace_stages": len(trace),
    }

    # ══════════════════════════════════════════════════════════════════════
    # W2: Operator correction mode
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("W2: Operator correction mode")
    print("-" * 70)

    # Operator hash from PSF
    psf_hash_data = make_gaussian_psf(PSF_SIGMA)
    a_sha256 = compute_operator_hash(psf_hash_data)
    print(f"    PSF sigma: {PSF_SIGMA}, PSF shape: {psf.shape}")
    print(f"    A_sha256: {a_sha256[:16]}...")

    # ── Inject mismatch: PSF sigma drift (2.0 → 2.5) ──
    perturbed_sigma = 2.5
    fwd_pert, adj_pert = build_widefield_operators(perturbed_sigma)

    # Generate "measured" y with perturbed operator + noise
    rng2 = np.random.RandomState(SEED + 10)
    y_clean_pert = fwd_pert(x_true)
    scaled_pert = np.maximum(y_clean_pert * PEAK_PHOTONS, 0.0)
    y_shot_pert = rng2.poisson(scaled_pert).astype(np.float64) / PEAK_PHOTONS
    y_measured = y_shot_pert + rng2.normal(0, READ_SIGMA, size=y_clean_pert.shape)

    # Effective noise sigma for NLL (use residual-based estimate)
    noise_sigma_eff = float(np.std(y_measured - y_clean_pert))
    if noise_sigma_eff < 1e-8:
        noise_sigma_eff = READ_SIGMA

    print(f"    Mismatch: PSF sigma {PSF_SIGMA} → {perturbed_sigma}")
    print(f"    y_measured shape={y_measured.shape}, range=[{y_measured.min():.4f}, {y_measured.max():.4f}]")
    print(f"    Effective noise sigma: {noise_sigma_eff:.6f}")

    # ── NLL before correction (using nominal sigma) ──
    fwd_nom, adj_nom = build_widefield_operators(PSF_SIGMA)
    y_pred_before = fwd_nom(x_true)
    nll_before = compute_nll_gaussian(y_measured, y_pred_before, sigma=noise_sigma_eff)
    print(f"    NLL before correction: {nll_before:.1f}")

    # ── Reconstruct with uncorrected PSF ──
    t0 = time.time()
    psf_nom = make_gaussian_psf(PSF_SIGMA)
    x_hat_uncorrected = richardson_lucy_2d(y_measured, psf_nom, iterations=RL_ITERS, clip=True)
    t_w2_uncorr = time.time() - t0
    x_hat_uncorrected = x_hat_uncorrected.astype(np.float64)

    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=float(x_true.max()))
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    print(f"    Recon (uncorrected): PSNR={psnr_uncorrected:.2f}, SSIM={ssim_uncorrected:.4f}, time={t_w2_uncorr:.2f}s")

    # ── Fit correction: grid search over PSF sigma ──
    print("    Grid search over PSF sigma [1.0, 4.0] ...")
    sigma_grid = np.arange(1.0, 4.05, 0.1)
    best_nll = np.inf
    best_sigma = PSF_SIGMA
    for trial_sigma in sigma_grid:
        fwd_trial, _ = build_widefield_operators(trial_sigma)
        y_pred_trial = fwd_trial(x_true)
        nll_trial = compute_nll_gaussian(y_measured, y_pred_trial, sigma=noise_sigma_eff)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_sigma = trial_sigma

    print(f"    Best sigma: {best_sigma:.1f} (NLL={best_nll:.1f})")

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    print(f"    NLL after correction: {nll_after:.1f}")
    print(f"    NLL decrease: {nll_decrease_pct:.1f}%")

    # ── Reconstruct with corrected PSF ──
    t0 = time.time()
    psf_corrected = make_gaussian_psf(best_sigma)
    x_hat_corrected = richardson_lucy_2d(y_measured, psf_corrected, iterations=RL_ITERS, clip=True)
    t_w2_corr = time.time() - t0
    x_hat_corrected = x_hat_corrected.astype(np.float64)

    a_corrected_sha256 = compute_operator_hash(psf_corrected)

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
        "mismatch_params": {"sigma_nominal": PSF_SIGMA, "sigma_perturbed": perturbed_sigma},
        "mismatch_description": f"PSF sigma {PSF_SIGMA} → {perturbed_sigma}",
        "correction_family": "Pre",
        "fitted_sigma": round(best_sigma, 1),
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

    save_artifacts(rb_dir, x_hat_w1, y, metrics_all, x_true=x_true)

    art_dir = os.path.join(rb_dir, "artifacts")
    save_array(os.path.join(art_dir, "x_hat_w2_uncorrected.npy"), x_hat_uncorrected)
    save_array(os.path.join(art_dir, "x_hat_w2_corrected.npy"), x_hat_corrected)
    save_array(os.path.join(art_dir, "y_measured_w2.npy"), y_measured)

    w2_meta = {
        "a_definition": "callable",
        "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256,
        "linearity": "linear",
        "linearization_notes": "N/A (linear Gaussian convolution)",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"sigma_nominal": PSF_SIGMA, "sigma_perturbed": perturbed_sigma},
        "correction_family": "Pre",
        "fitted_params": {"sigma": round(best_sigma, 1)},
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
        row = {
            "stage": i,
            "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 4),
            "range_max": round(float(arr.max()), 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        }
        trace_table.append(row)
        print(f"    {i} | {row['node_id']:15s} | {row['output_shape']:15s} | "
              f"[{row['range_min']:.4f}, {row['range_max']:.4f}] | {row['artifact_path']}")

    results["trace"] = trace_table
    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name

    results["hashes"] = {
        "y_hash": sha256_hex16(y),
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

    results_path = os.path.join(rb_dir, "widefield_experiment_results.json")
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
    print(f"  W2 fitted sigma:   {best_sigma:.1f}")
    print(f"  RunBundle:         runs/{rb_name}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
