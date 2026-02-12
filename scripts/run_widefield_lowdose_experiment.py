#!/usr/bin/env python3
"""Widefield Low-Dose Microscopy — Phase B Experiment Script.

Same as widefield but with reduced photon budget (1000 vs 10000) and
higher read noise (0.05 vs 0.01). Tests denoising + deconvolution.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_widefield_lowdose_experiment.py
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig
from pwm_core.graph.primitives import get_primitive
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.richardson_lucy import richardson_lucy_2d

SEED = 42
MODALITY = "widefield_lowdose"
TEMPLATE_KEY = "widefield_lowdose_graph_v2"
H, W = 64, 64
PSF_SIGMA = 3.0
PEAK_PHOTONS = 1000.0
READ_SIGMA = 0.05
RL_ITERS = 30

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml",
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr): return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
def nrmse(x_hat, x_true): return float(np.sqrt(np.mean((x_hat - x_true)**2)) / (np.max(x_true) - np.min(x_true) + 1e-12))
def compute_nll_gaussian(y, y_hat, sigma): return float(0.5 * np.sum((y - y_hat)**2) / sigma**2)


def make_gaussian_psf(sigma, size=15):
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def make_phantom(H, W, seed):
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    x += 0.05
    x += 0.4 * np.exp(-((xx - 0.1)**2 + (yy + 0.05)**2) / (2 * 0.35**2))
    x += 0.5 * np.exp(-((xx - 0.05)**2 + (yy - 0.0)**2) / (2 * 0.12**2))
    x += 0.3 * np.exp(-((xx + 0.3)**2 / (2 * 0.05**2) + (yy - 0.2)**2 / (2 * 0.3**2)))
    x += 0.25 * np.exp(-((xx - 0.4)**2 / (2 * 0.3**2) + (yy + 0.3)**2 / (2 * 0.04**2)))
    for _ in range(8):
        cx, cy = rng.uniform(-0.7, 0.7, 2)
        brightness = rng.uniform(0.3, 0.7)
        x += brightness * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 0.03**2))
    return np.clip(x, 0, 1).astype(np.float64)


def load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def compile_graph(key):
    tpl = load_template(key)
    tpl_clean = dict(tpl)
    tpl_clean.pop("description", None)
    tpl_clean["metadata"]["x_shape"] = [H, W]
    tpl_clean["metadata"]["y_shape"] = [H, W]
    spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl_clean})
    return GraphCompiler().compile(spec)


def build_operators(sigma):
    source = get_primitive("photon_source", {"strength": 1.0})
    blur = get_primitive("conv2d", {"sigma": sigma, "mode": "reflect"})
    sensor = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
    def fwd(x): return sensor.forward(blur.forward(source.forward(x)))
    def adj(y): return source.adjoint(blur.adjoint(sensor.adjoint(y)))
    return fwd, adj


def main():
    results = {}
    psnr_fn, ssim_fn = PSNR(), SSIM()

    print("=" * 70)
    print("Widefield Low-Dose Microscopy — Phase B Experiment")
    print("=" * 70)

    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"\n[1] Nodes: {[nid for nid, _ in graph.forward_plan]}")

    x_true = make_phantom(H, W, SEED)
    print(f"[2] Phantom: range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "widefield_lowdose_exp")
    rb_name = os.path.basename(rb_dir)

    # ── W1 ──
    print("\n" + "-" * 70)
    print("W1: Prompt-driven simulation + reconstruction")
    print("-" * 70)

    t0 = time.time()
    sim_result = executor.execute(x=x_true, config=ExecutionConfig(
        mode=ExecutionMode.simulate, seed=SEED, add_noise=False, capture_trace=True,
    ))
    t_sim = time.time() - t0
    y_clean = sim_result.y.copy()
    trace = sim_result.diagnostics.get("trace", {})

    rng = np.random.RandomState(SEED)
    scaled = np.maximum(y_clean * PEAK_PHOTONS, 0.0)
    y_shot = rng.poisson(scaled).astype(np.float64) / PEAK_PHOTONS
    y = y_shot + rng.normal(0, READ_SIGMA, size=y_clean.shape)
    trace[f"{len(trace):02d}_noise_y"] = y.copy()

    noise_total = y - y_clean
    snr_db = float(10 * np.log10(np.sum(y_clean**2) / (np.sum(noise_total**2) + 1e-12)))

    print(f"    y shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}], SNR={snr_db:.1f} dB")
    save_trace(rb_dir, trace)

    t0 = time.time()
    psf = make_gaussian_psf(PSF_SIGMA)
    x_hat_w1 = richardson_lucy_2d(y, psf, iterations=RL_ITERS, clip=True).astype(np.float64)
    t_inv = time.time() - t0

    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)
    print(f"    RL recon: PSNR={w1_psnr:.2f}, SSIM={w1_ssim:.4f}, NRMSE={w1_nrmse:.4f}")

    results["w1"] = {
        "psnr": round(w1_psnr, 2), "ssim": round(w1_ssim, 4), "nrmse": round(w1_nrmse, 4),
        "solver": "richardson_lucy_2d", "snr_db": round(snr_db, 1),
        "y_shape": list(y.shape), "n_trace_stages": len(trace),
        "sim_time": round(t_sim, 3), "inv_time": round(t_inv, 2),
    }

    # ── W2 ──
    print("\n" + "-" * 70)
    print("W2: Operator correction mode")
    print("-" * 70)

    psf_hash_data = make_gaussian_psf(PSF_SIGMA)
    a_sha256 = compute_operator_hash(psf_hash_data)

    # Use gain mismatch: nominal gain 1.0, perturbed 1.3 (detector miscalibration)
    perturbed_gain = 1.3

    def build_operators_gain(sigma, gain):
        source = get_primitive("photon_source", {"strength": 1.0})
        blur = get_primitive("conv2d", {"sigma": sigma, "mode": "reflect"})
        sensor = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": gain})
        def fwd(x): return sensor.forward(blur.forward(source.forward(x)))
        return fwd

    fwd_pert = build_operators_gain(PSF_SIGMA, perturbed_gain)

    rng2 = np.random.RandomState(SEED + 10)
    y_clean_pert = fwd_pert(x_true)
    scaled_pert = np.maximum(y_clean_pert * PEAK_PHOTONS, 0.0)
    y_shot_pert = rng2.poisson(scaled_pert).astype(np.float64) / PEAK_PHOTONS
    y_measured = y_shot_pert + rng2.normal(0, READ_SIGMA, size=y_clean_pert.shape)

    noise_sigma_eff = float(np.std(y_measured - y_clean_pert))
    if noise_sigma_eff < 1e-8:
        noise_sigma_eff = READ_SIGMA

    fwd_nom = build_operators_gain(PSF_SIGMA, 1.0)
    y_pred_before = fwd_nom(x_true)
    nll_before = compute_nll_gaussian(y_measured, y_pred_before, noise_sigma_eff)
    print(f"    Mismatch: gain 1.0 → {perturbed_gain}")
    print(f"    NLL before: {nll_before:.1f}")

    psf_nom = make_gaussian_psf(PSF_SIGMA)
    x_hat_uncorrected = richardson_lucy_2d(y_measured, psf_nom, iterations=RL_ITERS, clip=True).astype(np.float64)
    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=float(x_true.max()))
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    best_nll, best_gain = np.inf, 1.0
    for trial_gain in np.arange(0.5, 1.55, 0.02):
        fwd_trial = build_operators_gain(PSF_SIGMA, trial_gain)
        y_pred = fwd_trial(x_true)
        nll_t = compute_nll_gaussian(y_measured, y_pred, noise_sigma_eff)
        if nll_t < best_nll:
            best_nll, best_gain = nll_t, trial_gain

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100
    print(f"    Best gain: {best_gain:.2f}, NLL after: {nll_after:.1f}, decrease: {nll_decrease_pct:.1f}%")

    # Reconstruct corrected: scale y by gain ratio then RL
    y_corrected = y_measured / best_gain
    psf_corrected = make_gaussian_psf(PSF_SIGMA)
    x_hat_corrected = richardson_lucy_2d(y_corrected, psf_corrected, iterations=RL_ITERS, clip=True).astype(np.float64)
    a_corrected_sha256 = compute_operator_hash(psf_corrected)

    psnr_corrected = psnr_fn(x_hat_corrected, x_true, max_val=float(x_true.max()))
    ssim_corrected = ssim_fn(x_hat_corrected, x_true)
    nrmse_corrected = nrmse(x_hat_corrected, x_true)
    psnr_delta = psnr_corrected - psnr_uncorrected
    ssim_delta = ssim_corrected - ssim_uncorrected
    print(f"    PSNR uncorr={psnr_uncorrected:.2f}, corr={psnr_corrected:.2f}, delta={psnr_delta:+.2f}")

    results["w2"] = {
        "a_definition": "callable", "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256, "a_corrected_sha256": a_corrected_sha256,
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_description": f"Detector gain 1.0 → {perturbed_gain}",
        "correction_family": "Pre", "fitted_gain": round(best_gain, 2),
        "nll_before": round(nll_before, 1), "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_uncorrected": round(psnr_uncorrected, 2), "ssim_uncorrected": round(ssim_uncorrected, 4),
        "psnr_corrected": round(psnr_corrected, 2), "ssim_corrected": round(ssim_corrected, 4),
        "nrmse_corrected": round(nrmse_corrected, 4),
        "psnr_delta": round(psnr_delta, 2), "ssim_delta": round(ssim_delta, 4),
    }

    # ── Save artifacts ──
    metrics_all = {
        "w1_psnr": w1_psnr, "w1_ssim": w1_ssim, "w1_nrmse": w1_nrmse,
        "w2_nll_before": nll_before, "w2_nll_after": nll_after,
        "w2_nll_decrease_pct": nll_decrease_pct, "w2_psnr_corrected": psnr_corrected,
    }
    save_artifacts(rb_dir, x_hat_w1, y, metrics_all, x_true=x_true)
    art_dir = os.path.join(rb_dir, "artifacts")
    save_array(os.path.join(art_dir, "x_hat_w2_uncorrected.npy"), x_hat_uncorrected)
    save_array(os.path.join(art_dir, "x_hat_w2_corrected.npy"), x_hat_corrected)

    w2_meta = {
        "a_definition": "callable", "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256, "linearity": "linear",
        "linearization_notes": "N/A (linear Gaussian convolution)",
        "mismatch_type": "synthetic_injected",
        "correction_family": "Pre", "fitted_params": {"gain": round(best_gain, 2)},
        "nll_before": round(nll_before, 1), "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_operator_meta(rb_dir, w2_meta)

    trace_table = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        trace_table.append({
            "stage": i, "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape), "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 4), "range_max": round(float(arr.max()), 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        })
    results["trace"] = trace_table
    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name

    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()[:12]
    except Exception: pass

    rng_state = np.random.RandomState(SEED)
    results["env"] = {
        "seed": SEED,
        "rng_state_hash": hashlib.sha256(str(rng_state.get_state()[1][:10]).encode()).hexdigest()[:16],
        "pwm_version": git_sha,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    try:
        import scipy; results["env"]["scipy_version"] = scipy.__version__
    except ImportError: pass

    save_json(os.path.join(rb_dir, "widefield_lowdose_experiment_results.json"), results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  W1 PSNR:         {w1_psnr:.2f} dB")
    print(f"  W1 SSIM:         {w1_ssim:.4f}")
    print(f"  W1 NRMSE:        {w1_nrmse:.4f}")
    print(f"  W2 NLL decrease: {nll_decrease_pct:.1f}%")
    print(f"  W2 PSNR gain:    {psnr_delta:+.2f} dB")
    print(f"  RunBundle:       runs/{rb_name}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
