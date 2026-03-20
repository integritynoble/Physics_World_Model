#!/usr/bin/env python3
"""CT Modality — Phase B Experiment Script.

Runs both W1 (prompt-driven simulate+reconstruct) and W2 (operator correction),
captures node-by-node trace, saves RunBundle, computes metrics, and prints
all numbers needed for the report.

CT is NONLINEAR due to Beer-Lambert (I = I0 * exp(-sinogram)).
Reconstruction: invert Beer-Lambert (-log), then apply FBP or SART.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_ct_experiment.py
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
from pwm_core.graph.primitives import CTRadon, BeerLambert, PhotonSensor
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.ct_solvers import fbp_2d, sart_operator

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "ct"
TEMPLATE_KEY = "ct_graph_v2"
H, W = 64, 64
N_ANGLES = 180
I_0 = 10000.0
NOISE_SIGMA = 0.005  # Gaussian noise on log-sinogram scale
FBP_FILTER = "ramlak"
SART_ITERS = 20

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml",
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) / (np.max(x_true) - np.min(x_true) + 1e-12))


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def make_ct_phantom(H: int, W: int, seed: int) -> np.ndarray:
    """Create a Shepp-Logan-like 2D CT phantom with ellipses.

    Attenuation values in range [0, 1] representing tissue densities.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')

    # Outer ellipse (body)
    mask_body = ((xx / 0.8) ** 2 + (yy / 0.95) ** 2) <= 1.0
    x[mask_body] = 0.4

    # Inner structures (organs, bones)
    ellipses = [
        (0.2, 0.3, 0.3, 0.3, 0.7),    # large organ
        (-0.2, 0.1, 0.15, 0.2, 0.5),   # medium organ
        (0.0, -0.3, 0.25, 0.15, 0.6),  # lower structure
        (0.3, -0.1, 0.08, 0.08, 0.9),  # small dense (bone-like)
        (-0.3, -0.2, 0.1, 0.1, 0.85),  # another dense region
    ]
    for cx, cy, rx, ry, val in ellipses:
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        x[mask] = val

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


def build_ct_operators(H: int, W: int, n_angles: int, angle_offset: float = 0.0):
    """Build CT forward/adjoint callables.

    Returns (forward_fn, adjoint_fn, radon_op, beer_op, sensor_op, angles).
    The forward model is NONLINEAR: y = sensor(beer_lambert(radon(x))).
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False) + angle_offset
    radon = CTRadon(params={"n_angles": n_angles, "H": H, "W": W})
    # Override angles if offset is applied
    if angle_offset != 0.0:
        radon._angles = angles
        if radon._use_matrix:
            radon._build_matrix()
    beer = BeerLambert(params={"I_0": I_0})
    sensor = PhotonSensor(params={"quantum_efficiency": 0.9, "gain": 1.0})

    def forward_fn(x):
        sinogram = radon.forward(x)
        intensity = beer.forward(sinogram)
        return sensor.forward(intensity)

    def adjoint_fn(y):
        """Linearized adjoint at the operating point (for SART)."""
        return radon.adjoint(sensor.adjoint(y))

    return forward_fn, adjoint_fn, radon, beer, sensor, angles


def invert_beer_lambert(y_intensity: np.ndarray, I_0: float, qe: float, gain: float) -> np.ndarray:
    """Invert Beer-Lambert + sensor to recover sinogram.

    y = sensor(I_0 * exp(-sinogram)) = I_0 * qe * gain * exp(-sinogram)
    sinogram = -log(y / (I_0 * qe * gain))
    """
    scale = I_0 * qe * gain
    clipped = np.clip(y_intensity / scale, 1e-10, None)
    return -np.log(clipped)


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def main():
    results = {}
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("CT Phase B Experiment")
    print("=" * 70)

    # ── 1. Compile graph ──────────────────────────────────────────────────
    print("\n[1] Compiling ct_graph_v2 ...")
    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    print(f"    Nodes: {[nid for nid, _ in graph.forward_plan]}")
    print(f"    x_shape: {graph.x_shape}, y_shape: {graph.y_shape}")

    # ── 2. Create phantom ─────────────────────────────────────────────────
    x_true = make_ct_phantom(H, W, SEED)
    print(f"\n[2] Phantom: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")

    # ── 3. Create RunBundle ───────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "ct_exp")
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

    # Add Gaussian noise on the intensity measurements
    rng = np.random.RandomState(SEED)
    noise = rng.randn(*y_clean.shape) * NOISE_SIGMA * y_clean.mean()
    y = y_clean + noise

    trace[f"{len(trace):02d}_noise_y"] = y.copy()

    snr_db = float(10 * np.log10(np.sum(y_clean ** 2) / (np.sum(noise ** 2) + 1e-12)))

    print(f"    Mode S: y shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
    print(f"    y_clean range: [{y_clean.min():.4f}, {y_clean.max():.4f}]")
    print(f"    SNR: {snr_db:.1f} dB")
    print(f"    Trace stages: {len(trace)}")
    print(f"    Simulate time: {t_sim:.3f}s")

    trace_paths = save_trace(rb_dir, trace)
    print(f"    Saved {len(trace_paths)} trace files")

    # ── Mode I: reconstruct with FBP ──────────────────────────────────
    # First invert Beer-Lambert to get sinogram, then apply FBP
    sinogram_est = invert_beer_lambert(y, I_0, qe=0.9, gain=1.0)
    angles_rad = np.deg2rad(np.linspace(0, 180, N_ANGLES, endpoint=False))

    t0 = time.time()
    x_hat_w1 = fbp_2d(sinogram_est, angles_rad, filter_type=FBP_FILTER, output_size=H)
    t_inv = time.time() - t0
    x_hat_w1 = np.clip(x_hat_w1, 0, None).astype(np.float64)
    solver_used = "fbp_2d"

    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)

    print(f"\n    Mode I: solver={solver_used}, filter={FBP_FILTER}, time={t_inv:.2f}s")
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
    print("W2: Operator correction mode")
    print("-" * 70)

    # Build nominal operator
    fwd_nom, adj_nom, radon_nom, beer_nom, sensor_nom, angles_nom = build_ct_operators(
        H, W, N_ANGLES, angle_offset=0.0,
    )
    # Use a deterministic probe for operator hash
    probe = np.eye(H, W, dtype=np.float64)
    a_sha256 = compute_operator_hash(radon_nom.forward(probe))
    print(f"    A_sha256: {a_sha256[:16]}...")

    # ── Inject mismatch: I_0 drift (beam intensity miscalibration) ────────
    # Perturbed I_0 = 12000 (20% drift), but reconstruction assumes I_0 = 10000
    perturbed_I0 = 12000.0

    def _fwd_with_I0(x, I0_val):
        radon_op = CTRadon(params={"n_angles": N_ANGLES, "H": H, "W": W})
        beer_op = BeerLambert(params={"I_0": I0_val})
        sensor_op = PhotonSensor(params={"quantum_efficiency": 0.9, "gain": 1.0})
        return sensor_op.forward(beer_op.forward(radon_op.forward(x)))

    # Generate "measured" y with perturbed I_0 + noise
    rng2 = np.random.RandomState(SEED + 10)
    y_measured = _fwd_with_I0(x_true, perturbed_I0) + rng2.randn(*y.shape) * NOISE_SIGMA * y_clean.mean()

    print(f"    Mismatch: I_0 drift {I_0} → {perturbed_I0} (20% increase)")
    print(f"    y_measured shape={y_measured.shape}, range=[{y_measured.min():.4f}, {y_measured.max():.4f}]")

    # ── NLL before correction (using nominal I_0) ──
    y_pred_before = _fwd_with_I0(x_true, I_0)
    noise_sigma_w2 = NOISE_SIGMA * y_clean.mean()
    nll_before = compute_nll_gaussian(y_measured, y_pred_before, sigma=noise_sigma_w2)
    print(f"    NLL before correction: {nll_before:.1f}")

    # ── Reconstruct with uncorrected operator (wrong I_0) ──
    sinogram_uncorr = invert_beer_lambert(y_measured, I_0, qe=0.9, gain=1.0)
    angles_nom_rad = np.deg2rad(np.linspace(0, 180, N_ANGLES, endpoint=False))
    t0 = time.time()
    x_hat_uncorrected = fbp_2d(sinogram_uncorr, angles_nom_rad, filter_type=FBP_FILTER, output_size=H)
    t_w2_uncorr = time.time() - t0
    x_hat_uncorrected = np.clip(x_hat_uncorrected, 0, None).astype(np.float64)

    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=float(x_true.max()))
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    print(f"    Recon (uncorrected): PSNR={psnr_uncorrected:.2f}, SSIM={ssim_uncorrected:.4f}, time={t_w2_uncorr:.2f}s")

    # ── Fit correction: grid search over I_0 ──
    print("    Grid search over I_0 [5000, 20000] ...")
    search_I0 = np.linspace(5000, 20000, 31)
    best_nll = np.inf
    best_I0 = I_0
    for trial_I0 in search_I0:
        y_pred_trial = _fwd_with_I0(x_true, trial_I0)
        nll_trial = compute_nll_gaussian(y_measured, y_pred_trial, sigma=noise_sigma_w2)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_I0 = trial_I0

    print(f"    Best I_0: {best_I0:.0f} (NLL={best_nll:.1f})")

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100
    print(f"    NLL after correction: {nll_after:.1f}")
    print(f"    NLL decrease: {nll_decrease_pct:.1f}%")

    # ── Reconstruct with corrected operator (correct I_0) ──
    sinogram_corr = invert_beer_lambert(y_measured, best_I0, qe=0.9, gain=1.0)
    t0 = time.time()
    x_hat_corrected = fbp_2d(sinogram_corr, angles_nom_rad, filter_type=FBP_FILTER, output_size=H)
    t_w2_corr = time.time() - t0
    x_hat_corrected = np.clip(x_hat_corrected, 0, None).astype(np.float64)

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
        "linearity": "nonlinear",
        "linearization_notes": "Beer-Lambert exp(-x) is nonlinear; FBP operates on log-inverted sinogram",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"I0_nominal": I_0, "I0_perturbed": perturbed_I0},
        "mismatch_description": f"X-ray beam intensity drift I_0={I_0} → {perturbed_I0} (20% increase)",
        "correction_family": "Pre",
        "fitted_I0": round(best_I0, 0),
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
        "linearity": "nonlinear",
        "linearization_notes": "Beer-Lambert exp(-x) is nonlinear",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"I0_nominal": I_0, "I0_perturbed": perturbed_I0},
        "correction_family": "Pre",
        "fitted_params": {"I_0": round(best_I0, 0)},
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

    results_path = os.path.join(rb_dir, "ct_experiment_results.json")
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
    print(f"  W2 fitted I_0:     {best_I0:.0f}")
    print(f"  RunBundle:         runs/{rb_name}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
