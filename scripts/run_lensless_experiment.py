#!/usr/bin/env python3
"""Lensless (Diffuser Camera) Imaging — Phase B Experiment Script.

Lensless: y = Poisson(peak * QE * PSF ** x) + noise
Large PSF (sigma=5.0) simulates diffuser spread.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_lensless_experiment.py
"""
from __future__ import annotations
import hashlib, os, platform, sys, time
from datetime import datetime, timezone
import numpy as np, yaml
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
MODALITY = "lensless"
TEMPLATE_KEY = "lensless_graph_v2"
H, W = 64, 64
PSF_SIGMA = 5.0
PEAK_PHOTONS = 20000.0
READ_SIGMA = 0.01
RL_ITERS = 100

TEMPLATES_PATH = os.path.join(PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

def sha256_hex16(arr): return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
def nrmse(x_hat, x_true): return float(np.sqrt(np.mean((x_hat - x_true)**2)) / (np.max(x_true) - np.min(x_true) + 1e-12))
def compute_nll_gaussian(y, y_hat, sigma): return float(0.5 * np.sum((y - y_hat)**2) / sigma**2)

def make_gaussian_psf(sigma, size=31):
    ax = np.arange(-size//2+1, size//2+1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    return psf / psf.sum()

def make_phantom(H, W, seed):
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W), indexing='ij')
    x += 0.05
    x += 0.6 * np.exp(-((xx-0.2)**2+(yy+0.1)**2)/(2*0.25**2))
    x += 0.4 * np.exp(-((xx+0.3)**2+(yy-0.2)**2)/(2*0.2**2))
    x += 0.3 * np.exp(-((xx)**2+(yy+0.4)**2)/(2*0.15**2))
    for _ in range(4):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        x += rng.uniform(0.2, 0.4) * np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*0.1**2))
    return np.clip(x, 0, 1).astype(np.float64)

def load_template(key):
    with open(TEMPLATES_PATH) as f:
        return yaml.safe_load(f)["templates"][key]

def compile_graph(key):
    tpl = dict(load_template(key)); tpl.pop("description", None)
    tpl["metadata"]["x_shape"] = [H, W]; tpl["metadata"]["y_shape"] = [H, W]
    return GraphCompiler().compile(OperatorGraphSpec.model_validate({"graph_id": key, **tpl}))

def build_operators(sigma, gain=1.0):
    src = get_primitive("photon_source", {"strength": 1.0})
    blur = get_primitive("conv2d", {"sigma": sigma, "mode": "constant"})
    sens = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": gain})
    def fwd(x): return sens.forward(blur.forward(src.forward(x)))
    def adj(y): return src.adjoint(blur.adjoint(sens.adjoint(y)))
    return fwd, adj

def main():
    psnr_fn, ssim_fn = PSNR(), SSIM()
    print("="*70); print(f"{MODALITY} — Phase B Experiment"); print("="*70)

    graph = compile_graph(TEMPLATE_KEY)
    executor = GraphExecutor(graph)
    x_true = make_phantom(H, W, SEED)
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, f"{MODALITY}_exp")
    rb_name = os.path.basename(rb_dir)

    # W1
    t0 = time.time()
    sim_result = executor.execute(x=x_true, config=ExecutionConfig(
        mode=ExecutionMode.simulate, seed=SEED, add_noise=False, capture_trace=True))
    y_clean = sim_result.y.copy()
    trace = sim_result.diagnostics.get("trace", {})

    rng = np.random.RandomState(SEED)
    scaled = np.maximum(y_clean * PEAK_PHOTONS, 0.0)
    y = rng.poisson(scaled).astype(np.float64) / PEAK_PHOTONS + rng.normal(0, READ_SIGMA, size=y_clean.shape)
    trace[f"{len(trace):02d}_noise_y"] = y.copy()
    snr_db = float(10*np.log10(np.sum(y_clean**2)/(np.sum((y-y_clean)**2)+1e-12)))
    save_trace(rb_dir, trace)

    psf = make_gaussian_psf(PSF_SIGMA)
    x_hat_w1 = richardson_lucy_2d(y, psf, iterations=RL_ITERS, clip=True).astype(np.float64)
    t_w1 = time.time() - t0

    w1_psnr = psnr_fn(x_hat_w1, x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1, x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)
    print(f"  W1: PSNR={w1_psnr:.2f}, SSIM={w1_ssim:.4f}, NRMSE={w1_nrmse:.4f}, SNR={snr_db:.1f}")

    # W2 - gain mismatch (1.0 → 1.3)
    perturbed_gain = 1.3
    a_sha256 = compute_operator_hash(make_gaussian_psf(PSF_SIGMA))

    fwd_pert, _ = build_operators(PSF_SIGMA, gain=perturbed_gain)
    rng2 = np.random.RandomState(SEED + 10)
    y_clean_pert = fwd_pert(x_true)
    y_measured = rng2.poisson(np.maximum(y_clean_pert*PEAK_PHOTONS,0)).astype(np.float64)/PEAK_PHOTONS + rng2.normal(0,READ_SIGMA,size=y_clean_pert.shape)
    noise_sigma_eff = max(float(np.std(y_measured - y_clean_pert)), READ_SIGMA)

    fwd_nom, _ = build_operators(PSF_SIGMA, gain=1.0)
    nll_before = compute_nll_gaussian(y_measured, fwd_nom(x_true), noise_sigma_eff)

    x_hat_uncorrected = richardson_lucy_2d(y_measured, psf, iterations=RL_ITERS, clip=True).astype(np.float64)
    psnr_uncorrected = psnr_fn(x_hat_uncorrected, x_true, max_val=float(x_true.max()))
    ssim_uncorrected = ssim_fn(x_hat_uncorrected, x_true)

    best_nll, best_gain = np.inf, 1.0
    for tg in np.arange(0.5, 2.05, 0.05):
        fwd_t, _ = build_operators(PSF_SIGMA, gain=tg)
        nll_t = compute_nll_gaussian(y_measured, fwd_t(x_true), noise_sigma_eff)
        if nll_t < best_nll: best_nll, best_gain = nll_t, tg

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    y_corrected = y_measured / best_gain
    x_hat_corrected = richardson_lucy_2d(y_corrected, psf, iterations=RL_ITERS, clip=True).astype(np.float64)
    psnr_corrected = psnr_fn(x_hat_corrected, x_true, max_val=float(x_true.max()))
    ssim_corrected = ssim_fn(x_hat_corrected, x_true)
    psnr_delta = psnr_corrected - psnr_uncorrected
    ssim_delta = ssim_corrected - ssim_uncorrected

    print(f"  W2: NLL {nll_before:.1f}→{nll_after:.1f} ({nll_decrease_pct:.1f}%), PSNR gain={psnr_delta:+.2f}")

    # Save
    metrics_all = {"w1_psnr": w1_psnr, "w1_ssim": w1_ssim, "w1_nrmse": w1_nrmse,
        "w2_nll_before": nll_before, "w2_nll_after": nll_after, "w2_nll_decrease_pct": nll_decrease_pct,
        "w2_psnr_corrected": psnr_corrected}
    save_artifacts(rb_dir, x_hat_w1, y, metrics_all, x_true=x_true)
    art_dir = os.path.join(rb_dir, "artifacts")
    save_array(os.path.join(art_dir, "x_hat_w2_uncorrected.npy"), x_hat_uncorrected)
    save_array(os.path.join(art_dir, "x_hat_w2_corrected.npy"), x_hat_corrected)
    save_operator_meta(rb_dir, {"a_definition": "callable", "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256, "linearity": "linear", "linearization_notes": "N/A",
        "mismatch_type": "synthetic_injected", "correction_family": "Pre",
        "nll_before": round(nll_before,1), "nll_after": round(nll_after,1),
        "nll_decrease_pct": round(nll_decrease_pct,1),
        "timestamp": datetime.now(timezone.utc).isoformat()})

    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(["git","rev-parse","HEAD"], cwd=PROJECT_ROOT, text=True).strip()[:12]
    except: pass

    print(f"\n  SUMMARY: W1 PSNR={w1_psnr:.2f}, W2 NLL decrease={nll_decrease_pct:.1f}%, PSNR gain={psnr_delta:+.2f}")
    print(f"  RunBundle: runs/{rb_name}")
    return {"w1": {"psnr": round(w1_psnr,2), "ssim": round(w1_ssim,4), "nrmse": round(w1_nrmse,4)},
            "w2": {"nll_before": round(nll_before,1), "nll_after": round(nll_after,1),
                   "nll_decrease_pct": round(nll_decrease_pct,1), "psnr_corrected": round(psnr_corrected,2),
                   "psnr_delta": round(psnr_delta,2)},
            "rb_name": rb_name}

if __name__ == "__main__":
    main()
