#!/usr/bin/env python3
"""Phase Retrieval (CDI) — Phase B Experiment Script.

Phase retrieval: y = |Angular_Spectrum(x)|² + noise

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_phase_retrieval_experiment.py
"""
from __future__ import annotations
import hashlib, os, sys, time
from datetime import datetime, timezone
import numpy as np, yaml

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

SEED = 42
MODALITY = "phase_retrieval"
TEMPLATE_KEY = "phase_retrieval_graph_v2"
H, W = 64, 64
PEAK_PHOTONS = 10000.0
READ_SIGMA = 0.01

TEMPLATES_PATH = os.path.join(PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

def nrmse(x_hat, x_true): return float(np.sqrt(np.mean((x_hat - x_true)**2)) / (np.max(x_true) - np.min(x_true) + 1e-12))
def compute_nll_gaussian(y, y_hat, sigma): return float(0.5 * np.sum((y - y_hat)**2) / sigma**2)

def make_phantom(H, W, seed):
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W), indexing='ij')
    x += 0.05
    x += 0.5 * np.exp(-((xx-0.1)**2+(yy)**2)/(2*0.3**2))
    x += 0.4 * np.exp(-((xx+0.2)**2+(yy-0.3)**2)/(2*0.15**2))
    for _ in range(5):
        cx, cy = rng.uniform(-0.6, 0.6, 2)
        x += rng.uniform(0.15, 0.4) * np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*0.08**2))
    return np.clip(x, 0, 1).astype(np.float64)

def load_template(key):
    with open(TEMPLATES_PATH) as f:
        return yaml.safe_load(f)["templates"][key]

def compile_graph(key):
    tpl = dict(load_template(key)); tpl.pop("description", None)
    tpl["metadata"]["x_shape"] = [H, W]; tpl["metadata"]["y_shape"] = [H, W]
    return GraphCompiler().compile(OperatorGraphSpec.model_validate({"graph_id": key, **tpl}))

def build_operators(distance):
    src = get_primitive("photon_source", {"strength": 1.0})
    prop = get_primitive("angular_spectrum", {"wavelength": 0.5e-6, "distance": distance, "pixel_size": 1.0e-6})
    det = get_primitive("magnitude_sq", {})
    sens = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
    def fwd(x):
        out = sens.forward(det.forward(prop.forward(src.forward(x))))
        return np.real(out) if np.iscomplexobj(out) else out
    def recon(y):
        amp = np.sqrt(np.maximum(y / 0.9, 0))
        field = prop.adjoint(amp.astype(np.complex128))
        out = np.abs(field) if np.iscomplexobj(field) else field
        return np.clip(out, 0, None)
    return fwd, recon

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
    y_clean = np.real(sim_result.y) if np.iscomplexobj(sim_result.y) else sim_result.y.copy()
    trace = sim_result.diagnostics.get("trace", {})

    rng = np.random.RandomState(SEED)
    scaled = np.maximum(np.abs(y_clean) * PEAK_PHOTONS, 0.0)
    y = rng.poisson(scaled).astype(np.float64) / PEAK_PHOTONS + rng.normal(0, READ_SIGMA, size=y_clean.shape)
    trace[f"{len(trace):02d}_noise_y"] = y.copy()
    snr_db = float(10*np.log10(np.sum(y_clean**2)/(np.sum((y-y_clean)**2)+1e-12)))
    save_trace(rb_dir, trace)

    _, recon_fn = build_operators(1.0e-3)
    x_hat_w1 = recon_fn(y)
    if x_hat_w1.max() > 0:
        x_hat_w1 = x_hat_w1 / x_hat_w1.max() * x_true.max()
    t_w1 = time.time() - t0

    w1_psnr = psnr_fn(x_hat_w1.astype(np.float64), x_true, max_val=float(x_true.max()))
    w1_ssim = ssim_fn(x_hat_w1.astype(np.float64), x_true)
    w1_nrmse = nrmse(x_hat_w1, x_true)
    print(f"  W1: PSNR={w1_psnr:.2f}, SSIM={w1_ssim:.4f}, NRMSE={w1_nrmse:.4f}, SNR={snr_db:.1f}")

    # W2 - gain mismatch (1.0 → 1.3)
    nom_dist = 1.0e-3
    a_sha256 = hashlib.sha256(f"phase_retrieval_{nom_dist}".encode()).hexdigest()[:16]

    # Build perturbed forward with gain=1.3
    src_p = get_primitive("photon_source", {"strength": 1.0})
    prop_p = get_primitive("angular_spectrum", {"wavelength": 0.5e-6, "distance": nom_dist, "pixel_size": 1.0e-6})
    det_p = get_primitive("magnitude_sq", {})
    sens_pert = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.3})
    def fwd_pert(x):
        out = sens_pert.forward(det_p.forward(prop_p.forward(src_p.forward(x))))
        return np.real(out) if np.iscomplexobj(out) else out

    rng2 = np.random.RandomState(SEED + 10)
    y_clean_pert = fwd_pert(x_true)
    y_measured = rng2.poisson(np.maximum(np.abs(y_clean_pert)*PEAK_PHOTONS,0)).astype(np.float64)/PEAK_PHOTONS + rng2.normal(0,READ_SIGMA,size=y_clean_pert.shape)
    noise_sigma_eff = max(float(np.std(y_measured - y_clean_pert)), READ_SIGMA)

    fwd_nom, _ = build_operators(nom_dist)
    nll_before = compute_nll_gaussian(y_measured, fwd_nom(x_true), noise_sigma_eff)

    _, recon_uncorr = build_operators(nom_dist)
    x_hat_uncorrected = recon_uncorr(y_measured)
    # No normalization — preserve gain effect for PSNR comparison
    psnr_uncorrected = psnr_fn(x_hat_uncorrected.astype(np.float64), x_true, max_val=float(x_true.max()))

    best_nll, best_gain = np.inf, 1.0
    for tg in np.arange(0.5, 2.05, 0.05):
        sens_t = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": tg})
        def fwd_t(x, s=sens_t):
            out = s.forward(det_p.forward(prop_p.forward(src_p.forward(x))))
            return np.real(out) if np.iscomplexobj(out) else out
        nll_t = compute_nll_gaussian(y_measured, fwd_t(x_true), noise_sigma_eff)
        if nll_t < best_nll: best_nll, best_gain = nll_t, tg

    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    _, recon_corr = build_operators(nom_dist)
    x_hat_corrected = recon_corr(y_measured / best_gain)
    psnr_corrected = psnr_fn(x_hat_corrected.astype(np.float64), x_true, max_val=float(x_true.max()))
    psnr_delta = psnr_corrected - psnr_uncorrected

    print(f"  W2: NLL {nll_before:.1f}→{nll_after:.1f} ({nll_decrease_pct:.1f}%), PSNR gain={psnr_delta:+.2f}")

    metrics_all = {"w1_psnr": w1_psnr, "w1_ssim": w1_ssim, "w1_nrmse": w1_nrmse,
        "w2_nll_before": nll_before, "w2_nll_after": nll_after, "w2_nll_decrease_pct": nll_decrease_pct,
        "w2_psnr_corrected": psnr_corrected}
    save_artifacts(rb_dir, x_hat_w1, y, metrics_all, x_true=x_true)
    save_operator_meta(rb_dir, {"a_definition": "callable", "a_extraction_method": "graph_stripped",
        "a_sha256": a_sha256, "linearity": "nonlinear", "linearization_notes": "magnitude_sq is nonlinear",
        "mismatch_type": "synthetic_injected", "correction_family": "Pre",
        "nll_before": round(nll_before,1), "nll_after": round(nll_after,1),
        "nll_decrease_pct": round(nll_decrease_pct,1),
        "timestamp": datetime.now(timezone.utc).isoformat()})

    print(f"\n  SUMMARY: W1 PSNR={w1_psnr:.2f}, W2 NLL decrease={nll_decrease_pct:.1f}%, PSNR gain={psnr_delta:+.2f}")
    print(f"  RunBundle: runs/{rb_name}")
    return {"w1": {"psnr": round(w1_psnr,2), "ssim": round(w1_ssim,4), "nrmse": round(w1_nrmse,4)},
            "w2": {"nll_before": round(nll_before,1), "nll_after": round(nll_after,1),
                   "nll_decrease_pct": round(nll_decrease_pct,1), "psnr_corrected": round(psnr_corrected,2),
                   "psnr_delta": round(psnr_delta,2)},
            "rb_name": rb_name}

if __name__ == "__main__":
    main()
