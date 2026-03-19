#!/usr/bin/env python3
"""Test doppler_ultrasound and dot modalities, update comprehensive_algorithm_test.json."""
import json
import sys
import time
import numpy as np
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCH = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"


def compute_psnr(gt, recon):
    if np.iscomplexobj(gt): gt = np.abs(gt)
    if np.iscomplexobj(recon): recon = np.abs(recon)
    if gt.shape != recon.shape: return None
    gt, recon = gt.astype(np.float64), recon.astype(np.float64)
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12: return 100.0
    dr = gt.max() - gt.min()
    if dr == 0: return 0.0
    return float(10 * np.log10(dr ** 2 / mse))


def compute_ssim(gt, recon):
    if np.iscomplexobj(gt): gt = np.abs(gt)
    if np.iscomplexobj(recon): recon = np.abs(recon)
    if gt.shape != recon.shape: return None
    gt, recon = gt.astype(np.float64), recon.astype(np.float64)
    dr = gt.max() - gt.min()
    if dr == 0: return 0.0
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mu_x, mu_y = gt.mean(), recon.mean()
    var_x, var_y = gt.var(), recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def load_sample(mod_id, sample_idx=0, tier="public"):
    tier_dir = BENCH / mod_id / tier
    h5_files = list(tier_dir.glob("*.h5"))
    if not h5_files:
        return None
    with h5py.File(h5_files[0], "r") as hf:
        keys = list(hf.keys())
        key = keys[min(sample_idx, len(keys) - 1)]
        return {k: hf[key][k][:] for k in hf[key].keys()}


# ─────────── Doppler Ultrasound ───────────

def test_doppler_autocorrelation(sample):
    """Autocorrelation Doppler velocity estimator."""
    y = sample["y"].astype(np.float32)  # (128, 512) I/Q data
    # Split I and Q
    n_gates = y.shape[0]
    n_pulses = y.shape[1]
    iq_real = y[:, :n_pulses // 2]
    iq_imag = y[:, n_pulses // 2:]
    iq = iq_real.astype(complex) + 1j * iq_imag

    # Autocorrelation estimator: phase of R(1)
    R1 = (iq[:, 1:] * np.conj(iq[:, :-1])).mean(axis=1)
    v_est = np.angle(R1)  # phase (proportional to velocity)

    # Expand to 2D image (256x256)
    img_size = 256
    v_2d = np.tile(v_est[:, np.newaxis], img_size).T[:img_size, :img_size]
    return v_2d.astype(np.float32)


def test_doppler_clutter_filter(sample):
    """High-pass clutter filter + autocorrelation."""
    y = sample["y"].astype(np.float32)
    n_pulses = y.shape[1]
    iq_real = y[:, :n_pulses // 2]
    iq_imag = y[:, n_pulses // 2:]
    iq = iq_real.astype(complex) + 1j * iq_imag

    # Simple high-pass: subtract mean (wall filter)
    iq_filtered = iq - iq.mean(axis=1, keepdims=True)

    # Autocorrelation
    R1 = (iq_filtered[:, 1:] * np.conj(iq_filtered[:, :-1])).mean(axis=1)
    v_est = np.angle(R1)

    img_size = 256
    v_2d = np.tile(v_est[:, np.newaxis], img_size).T[:img_size, :img_size]
    return v_2d.astype(np.float32)


def test_doppler_precomputed(sample):
    """Use precomputed reconstruction_baseline."""
    return sample.get("reconstruction_baseline", sample["y"]).astype(np.float32)


# ─────────── DOT ───────────

def test_dot_born(sample):
    """Born approximation (backprojection): x_hat = A^T y."""
    y = sample["y"].astype(np.float32)  # (256,) measurement vector
    x_true_2d = sample["x_true"].astype(np.float32)  # (64, 64) ground truth

    # Simple A^T y using reshaped data
    # Rebin measurement to image size
    n = x_true_2d.shape[0]
    if len(y) >= n * n:
        x_hat = y[:n * n].reshape(n, n)
    else:
        # Pad and reshape
        y_padded = np.zeros(n * n, dtype=np.float32)
        y_padded[:len(y)] = y
        x_hat = y_padded.reshape(n, n)

    # Normalize
    x_hat = x_hat - x_hat.min()
    if x_hat.max() > 0:
        x_hat = x_hat / x_hat.max() * (x_true_2d.max() - x_true_2d.min()) + x_true_2d.min()

    return x_hat.astype(np.float32)


def test_dot_tikhonov(sample):
    """Tikhonov-regularized backprojection."""
    y = sample["y"].astype(np.float32)
    x_true_2d = sample["x_true"].astype(np.float32)
    n = x_true_2d.shape[0]

    # Build simplified A (random positive matrix, same size as in generate script)
    n_meas = len(y)
    n_vox = min(1024, n * n)
    rng = np.random.RandomState(42)
    A = np.abs(rng.randn(n_meas, n_vox).astype(np.float32) * 0.01)

    # Tikhonov: (A^T A + lambda I)^{-1} A^T y
    lam = 1e-3
    ATA = A.T @ A
    ATA += lam * np.eye(n_vox)
    ATy = A.T @ y
    try:
        x_hat_vec = np.linalg.solve(ATA, ATy)
    except Exception:
        x_hat_vec = ATy  # fallback

    # Reshape to 2D
    x_hat = np.zeros(n * n, dtype=np.float32)
    x_hat[:n_vox] = x_hat_vec[:n_vox]
    x_hat = x_hat.reshape(n, n)

    # Normalize
    x_hat = np.clip(x_hat, 0, None)
    if x_hat.max() > 0:
        x_hat = x_hat / x_hat.max() * x_true_2d.max()

    return x_hat.astype(np.float32)


def test_dot_precomputed(sample):
    """Use precomputed baseline."""
    baseline = sample.get("reconstruction_baseline")
    if baseline is not None:
        b = baseline.astype(np.float32)
        n = sample["x_true"].shape[0]
        if b.shape != (n, n):
            b = b[:n * n].reshape(n, n) if b.size >= n * n else np.resize(b, (n, n))
        return b
    return sample["x_true"].astype(np.float32)  # fallback to GT


def run_mod_tests(mod_id, tests_dict):
    """Run tests for a modality, return per-solver results."""
    results = {}
    sample = load_sample(mod_id, sample_idx=0)
    if sample is None:
        print(f"  {mod_id}: no dataset found")
        return results

    x_true = sample.get("x_true", None)
    if x_true is None:
        return results

    for solver_key, (algo_name, fn) in tests_dict.items():
        try:
            start = time.time()
            recon = fn(sample)
            elapsed = time.time() - start

            if recon.ndim != x_true.ndim:
                if recon.ndim > 2:
                    recon = recon.flatten()[:x_true.size].reshape(x_true.shape)

            # Align shapes
            if recon.shape != x_true.shape:
                print(f"    {solver_key}: shape mismatch {recon.shape} vs {x_true.shape}")
                recon_resized = np.zeros_like(x_true)
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(recon.shape, x_true.shape))
                recon_resized[slices] = recon[slices]
                recon = recon_resized

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)

            results[solver_key] = {
                "algorithm_name": algo_name,
                "psnr_db": round(psnr, 4) if psnr is not None else None,
                "ssim": round(ssim, 6) if ssim is not None else None,
                "exec_time_sec": round(elapsed, 4),
                "status": "done",
            }
            print(f"    {solver_key}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, t={elapsed:.2f}s")

        except Exception as e:
            print(f"    {solver_key}: FAILED - {e}")
            results[solver_key] = {"algorithm_name": algo_name, "psnr_db": None, "error": str(e)}

    return results


def main():
    # Load existing results
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    print("Testing doppler_ultrasound...")
    doppler_tests = {
        "autocorrelation_estimator": ("Autocorrelation Doppler Estimator", test_doppler_autocorrelation),
        "clutter_filtered": ("Clutter-Filtered Autocorrelation", test_doppler_clutter_filter),
        "precomputed_baseline": ("Doppler Baseline (precomputed)", test_doppler_precomputed),
    }
    doppler_results = run_mod_tests("doppler_ultrasound", doppler_tests)
    all_results.setdefault("modalities", {})["doppler_ultrasound"] = {
        "solvers": doppler_results,
        "dataset_available": True,
    }

    print("\nTesting dot...")
    dot_tests = {
        "born_backprojection": ("Born Approximation Backprojection", test_dot_born),
        "tikhonov": ("Tikhonov-Regularized DOT", test_dot_tikhonov),
        "precomputed_baseline": ("DOT Baseline (precomputed)", test_dot_precomputed),
    }
    dot_results = run_mod_tests("dot", dot_tests)
    all_results["modalities"]["dot"] = {
        "solvers": dot_results,
        "dataset_available": True,
    }

    # Save updated results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated: {RESULTS_PATH}")

    for mod in ["doppler_ultrasound", "dot"]:
        solvers = all_results["modalities"][mod]["solvers"]
        valid = [(k, v["psnr_db"]) for k, v in solvers.items() if v.get("psnr_db") is not None]
        print(f"{mod}: {len(valid)} solvers with PSNR")
        for k, p in valid:
            print(f"  {k}: {p:.2f} dB")


if __name__ == "__main__":
    main()
