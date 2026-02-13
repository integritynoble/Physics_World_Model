#!/usr/bin/env python3
"""
CACTI Baseline Generator: No Mismatch Results

Generates reconstruction results without any mismatch parameters injected:
- All operator parameters are nominal (no mask translation, rotation, blur)
- Reconstructs all 6 CACTI benchmark scenes
- Provides baseline for understanding how good CACTI can be without calibration errors
"""

import sys
import numpy as np
from pathlib import Path
import json

# Add package to path
pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))

from pwm_core.data.loaders.cacti_bench import CACTIBenchmark


def compute_psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute SSIM (Structural Similarity Index) - simplified version."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Compute SSIM for the video frames (HxWx8)
        scores = []
        for t in range(x.shape[2]):
            s = ssim(x[:, :, t], y[:, :, t], data_range=1.0)
            scores.append(s)
        return float(np.mean(scores))
    except ImportError:
        # Fallback: simple correlation-based metric
        x_flat = x.flatten()
        y_flat = y.flatten()
        return float(np.corrcoef(x_flat, y_flat)[0, 1])


def _cacti_forward(x_hwt, mask_hwt):
    """CACTI forward model: y = sum_t(mask_t * x_t)."""
    H, W, T = x_hwt.shape
    y = np.zeros((H, W), dtype=np.float32)
    for t in range(T):
        y += mask_hwt[:, :, t] * x_hwt[:, :, t]
    return y.astype(np.float32)


def _gap_tv_recon(y, cube_shape, mask_hwt, max_iter=100, lam=1.0,
                  tv_weight=0.1, tv_iter=5, x_init=None, gauss_sigma=0.5):
    """GAP-TV reconstruction for CACTI temporal imaging."""
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None

    H, W, T = cube_shape

    # Compute adjoint mask sum for normalization
    Phi_sum = np.sum(mask_hwt, axis=2)  # (H, W)
    Phi_sum = np.maximum(Phi_sum, 1.0)

    def _A_fwd(x_hwt):
        """Forward: y = sum_t(mask_t * x_t)"""
        return _cacti_forward(x_hwt, mask_hwt)

    def _A_adj(y_hw):
        """Adjoint: x_t = mask_t * y / Phi_sum"""
        x = np.zeros((H, W, T), dtype=np.float32)
        for t in range(T):
            x[:, :, t] = mask_hwt[:, :, t] * y_hw / Phi_sum
        return x

    # Initialize
    if x_init is not None:
        x = x_init.copy()
    else:
        x = _A_adj(y)

    y1 = y.copy()
    for iteration in range(max_iter):
        yb = _A_fwd(x)
        y1 = y1 + (y - yb)
        x = x + lam * _A_adj(y1 - yb)

        # TV denoising per frame
        if denoise_tv_chambolle is not None:
            for t in range(T):
                x[:, :, t] = denoise_tv_chambolle(
                    x[:, :, t], weight=tv_weight, max_num_iter=tv_iter)
        else:
            from scipy.ndimage import gaussian_filter
            for t in range(T):
                x[:, :, t] = gaussian_filter(x[:, :, t], sigma=gauss_sigma)

        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def main():
    print("\n" + "="*70)
    print("CACTI Baseline Generator: No Mismatch Results")
    print("="*70)

    # Load CACTI benchmark
    try:
        benchmark = CACTIBenchmark()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease set CACTI_BENCHMARK_DIR or ensure data is available at:")
        print("  /home/spiritai/PnP-SCI_python-master/dataset/cacti/grayscale_benchmark/")
        return

    results_by_scene = {}
    all_psnr = []
    all_ssim = []

    # Process each scene (first group of 8 frames only for efficiency)
    for scene_name, group_gt, mask_hwt, group_meas in benchmark:
        if scene_name not in results_by_scene:
            results_by_scene[scene_name] = []

        print(f"\n{"-"*70}")
        print(f"Scene: {scene_name}")
        print(f"Group shape: {group_gt.shape}")
        print(f"Mask shape: {mask_hwt.shape}")
        print(f"Measurement shape: {group_meas.shape}")
        print(f"{"-"*70}")

        # Add realistic noise (Poisson + Gaussian) similar to CASSI
        rng = np.random.default_rng(42)
        alpha = 1000.0  # Poisson gain (higher = lower noise)
        sigma = 5.0     # Gaussian read noise

        # Measurement with noise
        lam = np.clip(alpha * group_meas, 0.0, 1e9)
        y_noisy = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
        y_noisy += rng.normal(0.0, sigma, size=group_meas.shape).astype(np.float32)
        y_noisy = np.clip(y_noisy, 0, 1)

        # Reconstruct with GAP-TV
        print(f"Reconstructing with GAP-TV (no mismatch)...")
        x_recon = _gap_tv_recon(y_noisy, group_gt.shape, mask_hwt,
                                max_iter=100, lam=0.5, tv_weight=0.1)

        # Compute metrics
        psnr = compute_psnr(x_recon, group_gt)
        ssim = compute_ssim(x_recon, group_gt)

        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")

        results_by_scene[scene_name].append({
            "group_idx": len(results_by_scene[scene_name]),
            "psnr_db": float(psnr),
            "ssim": float(ssim),
            "measurement_shape": list(group_meas.shape),
            "gt_shape": list(group_gt.shape),
        })

        all_psnr.append(psnr)
        all_ssim.append(ssim)

    # Compute per-scene averages
    print("\n" + "="*70)
    print("PER-SCENE AVERAGES (No Mismatch Baseline)")
    print("="*70)

    scene_averages = {}
    for scene_name in sorted(results_by_scene.keys()):
        results = results_by_scene[scene_name]
        psnr_vals = [r["psnr_db"] for r in results]
        ssim_vals = [r["ssim"] for r in results]

        avg_psnr = float(np.mean(psnr_vals))
        avg_ssim = float(np.mean(ssim_vals))

        scene_averages[scene_name] = {
            "psnr_db": avg_psnr,
            "ssim": avg_ssim,
            "num_groups": len(results),
        }

        print(f"{scene_name:15s} | PSNR: {avg_psnr:6.2f} dB | SSIM: {avg_ssim:.4f}")

    # Overall average
    overall_psnr = float(np.mean(all_psnr))
    overall_ssim = float(np.mean(all_ssim))

    print("\n" + "-"*70)
    print(f"{'OVERALL AVERAGE':15s} | PSNR: {overall_psnr:6.2f} dB | SSIM: {overall_ssim:.4f}")
    print("-"*70)

    # Save results
    results_json = {
        "description": "CACTI Baseline: No Mismatch Reconstruction Results",
        "dataset": "CACTI Grayscale SCI Video Benchmark (6 scenes, 8:1 compression)",
        "noise_model": {
            "poisson_gain": alpha,
            "gaussian_sigma": sigma,
        },
        "reconstruction_parameters": {
            "algorithm": "GAP-TV",
            "max_iterations": 100,
            "step_size_lam": 0.5,
            "tv_weight": 0.1,
            "tv_iterations": 5,
        },
        "per_scene_averages": scene_averages,
        "overall_metrics": {
            "mean_psnr_db": overall_psnr,
            "mean_ssim": overall_ssim,
            "std_psnr_db": float(np.std(all_psnr)),
            "std_ssim": float(np.std(all_ssim)),
        },
        "comparison_to_published_results": {
            "note": "Published GAP-TV results from cacti.md",
            "gap_tv_average_psnr_db": 26.62,
            "our_baseline_psnr_db": overall_psnr,
            "difference_db": overall_psnr - 26.62,
        },
    }

    output_file = Path(__file__).parent.parent / "pwm" / "reports" / "cacti_baseline_no_mismatch.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"\nWithout any mismatch (nominal parameters):")
    print(f"  Mean PSNR: {overall_psnr:.2f} dB")
    print(f"  Mean SSIM: {overall_ssim:.4f}")
    print(f"\nComparison to published GAP-TV results (cacti.md):")
    print(f"  Published average: 26.62 dB")
    print(f"  Our baseline:      {overall_psnr:.2f} dB")
    print(f"  Difference:        {overall_psnr - 26.62:+.2f} dB")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
