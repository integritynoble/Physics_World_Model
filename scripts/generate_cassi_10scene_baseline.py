#!/usr/bin/env python3
"""
CASSI Baseline Generator: All 10 Scenes, No Mismatch Results

Generates reconstruction results for all 10 TSA benchmark scenes without any
mismatch parameters injected:
- All operator parameters are nominal (dx=0, dy=0, theta=0, a1=2.0, a2=0.0, alpha=0, sigma=0)
- Reconstructs all scenes with GAP-TV
- Provides baseline for understanding CASSI performance without calibration errors
- Compare to W1 published results
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import json

# Add package to path
pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))

from pwm_core.data.loaders.kaist import KAISTDataset


def compute_psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute SSIM (Structural Similarity Index) - per-band average."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Compute SSIM for each spectral band
        scores = []
        for l in range(x.shape[2]):
            s = ssim(x[:, :, l], y[:, :, l], data_range=1.0)
            scores.append(s)
        return float(np.mean(scores))
    except ImportError:
        # Fallback: simple correlation-based metric
        x_flat = x.flatten()
        y_flat = y.flatten()
        corr = np.corrcoef(x_flat, y_flat)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.5


def compute_sam(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spectral Angle Mapper (SAM) in degrees."""
    # Reshape to (num_pixels, num_bands)
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])

    # Normalize
    x_norm = np.sqrt(np.sum(x_flat**2, axis=1, keepdims=True)) + 1e-8
    y_norm = np.sqrt(np.sum(y_flat**2, axis=1, keepdims=True)) + 1e-8
    x_norm = x_flat / x_norm
    y_norm = y_flat / y_norm

    # Dot product (clamped to avoid numerical errors)
    dots = np.sum(x_norm * y_norm, axis=1)
    dots = np.clip(dots, -1.0, 1.0)

    # Convert to degrees
    angles = np.arccos(dots) * 180.0 / np.pi
    return float(np.mean(angles))


def _warp_mask2d(mask2d, dx=0.0, dy=0.0, theta_deg=0.0):
    """Warp mask with affine transform."""
    from scipy.ndimage import affine_transform as _at
    H, W = mask2d.shape
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)
    M = R.T
    shift = np.array([dy, dx], dtype=np.float32)
    offset = (center - shift) - M @ center
    warped = _at(mask2d.astype(np.float32), matrix=M, offset=offset,
                 output_shape=(H, W), order=1, mode="constant", cval=0.0)
    return np.clip(warped, 0.0, 1.0).astype(np.float32)


def _make_dispersion_offsets(s_nom, dir_rot_deg):
    """Compute dispersion offsets."""
    theta = np.deg2rad(dir_rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    s_f = s_nom.astype(np.float32)
    return s_f * c, s_f * s


def _cassi_forward(x_hwl, mask2d, s_nom, dir_rot_deg):
    """CASSI forward model with integer offsets."""
    H, W, L = x_hwl.shape
    dx_f, dy_f = _make_dispersion_offsets(s_nom, dir_rot_deg)
    dx_i = np.rint(dx_f).astype(np.int32)
    dy_i = np.rint(dy_f).astype(np.int32)
    if dx_i.min() < 0:
        dx_i = dx_i - int(dx_i.min())
    if dy_i.min() < 0:
        dy_i = dy_i - int(dy_i.min())
    Wp = W + int(dx_i.max())
    Hp = H + int(dy_i.max())
    y = np.zeros((Hp, Wp), dtype=np.float32)
    for l in range(L):
        oy, ox = int(dy_i[l]), int(dx_i[l])
        y[oy:oy + H, ox:ox + W] += mask2d * x_hwl[:, :, l]
    return y


def _gap_tv_recon(y, cube_shape, mask2d, s_nom, dir_rot_deg,
                  max_iter=120, lam=1.0, tv_weight=0.4, tv_iter=5,
                  x_init=None, gauss_sigma=0.5):
    """GAP-TV reconstruction."""
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None

    H, W, L = cube_shape
    dx_f, dy_f = _make_dispersion_offsets(s_nom, dir_rot_deg)
    dx_i = np.rint(dx_f).astype(np.int32)
    dy_i = np.rint(dy_f).astype(np.int32)
    if dx_i.min() < 0:
        dx_i = dx_i - int(dx_i.min())
    if dy_i.min() < 0:
        dy_i = dy_i - int(dy_i.min())
    Wp = W + int(dx_i.max())
    Hp = H + int(dy_i.max())
    y_pad = np.zeros((Hp, Wp), dtype=np.float32)
    hh, ww = min(Hp, y.shape[0]), min(Wp, y.shape[1])
    y_pad[:hh, :ww] = y[:hh, :ww]
    y_w = y_pad

    Phi_sum = np.zeros((Hp, Wp), dtype=np.float32)
    for l in range(L):
        oy, ox = int(dy_i[l]), int(dx_i[l])
        Phi_sum[oy:oy + H, ox:ox + W] += mask2d
    Phi_sum = np.maximum(Phi_sum, 1.0)

    def _A_fwd(x_hwl):
        return _cassi_forward(x_hwl, mask2d, s_nom, dir_rot_deg)

    def _A_adj(r_hw):
        x = np.zeros((H, W, L), dtype=np.float32)
        for l in range(L):
            oy, ox = int(dy_i[l]), int(dx_i[l])
            x[:, :, l] += r_hw[oy:oy + H, ox:ox + W] * mask2d
        return x

    if x_init is not None:
        x = x_init.copy()
    else:
        x = _A_adj(y_w / Phi_sum)

    y1 = y_w.copy()
    for _ in range(max_iter):
        yb = _A_fwd(x)
        y1 = y1 + (y_w - yb)
        x = x + lam * _A_adj((y1 - yb) / Phi_sum)
        if denoise_tv_chambolle is not None:
            for l in range(L):
                x[:, :, l] = denoise_tv_chambolle(
                    x[:, :, l], weight=tv_weight, max_num_iter=tv_iter)
        else:
            from scipy.ndimage import gaussian_filter
            for l in range(L):
                x[:, :, l] = gaussian_filter(x[:, :, l], sigma=gauss_sigma)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def load_tsa_scene(scene_idx: int):
    """Load TSA scene."""
    try:
        from scipy.io import loadmat as _loadmat
        pkg_root = Path(__file__).parent.parent
        tsa_search_paths = [
            pkg_root / "datasets" / "TSA_simu_data",
            pkg_root.parent.parent / "datasets" / "TSA_simu_data",
            pkg_root / "data" / "TSA_simu_data",
            Path(__file__).parent / "TSA_simu_data",
            Path.home() / "datasets" / "TSA_simu_data",
            Path.home() / "Data" / "TSA_simu_data",
            Path.home() / "MST-main" / "datasets" / "TSA_simu_data",
            Path("/data/TSA_simu_data"),
            Path("/mnt/data/TSA_simu_data"),
        ]
        for data_dir in tsa_search_paths:
            mask_path = data_dir / "mask.mat"
            truth_dir = data_dir / "Truth"
            if mask_path.exists() and truth_dir.exists():
                mask_data = _loadmat(str(mask_path))
                mask2d_nom = mask_data["mask"].astype(np.float32)

                scene_path = truth_dir / f"scene{scene_idx:02d}.mat"
                if not scene_path.exists():
                    scenes = sorted(truth_dir.glob("scene*.mat"))
                    if scene_idx - 1 < len(scenes):
                        scene_path = scenes[scene_idx - 1]

                if scene_path.exists():
                    scene_data = _loadmat(str(scene_path))
                    for key in ["img", "cube", "hsi", "data"]:
                        if key in scene_data:
                            cube = scene_data[key].astype(np.float32)
                            break
                    if cube is None:
                        for key in scene_data:
                            if not key.startswith("__"):
                                cube = scene_data[key].astype(np.float32)
                                break

                    if cube is not None:
                        if cube.ndim == 3 and cube.shape[0] < cube.shape[1]:
                            cube = np.transpose(cube, (1, 2, 0))
                        return cube, mask2d_nom
    except Exception as e:
        print(f"TSA loading failed: {e}")

    return None, None


def main():
    print("\n" + "="*70)
    print("CASSI Baseline Generator: All 10 Scenes (No Mismatch)")
    print("="*70)

    results_by_scene = {}
    all_psnr = []
    all_ssim = []
    all_sam = []

    # Process all 10 scenes
    for scene_idx in range(1, 11):
        scene_name = f"scene{scene_idx:02d}"
        print(f"\n{"-"*70}")
        print(f"Loading {scene_name}...")
        print(f"{"-"*70}")

        cube_gt, mask_nom = load_tsa_scene(scene_idx)
        if cube_gt is None or mask_nom is None:
            print(f"❌ Failed to load {scene_name}, skipping...")
            continue

        H, W, L = cube_gt.shape
        print(f"Shape: {H}x{W}x{L}")
        print(f"Mask range: [{mask_nom.min():.3f}, {mask_nom.max():.3f}]")

        # Nominal parameters (no mismatch)
        s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)

        # Add realistic noise
        rng = np.random.default_rng(42)
        alpha = 1000.0  # Poisson gain
        sigma = 5.0     # Gaussian read noise

        # Perfect forward model (no mismatch)
        mask_used = _warp_mask2d(mask_nom, dx=0.0, dy=0.0, theta_deg=0.0)
        y_clean = _cassi_forward(cube_gt, mask_used, s_nom, dir_rot_deg=0.0)
        y_clean = np.maximum(y_clean, 0.0)

        # Add noise
        lam = np.clip(alpha * y_clean, 0.0, 1e9)
        y = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
        y += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)

        print(f"Measurement shape: {y.shape}")
        print(f"Measurement range: [{y.min():.3f}, {y.max():.3f}]")

        # Reconstruction with nominal (correct) parameters
        print(f"Reconstructing with GAP-TV (nominal parameters)...")
        x_recon = _gap_tv_recon(y, (H, W, L), mask_used, s_nom,
                                dir_rot_deg=0.0, max_iter=120)

        # Compute metrics
        psnr = compute_psnr(x_recon, cube_gt)
        ssim = compute_ssim(x_recon, cube_gt)
        sam = compute_sam(x_recon, cube_gt)

        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  SAM:  {sam:.2f}°")

        results_by_scene[scene_name] = {
            "psnr_db": float(psnr),
            "ssim": float(ssim),
            "sam_deg": float(sam),
            "measurement_shape": list(y.shape),
            "gt_shape": list(cube_gt.shape),
        }

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_sam.append(sam)

    if not all_psnr:
        print("\n❌ No scenes loaded! Please check TSA_simu_data location.")
        print("Expected paths:")
        print("  - ~/datasets/TSA_simu_data/")
        print("  - MST-main/datasets/TSA_simu_data/")
        return

    # Compute overall averages
    print("\n" + "="*70)
    print("BASELINE SUMMARY: ALL 10 SCENES (No Mismatch)")
    print("="*70)

    overall_psnr = float(np.mean(all_psnr))
    overall_ssim = float(np.mean(all_ssim))
    overall_sam = float(np.mean(all_sam))

    print(f"\nPer-Scene Results:")
    print(f"{"-"*70}")
    print(f"{'Scene':<12} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'SAM (°)':<8}")
    print(f"{"-"*70}")
    for scene_name in sorted(results_by_scene.keys()):
        r = results_by_scene[scene_name]
        print(f"{scene_name:<12} | {r['psnr_db']:<12.2f} | {r['ssim']:<8.4f} | {r['sam_deg']:<8.2f}")

    print(f"{"-"*70}")
    print(f"{'AVERAGE':<12} | {overall_psnr:<12.2f} | {overall_ssim:<8.4f} | {overall_sam:<8.2f}")
    print(f"{"-"*70}")

    print(f"\nComparison to Published W1 Results (from cassi.md):")
    print(f"  Published GAP-TV average: 14.92 dB")
    print(f"  Our baseline (no-mismatch): {overall_psnr:.2f} dB")
    print(f"  Difference: {overall_psnr - 14.92:+.2f} dB")

    # Save results
    results_json = {
        "description": "CASSI Baseline: All 10 Scenes, No Mismatch Reconstruction",
        "dataset": "TSA Simulation Benchmark (10 scenes, 256x256x28, step=2 dispersion)",
        "scenes_loaded": len(results_by_scene),
        "noise_model": {
            "poisson_gain": alpha,
            "gaussian_sigma": sigma,
        },
        "nominal_parameters": {
            "mask_dx_px": 0.0,
            "mask_dy_px": 0.0,
            "mask_theta_deg": 0.0,
            "disp_a1": 2.0,
            "disp_a2": 0.0,
            "disp_alpha_deg": 0.0,
            "psf_sigma_px": 0.0,
        },
        "reconstruction_parameters": {
            "algorithm": "GAP-TV",
            "max_iterations": 120,
            "step_size_lam": 1.0,
            "tv_weight": 0.4,
            "tv_iterations": 5,
        },
        "per_scene_results": results_by_scene,
        "overall_metrics": {
            "mean_psnr_db": overall_psnr,
            "mean_ssim": overall_ssim,
            "mean_sam_deg": overall_sam,
            "std_psnr_db": float(np.std(all_psnr)),
            "std_ssim": float(np.std(all_ssim)),
            "std_sam_deg": float(np.std(all_sam)),
        },
        "comparison_to_published_w1": {
            "note": "Published W1 GAP-TV results from cassi.md",
            "published_gap_tv_psnr_db": 14.92,
            "published_gap_tv_ssim": 0.1946,
            "our_baseline_psnr_db": overall_psnr,
            "our_baseline_ssim": overall_ssim,
            "difference_psnr_db": overall_psnr - 14.92,
            "difference_ssim": overall_ssim - 0.1946,
        },
    }

    output_file = Path(__file__).parent.parent / "pwm" / "reports" / "cassi_baseline_10scenes_no_mismatch.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
