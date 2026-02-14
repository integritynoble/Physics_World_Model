#!/usr/bin/env python3
"""
CASSI Noise Sensitivity Analysis

Tests how reconstruction quality varies with different noise levels.
Helps determine if aggressive noise model is the limiting factor.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import json

# Add package to path
pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))


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
        scores = []
        for l in range(x.shape[2]):
            s = ssim(x[:, :, l], y[:, :, l], data_range=1.0)
            scores.append(s)
        return float(np.mean(scores))
    except ImportError:
        x_flat = x.flatten()
        y_flat = y.flatten()
        corr = np.corrcoef(x_flat, y_flat)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.5


def compute_sam(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spectral Angle Mapper (SAM) in degrees."""
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    x_norm = np.sqrt(np.sum(x_flat**2, axis=1, keepdims=True)) + 1e-8
    y_norm = np.sqrt(np.sum(y_flat**2, axis=1, keepdims=True)) + 1e-8
    x_norm = x_flat / x_norm
    y_norm = y_flat / y_norm
    dots = np.sum(x_norm * y_norm, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
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
    """CASSI forward model."""
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
                  max_iter=120, lam=1.0, tv_weight=0.4, tv_iter=5):
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
                x[:, :, l] = gaussian_filter(x[:, :, l], sigma=0.5)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def load_tsa_scene(scene_idx: int):
    """Load TSA scene."""
    try:
        from scipy.io import loadmat as _loadmat
        tsa_search_paths = [
            Path.home() / "MST-main" / "datasets" / "TSA_simu_data",
            Path("/data/TSA_simu_data"),
        ]
        for data_dir in tsa_search_paths:
            mask_path = data_dir / "mask.mat"
            truth_dir = data_dir / "Truth"
            if mask_path.exists() and truth_dir.exists():
                mask_data = _loadmat(str(mask_path))
                mask2d_nom = mask_data["mask"].astype(np.float32)

                scene_path = truth_dir / f"scene{scene_idx:02d}.mat"
                if scene_path.exists():
                    scene_data = _loadmat(str(scene_path))
                    for key in ["img", "cube", "hsi", "data"]:
                        if key in scene_data:
                            cube = scene_data[key].astype(np.float32)
                            break
                    else:
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
    print("\n" + "="*80)
    print("CASSI Noise Sensitivity Analysis")
    print("="*80)

    # Load scene01 (representative)
    print("\nLoading scene01...")
    cube_gt, mask_nom = load_tsa_scene(1)
    if cube_gt is None or mask_nom is None:
        print("❌ Failed to load scene01")
        return

    H, W, L = cube_gt.shape
    print(f"Scene shape: {H}x{W}x{L}")

    # Nominal parameters
    s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)
    mask_used = _warp_mask2d(mask_nom, dx=0.0, dy=0.0, theta_deg=0.0)
    y_clean = _cassi_forward(cube_gt, mask_used, s_nom, dir_rot_deg=0.0)
    y_clean = np.maximum(y_clean, 0.0)

    # Define noise scenarios
    noise_scenarios = [
        {
            "name": "No noise",
            "poisson_gain": 1e9,
            "gaussian_sigma": 0.0,
            "description": "Perfect measurement (theoretical upper bound)"
        },
        {
            "name": "Very low noise",
            "poisson_gain": 100000.0,
            "gaussian_sigma": 0.1,
            "description": "High SNR (ideal lab conditions)"
        },
        {
            "name": "Low noise",
            "poisson_gain": 50000.0,
            "gaussian_sigma": 0.5,
            "description": "High SNR (good camera)"
        },
        {
            "name": "Medium noise",
            "poisson_gain": 10000.0,
            "gaussian_sigma": 1.0,
            "description": "Medium SNR (typical camera)"
        },
        {
            "name": "Medium-High noise",
            "poisson_gain": 5000.0,
            "gaussian_sigma": 2.0,
            "description": "Lower SNR"
        },
        {
            "name": "High noise",
            "poisson_gain": 2000.0,
            "gaussian_sigma": 3.0,
            "description": "Low SNR (current baseline)"
        },
        {
            "name": "Very high noise (current)",
            "poisson_gain": 1000.0,
            "gaussian_sigma": 5.0,
            "description": "Very low SNR (aggressive model)"
        },
    ]

    results = {}
    rng = np.random.default_rng(42)

    print("\n" + "="*80)
    print("Testing different noise levels on scene01")
    print("="*80)

    for scenario in noise_scenarios:
        name = scenario["name"]
        alpha = scenario["poisson_gain"]
        sigma = scenario["gaussian_sigma"]

        print(f"\n{"-"*80}")
        print(f"Scenario: {name}")
        print(f"  Poisson gain: {alpha:.1f}")
        print(f"  Gaussian σ: {sigma:.2f}")
        print(f"  Description: {scenario['description']}")
        print(f"{"-"*80}")

        # Add noise
        lam = np.clip(alpha * y_clean, 0.0, 1e9)
        y_noisy = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
        y_noisy += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)
        y_noisy = np.clip(y_noisy, 0, 10)  # Clip to reasonable range

        # Compute SNR
        signal_power = np.mean(y_clean ** 2)
        noise_power = np.mean((y_noisy - y_clean) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        print(f"  Measurement SNR: {snr_db:.2f} dB")
        print(f"  Measurement range: [{y_noisy.min():.3f}, {y_noisy.max():.3f}]")

        # Reconstruct
        print(f"  Reconstructing with GAP-TV...")
        x_recon = _gap_tv_recon(y_noisy, (H, W, L), mask_used, s_nom,
                                dir_rot_deg=0.0, max_iter=120)

        # Compute metrics
        psnr = compute_psnr(x_recon, cube_gt)
        ssim = compute_ssim(x_recon, cube_gt)
        sam = compute_sam(x_recon, cube_gt)

        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  SAM:  {sam:.2f}°")

        results[name] = {
            "poisson_gain": alpha,
            "gaussian_sigma": sigma,
            "snr_db": float(snr_db),
            "psnr_db": float(psnr),
            "ssim": float(ssim),
            "sam_deg": float(sam),
            "description": scenario["description"],
        }

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    print(f"\n{'Scenario':<30} | {'SNR (dB)':<10} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Gain from current':<15}")
    print(f"{"-"*30}-+-{"-"*10}-+-{"-"*12}-+-{"-"*8}-+-{"-"*15}")

    current_psnr = results["Very high noise (current)"]["psnr_db"]

    for name in results:
        r = results[name]
        gain = r["psnr_db"] - current_psnr
        gain_str = f"+{gain:.2f} dB" if gain >= 0 else f"{gain:.2f} dB"
        print(f"{name:<30} | {r['snr_db']:>10.2f} | {r['psnr_db']:>12.2f} | {r['ssim']:>8.4f} | {gain_str:<15}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    no_noise_psnr = results["No noise"]["psnr_db"]
    very_high_psnr = results["Very high noise (current)"]["psnr_db"]
    gap = no_noise_psnr - very_high_psnr

    print(f"\nNoise Impact on Reconstruction:")
    print(f"  • No-noise PSNR:      {no_noise_psnr:.2f} dB")
    print(f"  • Current model PSNR: {very_high_psnr:.2f} dB")
    print(f"  • Difference:         {gap:.2f} dB")
    print(f"\n  → Noise causes {gap:.2f} dB degradation!")

    # Find "reasonable" noise level
    medium_psnr = results["Medium noise"]["psnr_db"]
    medium_gain = medium_psnr - very_high_psnr
    print(f"\nRecommended Noise Model (Medium SNR):")
    print(f"  • Poisson gain: 10000.0")
    print(f"  • Gaussian σ: 1.0")
    print(f"  • Expected PSNR improvement: +{medium_gain:.2f} dB")
    print(f"  • Reasoning: Realistic camera + SNR conditions")

    # Save results
    output_file = Path(__file__).parent.parent / "pwm" / "reports" / "cassi_noise_sensitivity.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "description": "CASSI Noise Sensitivity Analysis - Scene01",
        "no_noise_psnr_db": no_noise_psnr,
        "current_model_psnr_db": very_high_psnr,
        "noise_impact_db": gap,
        "recommendation": {
            "poisson_gain": 10000.0,
            "gaussian_sigma": 1.0,
            "expected_psnr_db": medium_psnr,
            "expected_gain_db": medium_gain,
        },
        "all_scenarios": results,
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
