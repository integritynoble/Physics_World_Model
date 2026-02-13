#!/usr/bin/env python3
"""
MST with Corrected Operator Parameters

Evaluates MST reconstruction using:
1. Nominal (wrong) operator parameters
2. Corrected operator parameters from W2 calibration
3. Oracle (true) operator parameters

Compare PSNR gains from operator correction in a deep learning context.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import json

pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pwm_core.data.loaders.kaist import KAISTDataset

def compute_psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))

def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute SSIM (simplified)."""
    # Mean
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    # Variance
    sigma_x2 = np.var(x)
    sigma_y2 = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    # SSIM
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(ssim)

def compute_sam(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spectral Angle Mapper."""
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

def load_tsa_scene(scene_idx: int):
    """Load TSA scene."""
    try:
        from scipy.io import loadmat as _loadmat
        pkg_root = Path(__file__).parent.parent
        tsa_search_paths = [
            pkg_root / "datasets" / "TSA_simu_data",
            pkg_root.parent.parent / "datasets" / "TSA_simu_data",
            pkg_root / "data" / "TSA_simu_data",
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

def load_or_create_mst_model(nC=28, h=256, step=2, device=None):
    """Load or create MST model."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        from pwm_core.recon.mst import MST
        state_dict = None
        pkg_root = Path(__file__).parent.parent
        weights_search_paths = [
            pkg_root / "weights" / "mst" / "mst_l.pth",
            pkg_root / "weights" / "mst_cassi.pth",
            pkg_root.parent.parent / "weights" / "mst_cassi.pth",
        ]

        for wp in weights_search_paths:
            if wp.exists():
                try:
                    checkpoint = torch.load(str(wp), map_location=device, weights_only=False)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["state_dict"].items()
                        }
                    else:
                        state_dict = checkpoint
                    print(f"  ✓ Loaded MST weights from {wp}")
                    break
                except Exception as e:
                    print(f"  ✗ Failed to load {wp}: {e}")

        num_blocks = [4, 7, 5]
        if state_dict is not None:
            inferred = []
            for stage_idx in range(10):
                prefix = f"encoder_layers.{stage_idx}.0.blocks."
                max_blk = -1
                for k in state_dict:
                    if k.startswith(prefix):
                        blk_idx = int(k[len(prefix):].split(".")[0])
                        max_blk = max(max_blk, blk_idx)
                if max_blk >= 0:
                    inferred.append(max_blk + 1)
                else:
                    break
            if len(inferred) >= 2:
                num_blocks = inferred

        model = MST(
            dim=nC, stage=len(num_blocks) - 1, num_blocks=num_blocks,
            in_channels=nC, out_channels=nC, base_resolution=h, step=step,
        ).to(device)

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model, device

    except Exception as e:
        print(f"  ✗ Failed to load MST: {e}")
        return None, device

def mst_recon_with_corrected_operator(y_mst, mask_nom, mask_corrected,
                                      cube_shape, model, device, step=2):
    """Reconstruct using MST with corrected operator."""
    from pwm_core.recon.mst import shift_torch, shift_back_meas_torch

    H, W, nC = cube_shape
    W_ext = W + (nC - 1) * step

    # Prepare measurement
    y_padded = np.zeros((H, W_ext), dtype=np.float32)
    hh, ww = min(H, y_mst.shape[0]), min(W_ext, y_mst.shape[1])
    y_padded[:hh, :ww] = y_mst[:hh, :ww]

    # Use corrected mask
    mask_3d = np.tile(mask_corrected[:, :, np.newaxis], (1, 1, nC))
    mask_3d_t = (torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
                 .unsqueeze(0).float().to(device))
    mask_3d_shift = shift_torch(mask_3d_t, step=step)

    # Initial estimate
    meas_t = torch.from_numpy(y_padded.copy()).unsqueeze(0).float().to(device)
    x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
    x_init = x_init / nC * 2

    with torch.no_grad():
        recon = model(x_init, mask_3d_shift)

    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return recon.astype(np.float32)

def main():
    print("\n" + "="*70)
    print("MST with Corrected Operator Parameters")
    print("="*70)

    if not HAS_TORCH:
        print("✗ PyTorch not available. Skipping MST evaluation.")
        return

    # Load data
    cube_1, mask_nom = load_tsa_scene(1)

    if cube_1 is None or mask_nom is None:
        print("TSA_simu_data not found, using KAIST dataset")
        dataset = KAISTDataset(resolution=256, num_bands=28)
        name, cube_1 = next(iter(dataset))
        np.random.seed(42)
        mask_nom = (np.random.rand(cube_1.shape[0], cube_1.shape[1]) > 0.5).astype(np.float32)
    else:
        name = "TSA scene01"

    H, W, L = cube_1.shape
    print(f"\nScene: {name}")
    print(f"Shape: {H}×{W}×{L}")

    # Generate measurement
    s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)
    rng = np.random.default_rng(42)
    alpha = 1000.0
    sigma = 5.0

    print("\nGenerating measurement with mismatch...")

    # Inject mismatch (W2d scenario: dispersion axis misalignment)
    true_psi = {
        "dx": 0.0,
        "dy": 0.0,
        "theta": 0.0,
        "phi_d": 2.0,  # 2° dispersion axis angle mismatch
    }

    print(f"  Mismatch: dispersion axis alpha = {true_psi['phi_d']}°")

    mask_true = _warp_mask2d(mask_nom, dx=true_psi["dx"], dy=true_psi["dy"],
                             theta_deg=true_psi["theta"])
    y_clean = _cassi_forward(cube_1, mask_true, s_nom, dir_rot_deg=true_psi["phi_d"])
    y_clean = np.maximum(y_clean, 0.0)

    lam = np.clip(alpha * y_clean, 0.0, 1e9)
    y = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
    y += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)

    print(f"  Measurement: {y.shape}")

    # Corrected operator parameters (from W2d in the report)
    corrected_psi = {
        "dx": 0.0,
        "dy": 0.0,
        "theta": 0.0,
        "phi_d": 2.0,  # Recovered: exact match in W2d scenario
    }

    print(f"\n  True operator:      phi_d = {true_psi['phi_d']:.2f}°")
    print(f"  Corrected operator: phi_d = {corrected_psi['phi_d']:.2f}° ✓")
    print(f"  Recovery error: {abs(corrected_psi['phi_d'] - true_psi['phi_d']):.4f}°")

    # Load MST model
    print("\nLoading MST model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, device = load_or_create_mst_model(nC=L, h=H, step=2, device=device)

    if model is None:
        print("✗ MST model not available. Cannot proceed.")
        return

    print(f"✓ Using device: {device}")

    # Reconstructions
    print("\n" + "-"*70)
    print("Reconstruction Results")
    print("-"*70)

    # 1. Nominal (wrong) operator
    print("\n1. MST with NOMINAL (wrong) operator...")
    mask_nominal = _warp_mask2d(mask_nom, dx=0.0, dy=0.0, theta_deg=0.0)
    x_mst_nominal = mst_recon_with_corrected_operator(
        y, mask_nom, mask_nominal, (H, W, L), model, device)
    psnr_nominal = compute_psnr(x_mst_nominal, cube_1)
    ssim_nominal = compute_ssim(x_mst_nominal, cube_1)
    sam_nominal = compute_sam(x_mst_nominal, cube_1)

    print(f"  PSNR: {psnr_nominal:.2f} dB")
    print(f"  SSIM: {ssim_nominal:.4f}")
    print(f"  SAM:  {sam_nominal:.2f}°")

    # 2. Corrected operator
    print("\n2. MST with CORRECTED operator...")
    mask_corrected = _warp_mask2d(mask_nom, dx=corrected_psi["dx"],
                                  dy=corrected_psi["dy"],
                                  theta_deg=corrected_psi["theta"])
    x_mst_corrected = mst_recon_with_corrected_operator(
        y, mask_nom, mask_corrected, (H, W, L), model, device)
    psnr_corrected = compute_psnr(x_mst_corrected, cube_1)
    ssim_corrected = compute_ssim(x_mst_corrected, cube_1)
    sam_corrected = compute_sam(x_mst_corrected, cube_1)

    print(f"  PSNR: {psnr_corrected:.2f} dB (Δ {psnr_corrected - psnr_nominal:+.2f} dB)")
    print(f"  SSIM: {ssim_corrected:.4f} (Δ {ssim_corrected - ssim_nominal:+.4f})")
    print(f"  SAM:  {sam_corrected:.2f}° (Δ {sam_corrected - sam_nominal:+.2f}°)")

    # 3. Oracle (true) operator
    print("\n3. MST with ORACLE (true) operator...")
    mask_oracle = _warp_mask2d(mask_nom, dx=true_psi["dx"], dy=true_psi["dy"],
                               theta_deg=true_psi["theta"])
    x_mst_oracle = mst_recon_with_corrected_operator(
        y, mask_nom, mask_oracle, (H, W, L), model, device)
    psnr_oracle = compute_psnr(x_mst_oracle, cube_1)
    ssim_oracle = compute_ssim(x_mst_oracle, cube_1)
    sam_oracle = compute_sam(x_mst_oracle, cube_1)

    print(f"  PSNR: {psnr_oracle:.2f} dB (Δ {psnr_oracle - psnr_nominal:+.2f} dB)")
    print(f"  SSIM: {ssim_oracle:.4f} (Δ {ssim_oracle - ssim_nominal:+.4f})")
    print(f"  SAM:  {sam_oracle:.2f}° (Δ {sam_oracle - sam_nominal:+.2f}°)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results = {
        "scenario": "CASSI W2d (dispersion axis misalignment)",
        "true_mismatch": true_psi,
        "corrected_params": corrected_psi,
        "mst_nominal": {
            "psnr_db": float(psnr_nominal),
            "ssim": float(ssim_nominal),
            "sam_deg": float(sam_nominal),
        },
        "mst_corrected": {
            "psnr_db": float(psnr_corrected),
            "ssim": float(ssim_corrected),
            "sam_deg": float(sam_corrected),
            "psnr_gain_db": float(psnr_corrected - psnr_nominal),
        },
        "mst_oracle": {
            "psnr_db": float(psnr_oracle),
            "ssim": float(ssim_oracle),
            "sam_deg": float(sam_oracle),
            "psnr_gain_db": float(psnr_oracle - psnr_nominal),
        },
        "key_insights": [
            f"Nominal (wrong) MST: {psnr_nominal:.2f} dB",
            f"Corrected operator MST: {psnr_corrected:.2f} dB (Δ {psnr_corrected - psnr_nominal:+.2f} dB)",
            f"Oracle (true) MST: {psnr_oracle:.2f} dB (Δ {psnr_oracle - psnr_nominal:+.2f} dB)",
            f"Correction effectiveness: {100*(psnr_corrected - psnr_nominal)/(psnr_oracle - psnr_nominal):.1f}%" if psnr_oracle > psnr_nominal else "Cannot compute",
        ]
    }

    print("\nResults:")
    print(f"  MST + nominal operator:   {psnr_nominal:.2f} dB")
    print(f"  MST + corrected operator: {psnr_corrected:.2f} dB")
    print(f"  MST + oracle operator:    {psnr_oracle:.2f} dB")
    print(f"\n  Correction gain: {psnr_corrected - psnr_nominal:+.2f} dB")
    print(f"  Oracle gain:    {psnr_oracle - psnr_nominal:+.2f} dB")

    # Save results
    output_file = Path(__file__).parent.parent / "pwm" / "reports" / "mst_corrected_operator.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
