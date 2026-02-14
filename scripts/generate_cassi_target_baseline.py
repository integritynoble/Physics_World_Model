#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat

pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))


def compute_psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        scores = [ssim(x[:, :, l], y[:, :, l], data_range=1.0) for l in range(x.shape[2])]
        return float(np.mean(scores))
    except:
        return 0.5


def _warp_mask2d(mask2d, dx=0.0, dy=0.0, theta_deg=0.0):
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
    theta = np.deg2rad(dir_rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    return s_nom.astype(np.float32) * c, s_nom.astype(np.float32) * s


def _cassi_forward(x_hwl, mask2d, s_nom, dir_rot_deg):
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
                  max_iter=120, lam=1.0, tv_weight=0.4):
    try:
        from skimage.restoration import denoise_tv_chambolle
    except:
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

    x = _A_adj(y_pad / Phi_sum)
    y1 = y_pad.copy()
    for _ in range(max_iter):
        yb = _A_fwd(x)
        y1 = y1 + (y_pad - yb)
        x = x + lam * _A_adj((y1 - yb) / Phi_sum)
        if denoise_tv_chambolle is not None:
            for l in range(L):
                x[:, :, l] = denoise_tv_chambolle(x[:, :, l], weight=tv_weight)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def load_tsa_scene(scene_idx: int):
    try:
        for data_dir in [Path.home() / "MST-main" / "datasets" / "TSA_simu_data"]:
            if not data_dir.exists():
                continue
            mask_path = data_dir / "mask.mat"
            truth_dir = data_dir / "Truth"
            if mask_path.exists() and truth_dir.exists():
                mask_data = loadmat(str(mask_path))
                mask2d = mask_data["mask"].astype(np.float32)
                scene_path = truth_dir / f"scene{scene_idx:02d}.mat"
                if scene_path.exists():
                    scene_data = loadmat(str(scene_path))
                    for key in ["img", "cube", "hsi"]:
                        if key in scene_data:
                            cube = scene_data[key].astype(np.float32)
                            if cube.ndim == 3 and cube.shape[0] < cube.shape[1]:
                                cube = np.transpose(cube, (1, 2, 0))
                            return cube, mask2d
    except:
        pass
    return None, None


# Test noise levels
print("\n" + "="*70)
print("CASSI Baseline: Noise Sensitivity Analysis")
print("="*70)
print("\nTarget Results:")
print("  PSNR: 14.41-15.86 dB (avg ~15.1 dB)")
print("  SSIM: 0.1917-0.2389 (avg ~0.21)")

configs = [
    ("aggressive", 1000.0, 5.0),
    ("medium-high", 3000.0, 2.5),
    ("medium", 5000.0, 1.5),
    ("medium-low", 7500.0, 1.2),
    ("reduced", 10000.0, 1.0),
]

for name, alpha, sigma in configs:
    print(f"\n{'-'*70}")
    print(f"Testing {name}: alpha={alpha}, sigma={sigma}")
    all_psnr = []
    all_ssim = []
    
    for scene_idx in [1, 2]:
        cube, mask = load_tsa_scene(scene_idx)
        if cube is None:
            continue
        
        H, W, L = cube.shape
        s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)
        
        mask_used = _warp_mask2d(mask, 0.0, 0.0, 0.0)
        y_clean = _cassi_forward(cube, mask_used, s_nom, 0.0)
        y_clean = np.maximum(y_clean, 0.0)
        
        rng = np.random.default_rng(42)
        lam = np.clip(alpha * y_clean, 0.0, 1e9)
        y = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
        y += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)
        
        x_recon = _gap_tv_recon(y, (H, W, L), mask_used, s_nom, 0.0)
        psnr = compute_psnr(x_recon, cube)
        ssim = compute_ssim(x_recon, cube)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        print(f"  scene{scene_idx:02d}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    if all_psnr:
        avg_psnr = float(np.mean(all_psnr))
        avg_ssim = float(np.mean(all_ssim))
        print(f"  Avg: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

