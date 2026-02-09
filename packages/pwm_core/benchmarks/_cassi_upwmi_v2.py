# ========================================================================
# CASSI: UPWMI Algorithm 2 - Differentiable Calibration via Backprop
# Operator calibration: mask geo (dx, dy, theta) via gradient descent
#                        + dispersion direction (phi_d) via discrete search
# ========================================================================
import time as _time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict

import numpy as np


def compute_psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ====================================================================
# Component 1: DifferentiableMaskWarp
# ====================================================================
if HAS_TORCH:

    class DifferentiableMaskWarp(nn.Module):
        """Differentiable affine warp of a 2D mask via (dx, dy, theta_deg).

        Uses F.affine_grid + F.grid_sample so gradients flow back to all
        three parameters.
        """

        def __init__(self, mask2d_nominal: np.ndarray,
                     dx_init: float = 0.0, dy_init: float = 0.0,
                     theta_init: float = 0.0):
            super().__init__()
            self.dx = nn.Parameter(torch.tensor(dx_init, dtype=torch.float32))
            self.dy = nn.Parameter(torch.tensor(dy_init, dtype=torch.float32))
            self.theta_deg = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
            # Register nominal mask as buffer (not optimised)
            mask_t = torch.from_numpy(mask2d_nominal.astype(np.float32))
            self.register_buffer("mask_nom", mask_t.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]

        def forward(self) -> torch.Tensor:
            """Return warped mask [H, W] with gradients w.r.t. dx, dy, theta_deg."""
            theta_rad = self.theta_deg * (3.141592653589793 / 180.0)
            cos_t = torch.cos(theta_rad)
            sin_t = torch.sin(theta_rad)
            H, W = self.mask_nom.shape[2], self.mask_nom.shape[3]
            # Normalise translations to [-1, 1] grid coordinates
            # F.grid_sample uses (x, y) with x = col, y = row
            # dx is column shift, dy is row shift
            tx = -2.0 * self.dx / float(W)
            ty = -2.0 * self.dy / float(H)
            # Build 2x3 affine: rotation + translation
            row0 = torch.stack([cos_t, -sin_t, tx])
            row1 = torch.stack([sin_t, cos_t, ty])
            affine = torch.stack([row0, row1]).unsqueeze(0)  # [1, 2, 3]
            grid = F.affine_grid(affine, self.mask_nom.shape, align_corners=False)
            warped = F.grid_sample(self.mask_nom, grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=False)
            return warped.squeeze(0).squeeze(0).clamp(0.0, 1.0)  # [H, W]


    # ====================================================================
    # Component 2: DifferentiableCassiForward (sub-pixel, phi_d-aware)
    # ====================================================================

    class DifferentiableCassiForward(nn.Module):
        """Sub-pixel CASSI forward model, differentiable w.r.t. phi_d.

        Uses bilinear splatting instead of integer offsets so that
        gradients flow back through phi_d even when the dispersion
        direction rotation is small (< 1 degree).
        """

        def __init__(self, s_nom: np.ndarray, H: int, W: int, nC: int,
                     phi_d_max_deg: float = 0.5):
            super().__init__()
            self.register_buffer(
                "s_nom", torch.from_numpy(s_nom.astype(np.float32)))
            # Fixed canvas size from worst-case bounds (avoids dynamic shapes)
            max_dx = int(s_nom.max()) + 2
            max_dy = int(np.ceil(
                s_nom.max() * np.abs(np.sin(np.deg2rad(phi_d_max_deg))))) + 2
            self.Wp = W + max_dx
            self.Hp = H + max_dy

        def forward(self, x_bchw: torch.Tensor, mask2d: torch.Tensor,
                    phi_d_deg: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x_bchw:   [B, nC, H, W] reconstructed cube
                mask2d:   [H, W]  warped mask (carries gradients)
                phi_d_deg: scalar tensor (carries gradients)
            Returns:
                y: [B, Hp, Wp] simulated measurement
            """
            B, nC, H, W = x_bchw.shape
            phi_rad = phi_d_deg * (3.141592653589793 / 180.0)
            dx_f = self.s_nom * torch.cos(phi_rad)
            dy_f = self.s_nom * torch.sin(phi_rad)
            # Shift to non-negative offsets
            dx_f = dx_f - dx_f.min()
            dy_f = dy_f - dy_f.min()

            y = x_bchw.new_zeros(B, self.Hp, self.Wp)
            for l in range(nC):
                band = mask2d * x_bchw[:, l, :, :]  # [B, H, W]
                ox, oy = dx_f[l], dy_f[l]
                ox_fl, oy_fl = ox.floor(), oy.floor()
                fx, fy = ox - ox_fl, oy - oy_fl
                ix = int(ox_fl.detach().item())
                iy = int(oy_fl.detach().item())
                # Bilinear splatting (all 4 quadrants)
                w00 = (1 - fy) * (1 - fx)
                w01 = (1 - fy) * fx
                w10 = fy * (1 - fx)
                w11 = fy * fx
                y[:, iy:iy+H, ix:ix+W] = (
                    y[:, iy:iy+H, ix:ix+W] + w00 * band)
                if ix + 1 + W <= self.Wp:
                    y[:, iy:iy+H, ix+1:ix+1+W] = (
                        y[:, iy:iy+H, ix+1:ix+1+W] + w01 * band)
                if iy + 1 + H <= self.Hp:
                    y[:, iy+1:iy+1+H, ix:ix+W] = (
                        y[:, iy+1:iy+1+H, ix:ix+W] + w10 * band)
                if ix + 1 + W <= self.Wp and iy + 1 + H <= self.Hp:
                    y[:, iy+1:iy+1+H, ix+1:ix+1+W] = (
                        y[:, iy+1:iy+1+H, ix+1:ix+1+W] + w11 * band)
            return y


# ====================================================================
# Component 3: mst_recon_differentiable
# ====================================================================

def _mst_recon_differentiable(model, device, y_np, mask2d_torch, cube_shape,
                               step=2):
    """MST forward pass WITH gradient tracking through the mask input.

    The initial estimate x_init is detached so only mask carries grads.

    Args:
        model: loaded MST model (eval mode, but NOT wrapped in no_grad)
        device: torch device
        y_np: 2D measurement numpy array [Hy, Wy]
        mask2d_torch: [H, W] torch tensor WITH gradient graph
        cube_shape: (H, W, nC)
        step: dispersion step

    Returns:
        recon: [1, nC, H, W] tensor with grad graph through mask
    """
    from pwm_core.recon.mst import shift_torch, shift_back_meas_torch

    H, W, nC = cube_shape
    W_ext = W + (nC - 1) * step

    # Prepare measurement -> x_init (detached, no gradients)
    y_mst = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, y_np.shape[0])
    ww = min(W_ext, y_np.shape[1])
    y_mst[:hh, :ww] = y_np[:hh, :ww]
    meas_t = torch.from_numpy(y_mst.copy()).unsqueeze(0).float().to(device)
    x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
    x_init = (x_init / nC * 2).detach()  # detach: no grad through x_init

    # Prepare mask: [H, W] -> [1, nC, H, W] -> shifted [1, nC, H, W_ext]
    # mask2d_torch carries gradients from DifferentiableMaskWarp
    mask_3d = mask2d_torch.unsqueeze(0).unsqueeze(0).expand(1, nC, H, W)
    mask_3d_shift = shift_torch(mask_3d, step=step)

    # Forward through MST (no torch.no_grad!)
    recon = model(x_init, mask_3d_shift)  # [1, nC, H, W]
    return recon


# ====================================================================
# Component 4 & 5: Optimisation loop + test method
# ====================================================================

def test_cassi_correction_v2(self) -> Dict[str, Any]:
    """UPWMI Algorithm 2: Differentiable Calibration via Backpropagation.

    Replaces Algorithm 1's discrete beam search with PyTorch gradient-based
    optimisation through a frozen MST model.  Mask warping is differentiable
    via F.affine_grid / F.grid_sample, enabling Adam to continuously
    optimise (dx, dy, theta) while phi_d is searched discretely.

    Loss:
        L(psi) = ||y - A_psi(MST(y, mask(psi)))||^2

    Outer loop: discrete search over phi_d (7 candidates)
    Inner loop: Adam on (dx, dy, theta) for ~80 steps per phi_d
    """
    self.log("\n[CASSI] UPWMI Algorithm 2: Differentiable Calibration")
    self.log("=" * 70)

    import torch

    @dataclass
    class OperatorSpec:
        """Operator belief psi = (dx, dy, theta, phi_d)."""
        dx: float = 0.0
        dy: float = 0.0
        theta: float = 0.0
        phi_d: float = 0.0

        def as_dict(self):
            return {"dx": self.dx, "dy": self.dy,
                    "theta_deg": self.theta, "phi_d_deg": self.phi_d}

        def distance(self, other):
            return float(np.sqrt(
                (self.dx - other.dx) ** 2 + (self.dy - other.dy) ** 2
                + (self.theta - other.theta) ** 2
                + (self.phi_d - other.phi_d) ** 2))

        def copy(self):
            return OperatorSpec(self.dx, self.dy, self.theta, self.phi_d)

        def __repr__(self):
            return (f"psi(dx={self.dx:.4f}, dy={self.dy:.4f}, "
                    f"theta={self.theta:.4f}, phi_d={self.phi_d:.4f})")

    # ------------------------------------------------------------------
    # Simulation helpers (same as Algorithm 1)
    # ------------------------------------------------------------------
    class _AffineParams:
        __slots__ = ("dx", "dy", "theta_deg")
        def __init__(self, dx=0.0, dy=0.0, theta_deg=0.0):
            self.dx, self.dy, self.theta_deg = float(dx), float(dy), float(theta_deg)

    def _warp_mask2d(mask2d, affine):
        from scipy.ndimage import affine_transform as _at
        H, W = mask2d.shape
        theta = np.deg2rad(affine.theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        center = np.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)
        M = R.T
        shift = np.array([affine.dy, affine.dx], dtype=np.float32)
        offset = (center - shift) - M @ center
        warped = _at(mask2d.astype(np.float32), matrix=M, offset=offset,
                     output_shape=(H, W), order=1, mode="constant", cval=0.0)
        return np.clip(warped, 0.0, 1.0).astype(np.float32)

    def _make_dispersion_offsets(s_nom, dir_rot_deg):
        theta = np.deg2rad(dir_rot_deg)
        c, s = np.cos(theta), np.sin(theta)
        s_f = s_nom.astype(np.float32)
        return s_f * c, s_f * s

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

    def _simulate_measurement(cube, mask2d_nom, s_nom, psi, alpha, sigma, rng):
        aff = _AffineParams(psi.dx, psi.dy, psi.theta)
        mask2d_used = _warp_mask2d(mask2d_nom, aff)
        y_clean = _cassi_forward(cube, mask2d_used, s_nom, psi.phi_d)
        y_clean = np.maximum(y_clean, 0.0)
        lam = np.clip(alpha * y_clean, 0.0, 1e9)
        y = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
        y += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)
        return y, mask2d_used

    def _gap_tv_recon(y, cube_shape, mask2d, s_nom, dir_rot_deg,
                      max_iter=80, lam=1.0, tv_weight=0.4, tv_iter=5,
                      x_init=None, gauss_sigma=0.5):
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

    # ------------------------------------------------------------------
    # MST model loading (shared with Algorithm 1)
    # ------------------------------------------------------------------
    _mst_cache = [None]

    def _load_mst_model(nC, h, step):
        if _mst_cache[0] is not None:
            return _mst_cache[0]
        from pwm_core.recon.mst import MST

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                    checkpoint = torch.load(str(wp), map_location=device,
                                            weights_only=False)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["state_dict"].items()
                        }
                    else:
                        state_dict = checkpoint
                    self.log(f"  Loaded MST weights from {wp}")
                    break
                except Exception as e:
                    self.log(f"  Failed to load weights from {wp}: {e}")

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
            bot_prefix = "bottleneck.blocks."
            max_bot = -1
            for k in state_dict:
                if k.startswith(bot_prefix):
                    blk_idx = int(k[len(bot_prefix):].split(".")[0])
                    max_bot = max(max_bot, blk_idx)
            if max_bot >= 0:
                inferred.append(max_bot + 1)
            if len(inferred) >= 2:
                num_blocks = inferred
                self.log(f"  MST architecture: stage={len(inferred)-1}, "
                         f"num_blocks={num_blocks}")

        model = MST(
            dim=nC, stage=len(num_blocks) - 1, num_blocks=num_blocks,
            in_channels=nC, out_channels=nC, base_resolution=h, step=step,
        ).to(device)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        else:
            raise RuntimeError("MST: no pretrained weights found")
        model.eval()
        _mst_cache[0] = (model, device)
        return model, device

    def _mst_recon(y, mask2d, cube_shape, step=2):
        """Standard MST recon (no_grad) for baselines."""
        from pwm_core.recon.mst import shift_torch, shift_back_meas_torch
        H, W, nC = cube_shape
        model, device = _load_mst_model(nC, H, step)
        W_ext = W + (nC - 1) * step
        y_mst = np.zeros((H, W_ext), dtype=np.float32)
        hh, ww = min(H, y.shape[0]), min(W_ext, y.shape[1])
        y_mst[:hh, :ww] = y[:hh, :ww]
        mask_3d = np.tile(mask2d[:, :, np.newaxis], (1, 1, nC))
        mask_3d_t = (torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
                     .unsqueeze(0).float().to(device))
        mask_3d_shift = shift_torch(mask_3d_t, step=step)
        meas_t = torch.from_numpy(y_mst.copy()).unsqueeze(0).float().to(device)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
        x_init = x_init / nC * 2
        with torch.no_grad():
            recon = model(x_init, mask_3d_shift)
        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return recon.astype(np.float32)

    # ==================================================================
    # Load Data (same as Algorithm 1)
    # ==================================================================
    cube = None
    mask2d_nom = None
    data_source = "unknown"

    try:
        from scipy.io import loadmat as _loadmat
        pkg_root = Path(__file__).parent.parent
        tsa_search_paths = [
            pkg_root / "datasets" / "TSA_simu_data",
            pkg_root.parent.parent / "datasets" / "TSA_simu_data",
            pkg_root / "data" / "TSA_simu_data",
            Path(__file__).parent / "TSA_simu_data",
        ]
        for data_dir in tsa_search_paths:
            mask_path = data_dir / "mask.mat"
            truth_dir = data_dir / "Truth"
            if mask_path.exists() and truth_dir.exists():
                mask_data = _loadmat(str(mask_path))
                mask2d_nom = mask_data["mask"].astype(np.float32)
                scene_path = truth_dir / "scene03.mat"
                if not scene_path.exists():
                    scene_path = sorted(truth_dir.glob("scene*.mat"))[0]
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
                    data_source = f"TSA ({scene_path.stem})"
                    self.log(f"  Loaded TSA data from {data_dir}")
                    break
    except Exception as e:
        self.log(f"  TSA loading failed: {e}")

    if cube is None or mask2d_nom is None:
        self.log("  TSA_simu_data not found, falling back to KAIST + random mask")
        from pwm_core.data.loaders.kaist import KAISTDataset
        dataset = KAISTDataset(resolution=256, num_bands=28)
        name, cube = next(iter(dataset))
        np.random.seed(42)
        mask2d_nom = (np.random.rand(cube.shape[0], cube.shape[1]) > 0.5).astype(np.float32)
        data_source = f"KAIST ({name})"

    H, W, L = cube.shape
    self.log(f"  Data source: {data_source}")
    self.log(f"  Scene shape: ({H}x{W}x{L})")
    self.log(f"  Mask density: {mask2d_nom.mean():.3f}, "
             f"range: [{mask2d_nom.min():.3f}, {mask2d_nom.max():.3f}]")

    s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)

    param_ranges = {
        'dx_min': -3.0, 'dx_max': 3.0,
        'dy_min': -3.0, 'dy_max': 3.0,
        'theta_min': -1.0, 'theta_max': 1.0,
        'phi_d_min': -0.5, 'phi_d_max': 0.5,
    }
    noise_ranges = {
        'sigma_min': 0.003, 'sigma_max': 0.015,
        'alpha_min': 600.0, 'alpha_max': 2500.0,
    }

    # ==================================================================
    # TRUE parameters
    # ==================================================================
    rng = np.random.default_rng(123)
    true_psi = OperatorSpec(
        dx=float(rng.uniform(param_ranges['dx_min'], param_ranges['dx_max'])),
        dy=float(rng.uniform(param_ranges['dy_min'], param_ranges['dy_max'])),
        theta=float(rng.uniform(param_ranges['theta_min'], param_ranges['theta_max'])),
        phi_d=float(rng.uniform(param_ranges['phi_d_min'], param_ranges['phi_d_max'])),
    )
    true_alpha = float(rng.uniform(noise_ranges['alpha_min'], noise_ranges['alpha_max']))
    true_sigma = float(rng.uniform(noise_ranges['sigma_min'], noise_ranges['sigma_max']))
    self.log(f"  TRUE operator: {true_psi}")
    self.log(f"  TRUE noise:    alpha={true_alpha:.0f}, sigma={true_sigma:.4f}")

    # ==================================================================
    # Simulate measurement
    # ==================================================================
    y, mask2d_true = _simulate_measurement(
        cube, mask2d_nom, s_nom, true_psi, true_alpha, true_sigma, rng)
    self.log(f"  Measurement shape: {y.shape}")

    # ==================================================================
    # Baselines
    # ==================================================================
    self.log("\n  [Baseline] Reconstructing with nominal (wrong) params...")
    mask_wrong = _warp_mask2d(mask2d_nom, _AffineParams(0, 0, 0))
    x_wrong = _gap_tv_recon(y, (H, W, L), mask_wrong, s_nom, 0.0, max_iter=80)
    psnr_wrong = compute_psnr(x_wrong, cube)
    self.log(f"  PSNR (GAP-TV wrong):  {psnr_wrong:.2f} dB")

    self.log("  [Baseline] Reconstructing with oracle (true) params...")
    x_oracle = _gap_tv_recon(y, (H, W, L), mask2d_true, s_nom,
                             true_psi.phi_d, max_iter=80)
    psnr_oracle = compute_psnr(x_oracle, cube)
    self.log(f"  PSNR (GAP-TV oracle): {psnr_oracle:.2f} dB")

    psnr_mst_wrong = None
    psnr_mst_oracle = None
    try:
        self.log("\n  [Baseline] MST with nominal (wrong) mask...")
        x_mst_wrong = _mst_recon(y, mask_wrong, (H, W, L), step=2)
        psnr_mst_wrong = compute_psnr(x_mst_wrong, cube)
        self.log(f"  PSNR (MST wrong):  {psnr_mst_wrong:.2f} dB")

        self.log("  [Baseline] MST with oracle (true) mask...")
        x_mst_oracle = _mst_recon(y, mask2d_true, (H, W, L), step=2)
        psnr_mst_oracle = compute_psnr(x_mst_oracle, cube)
        self.log(f"  PSNR (MST oracle): {psnr_mst_oracle:.2f} dB")
    except Exception as e:
        self.log(f"  MST baselines unavailable: {e}")

    # ==================================================================
    # Algorithm 2: Differentiable Calibration
    # ==================================================================
    self.log("\n  === Algorithm 2: Differentiable Calibration ===")
    self.log("  (sub-pixel forward; phi_d candidates + joint optimisation)")
    t_total = _time.time()

    model, device = _load_mst_model(L, H, step=2)
    # Freeze MST parameters
    for p in model.parameters():
        p.requires_grad_(False)

    # phi_d discrete candidates (for initialisation coverage)
    phi_d_candidates = np.linspace(
        param_ranges['phi_d_min'], param_ranges['phi_d_max'], 9)

    # Optimisation hyperparameters
    n_steps = 200
    lr_init = 0.03
    lr_final = 0.001
    grad_clip = 1.0
    n_starts = 4  # geo multi-starts per phi_d candidate

    best_global_loss = float('inf')
    best_global_params = OperatorSpec()
    start_results = []

    cassi_fwd = DifferentiableCassiForward(s_nom, H, W, L).to(device)
    y_t = torch.from_numpy(y.copy()).unsqueeze(0).float().to(device)

    self.log(f"  phi_d candidates: {len(phi_d_candidates)}, "
             f"steps/candidate: {n_steps}, geo starts: {n_starts}")

    for phi_idx, phi_d_init in enumerate(phi_d_candidates):
        phi_d_init = float(phi_d_init)
        best_start_loss = float('inf')
        best_start_params = None

        for start_idx in range(n_starts):
            # Random geo initialisation (start 0 = origin)
            if start_idx == 0:
                dx0, dy0, th0 = 0.0, 0.0, 0.0
            else:
                dx0 = float(np.random.uniform(-1.5, 1.5))
                dy0 = float(np.random.uniform(-1.5, 1.5))
                th0 = float(np.random.uniform(-0.5, 0.5))

            warp = DifferentiableMaskWarp(
                mask2d_nom, dx_init=dx0, dy_init=dy0, theta_init=th0
            ).to(device)
            # phi_d starts at candidate value, then optimised jointly
            phi_d_param = torch.nn.Parameter(
                torch.tensor(phi_d_init, dtype=torch.float32, device=device))

            all_params = list(warp.parameters()) + [phi_d_param]
            optimizer = torch.optim.Adam(all_params, lr=lr_init)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_steps, eta_min=lr_final)

            final_loss_val = float('inf')

            for step_i in range(n_steps):
                optimizer.zero_grad()

                # Forward: warp mask -> MST recon -> sub-pixel CASSI fwd -> loss
                mask_warped = warp()  # [H, W], differentiable
                recon = _mst_recon_differentiable(
                    model, device, y, mask_warped, (H, W, L), step=2)

                y_pred = cassi_fwd(recon, mask_warped, phi_d_param)

                # Match sizes
                hh = min(y_t.shape[1], y_pred.shape[1])
                ww = min(y_t.shape[2], y_pred.shape[2])
                loss = torch.mean(
                    (y_t[:, :hh, :ww] - y_pred[:, :hh, :ww]) ** 2)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)

                optimizer.step()
                scheduler.step()

                # Clamp parameters
                with torch.no_grad():
                    warp.dx.clamp_(-3.0, 3.0)
                    warp.dy.clamp_(-3.0, 3.0)
                    warp.theta_deg.clamp_(-1.0, 1.0)
                    phi_d_param.clamp_(-0.5, 0.5)

                final_loss_val = loss.item()

                # Verify gradient flow on first step
                if step_i == 0 and phi_idx == 0 and start_idx == 0:
                    dx_g = warp.dx.grad
                    phi_g = phi_d_param.grad
                    self.log(f"  Gradient flow check: "
                             f"dx.grad={dx_g}, "
                             f"phi_d.grad={phi_g}")

            if final_loss_val < best_start_loss:
                best_start_loss = final_loss_val
                best_start_params = (
                    warp.dx.item(), warp.dy.item(),
                    warp.theta_deg.item(), phi_d_param.item())

        dx_b, dy_b, th_b, phi_b = best_start_params
        start_results.append({
            'phi_d_init': phi_d_init,
            'loss': best_start_loss,
            'dx': dx_b, 'dy': dy_b,
            'theta': th_b, 'phi_d': phi_b,
        })

        if best_start_loss < best_global_loss:
            best_global_loss = best_start_loss
            best_global_params = OperatorSpec(
                dx=dx_b, dy=dy_b, theta=th_b, phi_d=phi_b)

        self.log(f"  phi_d_init={phi_d_init:+.3f}: loss={best_start_loss:.6f}, "
                 f"dx={dx_b:.4f}, dy={dy_b:.4f}, "
                 f"theta={th_b:.4f}, phi_d={phi_b:.4f}")

    loop_time = _time.time() - t_total
    psi_final = best_global_params
    self.log(f"\n  Calibrated operator: {psi_final}")
    self.log(f"  Optimisation time: {loop_time:.1f}s")

    # ==================================================================
    # Final reconstruction with calibrated mask
    # ==================================================================
    self.log("\n  Step 3: Final MST reconstruction with calibrated mask")
    t_final = _time.time()
    aff_calib = _AffineParams(psi_final.dx, psi_final.dy, psi_final.theta)
    mask_calib = _warp_mask2d(mask2d_nom, aff_calib)
    x_final = _mst_recon(y, mask_calib, (H, W, L), step=2)
    final_time = _time.time() - t_final
    psnr_corrected = compute_psnr(x_final, cube)

    # GAP-TV with calibrated mask
    self.log("  [Baseline] GAP-TV with calibrated mask...")
    x_gaptv_calib = _gap_tv_recon(y, (H, W, L), mask_calib, s_nom,
                                   psi_final.phi_d, max_iter=120)
    psnr_gaptv_calib = compute_psnr(x_gaptv_calib, cube)
    self.log(f"  PSNR (GAP-TV calibrated): {psnr_gaptv_calib:.2f} dB")

    # ==================================================================
    # Output artifacts
    # ==================================================================
    calib = psi_final.as_dict()

    diagnosis = {
        'dx_error': abs(psi_final.dx - true_psi.dx),
        'dy_error': abs(psi_final.dy - true_psi.dy),
        'theta_error': abs(psi_final.theta - true_psi.theta),
        'phi_d_error': abs(psi_final.phi_d - true_psi.phi_d),
    }

    belief_state = {
        'algorithm': 'UPWMI_Algorithm2_DifferentiableCalibration',
        'multi_start_results': start_results,
        'n_steps': n_steps,
        'n_starts': n_starts,
        'lr_init': lr_init,
        'lr_final': lr_final,
        'total_time_s': float(loop_time),
    }

    report = {
        'diagnosis': diagnosis,
        'psnr_wrong': float(psnr_wrong),
        'psnr_corrected': float(psnr_corrected),
        'psnr_oracle': float(psnr_oracle),
        'psnr_mst_wrong': float(psnr_mst_wrong) if psnr_mst_wrong is not None else None,
        'psnr_mst_oracle': float(psnr_mst_oracle) if psnr_mst_oracle is not None else None,
        'psnr_gaptv_calibrated': float(psnr_gaptv_calib),
        'final_recon_method': 'MST',
        'improvement_db': float(psnr_corrected - psnr_wrong),
    }

    import json as _json
    output_dir = Path(__file__).parent / "results" / "cassi_upwmi_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in [("OperatorSpec_calib.json", calib),
                        ("BeliefState.json", belief_state),
                        ("Report.json", report)]:
        with open(output_dir / fname, "w") as f:
            _json.dump(data, f, indent=2, default=str)

    # ==================================================================
    # Summary
    # ==================================================================
    self.log(f"\n  {'=' * 60}")
    self.log(f"  RESULTS SUMMARY (Algorithm 2 - Differentiable)")
    self.log(f"  {'=' * 60}")
    self.log(f"  True operator:       {true_psi}")
    self.log(f"  Calibrated operator: {psi_final}")
    self.log(f"  Errors: dx={diagnosis['dx_error']:.4f}, "
             f"dy={diagnosis['dy_error']:.4f}, "
             f"theta={diagnosis['theta_error']:.4f} deg, "
             f"phi_d={diagnosis['phi_d_error']:.4f} deg")
    self.log(f"  True noise:  alpha={true_alpha:.0f}, sigma={true_sigma:.4f}")
    self.log(f"  GAP-TV wrong:       {psnr_wrong:.2f} dB")
    self.log(f"  GAP-TV oracle:      {psnr_oracle:.2f} dB")
    self.log(f"  GAP-TV calibrated:  {psnr_gaptv_calib:.2f} dB")
    if psnr_mst_wrong is not None:
        self.log(f"  MST wrong:          {psnr_mst_wrong:.2f} dB")
    if psnr_mst_oracle is not None:
        self.log(f"  MST oracle:         {psnr_mst_oracle:.2f} dB")
    self.log(f"  MST calibrated:     {psnr_corrected:.2f} dB "
             f"(+{psnr_corrected - psnr_wrong:.2f} dB from wrong)")
    self.log(f"  Total time:  {loop_time:.1f}s (optim) + "
             f"{final_time:.1f}s (final recon)")
    self.log(f"  Artifacts:   {output_dir}")

    result = {
        "modality": "cassi_v2",
        "algorithm": "UPWMI_Algorithm2_DifferentiableCalibration",
        "mismatch_param": ["mask_geo", "disp_dir_rot"],
        "true_value": {
            "geo": {"dx": true_psi.dx, "dy": true_psi.dy,
                    "theta_deg": true_psi.theta},
            "disp": {"dir_rot_deg": true_psi.phi_d},
            "noise": {"alpha": true_alpha, "sigma": true_sigma},
        },
        "wrong_value": {
            "geo": {"dx": 0.0, "dy": 0.0, "theta_deg": 0.0},
            "disp": {"dir_rot_deg": 0.0},
        },
        "calibrated_value": calib,
        "oracle_psnr": float(psnr_oracle),
        "psnr_without_correction": float(psnr_wrong),
        "psnr_with_correction": float(psnr_corrected),
        "improvement_db": float(psnr_corrected - psnr_wrong),
        "final_recon_method": "MST",
        "psnr_mst_wrong": float(psnr_mst_wrong) if psnr_mst_wrong is not None else None,
        "psnr_mst_oracle": float(psnr_mst_oracle) if psnr_mst_oracle is not None else None,
        "psnr_gaptv_calibrated": float(psnr_gaptv_calib),
        "data_source": data_source,
        "belief_state": belief_state,
        "report": report,
    }

    return result
