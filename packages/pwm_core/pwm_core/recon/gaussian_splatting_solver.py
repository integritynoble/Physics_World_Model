"""3D Gaussian Splatting for Real-Time Radiance Field Rendering.

Differentiable Gaussian splatting for novel view synthesis.

References:
- Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance
  Field Rendering", SIGGRAPH 2023.

Benchmark:
- ~34 dB on Mip-NeRF 360 scenes
- Real-time rendering (100+ FPS)
- Params: 248 bytes per splat, typically 0.5-3M splats
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_PKG_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _PKG_ROOT / "weights" / "gaussian_splatting"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "Gaussian splatting requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Spherical harmonics helpers
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
_SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.4453057213202769,
    -0.5900435899266435,
]


def eval_sh(deg: int, sh_coeffs: "torch.Tensor", dirs: "torch.Tensor") -> "torch.Tensor":
    """Evaluate spherical harmonics at given directions.

    Args:
        deg: Maximum SH degree (0-3).
        sh_coeffs: SH coefficients (N, C, 3) where C = (deg+1)^2.
        dirs: Unit direction vectors (N, 3).

    Returns:
        RGB colors (N, 3).
    """
    result = _SH_C0 * sh_coeffs[:, 0]

    if deg < 1:
        return result

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = (
        result
        - _SH_C1 * y * sh_coeffs[:, 1]
        + _SH_C1 * z * sh_coeffs[:, 2]
        - _SH_C1 * x * sh_coeffs[:, 3]
    )

    if deg < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + _SH_C2[0] * xy * sh_coeffs[:, 4]
        + _SH_C2[1] * yz * sh_coeffs[:, 5]
        + _SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6]
        + _SH_C2[3] * xz * sh_coeffs[:, 7]
        + _SH_C2[4] * (xx - yy) * sh_coeffs[:, 8]
    )

    if deg < 3:
        return result

    result = (
        result
        + _SH_C3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9]
        + _SH_C3[1] * xy * z * sh_coeffs[:, 10]
        + _SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11]
        + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12]
        + _SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13]
        + _SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14]
        + _SH_C3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]
    )

    return result


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: "torch.Tensor") -> "torch.Tensor":
    """Convert unit quaternions to rotation matrices.

    Args:
        q: Quaternions (N, 4) as (w, x, y, z).

    Returns:
        Rotation matrices (N, 3, 3).
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)

    return R


def build_covariance_3d(
    scales: "torch.Tensor", rotations: "torch.Tensor"
) -> "torch.Tensor":
    """Build 3D covariance matrices from scales and rotations.

    Sigma = R @ S @ S^T @ R^T

    Args:
        scales: Log-scale vectors (N, 3).
        rotations: Quaternions (N, 4).

    Returns:
        Covariance matrices (N, 3, 3).
    """
    S = torch.diag_embed(torch.exp(scales))  # (N, 3, 3)
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
    M = R @ S  # (N, 3, 3)
    cov = M @ M.transpose(-1, -2)  # (N, 3, 3)
    return cov


# ---------------------------------------------------------------------------
# GaussianModel
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class GaussianModel(nn.Module):
        """Stores and manages 3D Gaussian splat parameters.

        Each Gaussian is parameterized by:
        - position (xyz center)
        - scale (log-space, per-axis)
        - rotation (unit quaternion)
        - opacity (sigmoid-space)
        - SH coefficients (for view-dependent color)

        Args:
            n_points: Number of initial Gaussian splats.
            sh_degree: Spherical harmonics degree (0-3).
        """

        def __init__(self, n_points: int = 1000, sh_degree: int = 3):
            super().__init__()
            self.sh_degree = sh_degree
            n_sh = (sh_degree + 1) ** 2

            self.positions = nn.Parameter(torch.randn(n_points, 3) * 0.5)
            self.scales = nn.Parameter(torch.full((n_points, 3), -3.0))
            self.rotations = nn.Parameter(
                F.normalize(torch.randn(n_points, 4), p=2, dim=-1)
            )
            self.opacities = nn.Parameter(torch.full((n_points, 1), -1.0))
            self.sh_coeffs = nn.Parameter(torch.zeros(n_points, n_sh, 3))

            # Initialize DC component to gray
            with torch.no_grad():
                self.sh_coeffs[:, 0, :] = 0.5 / _SH_C0

        @property
        def n_gaussians(self) -> int:
            return self.positions.shape[0]

        def get_scales(self) -> "torch.Tensor":
            """Activated scales (positive)."""
            return torch.exp(self.scales)

        def get_opacities(self) -> "torch.Tensor":
            """Activated opacities in [0, 1]."""
            return torch.sigmoid(self.opacities)

        def get_colors(self, viewdirs: "torch.Tensor") -> "torch.Tensor":
            """Compute view-dependent RGB from SH coefficients.

            Args:
                viewdirs: Unit viewing directions (N, 3).

            Returns:
                RGB colors (N, 3) clamped to [0, 1].
            """
            raw = eval_sh(self.sh_degree, self.sh_coeffs, viewdirs)
            return torch.clamp(raw + 0.5, 0.0, 1.0)

        def get_covariance_3d(self) -> "torch.Tensor":
            """Build 3D covariance matrices (N, 3, 3)."""
            return build_covariance_3d(self.scales, self.rotations)

        def init_from_points(
            self,
            points: "torch.Tensor",
            colors: Optional["torch.Tensor"] = None,
        ):
            """Re-initialize from a point cloud.

            Args:
                points: Point positions (M, 3).
                colors: Optional RGB colors (M, 3) in [0, 1].
            """
            M = points.shape[0]
            device = self.positions.device

            self.positions = nn.Parameter(points.to(device).float())
            self.scales = nn.Parameter(torch.full((M, 3), -3.0, device=device))
            self.rotations = nn.Parameter(
                F.normalize(torch.randn(M, 4, device=device), p=2, dim=-1)
            )
            self.opacities = nn.Parameter(torch.full((M, 1), -1.0, device=device))

            n_sh = (self.sh_degree + 1) ** 2
            self.sh_coeffs = nn.Parameter(torch.zeros(M, n_sh, 3, device=device))

            if colors is not None:
                with torch.no_grad():
                    self.sh_coeffs[:, 0, :] = (colors.to(device).float() - 0.5) / _SH_C0
            else:
                with torch.no_grad():
                    self.sh_coeffs[:, 0, :] = 0.5 / _SH_C0


# ---------------------------------------------------------------------------
# Projection: 3D Gaussians -> 2D screen-space
# ---------------------------------------------------------------------------

def project_gaussians(
    positions: "torch.Tensor",
    covs_3d: "torch.Tensor",
    viewmat: "torch.Tensor",
    projmat: "torch.Tensor",
    W: int,
    H: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Project 3D Gaussians to 2D screen space.

    Args:
        positions: Gaussian centers (N, 3).
        covs_3d: 3D covariance matrices (N, 3, 3).
        viewmat: World-to-camera matrix (4, 4).
        projmat: Full projection matrix (4, 4).
        W: Image width.
        H: Image height.

    Returns:
        means2d: 2D pixel means (N, 2).
        covs2d: 2D covariance matrices (N, 2, 2).
        depths: Depth values (N,).
        visible: Visibility mask (N,) boolean.
    """
    N = positions.shape[0]

    # Transform to camera coordinates
    ones = torch.ones(N, 1, device=positions.device, dtype=positions.dtype)
    pos_h = torch.cat([positions, ones], dim=-1)  # (N, 4)
    pos_cam = (viewmat @ pos_h.T).T  # (N, 4)

    depths = pos_cam[:, 2]

    # Frustum culling: keep points with positive depth
    visible = depths > 0.1

    # Project to clip space
    pos_clip = (projmat @ pos_h.T).T  # (N, 4)
    w_clip = pos_clip[:, 3:4].clamp(min=1e-6)
    ndc = pos_clip[:, :2] / w_clip  # (N, 2) in [-1, 1]

    # NDC to pixel coordinates
    means2d = torch.zeros(N, 2, device=positions.device, dtype=positions.dtype)
    means2d[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * W
    means2d[:, 1] = (ndc[:, 1] + 1.0) * 0.5 * H

    # Screen culling
    margin = 100
    visible = visible & (means2d[:, 0] > -margin) & (means2d[:, 0] < W + margin)
    visible = visible & (means2d[:, 1] > -margin) & (means2d[:, 1] < H + margin)

    # Compute 2D covariance via Jacobian of perspective projection
    # J = d(pixel) / d(cam) for the projection
    fx = projmat[0, 0] * W * 0.5
    fy = projmat[1, 1] * H * 0.5

    tz = depths.clamp(min=1e-6)
    tx = pos_cam[:, 0]
    ty = pos_cam[:, 1]

    J = torch.zeros(N, 2, 3, device=positions.device, dtype=positions.dtype)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / (tz * tz)

    # Rotate 3D covariance to camera frame
    R_cam = viewmat[:3, :3]  # (3, 3)
    cov_cam = R_cam @ covs_3d @ R_cam.T  # (N, 3, 3) broadcast

    # Project to 2D: Sigma_2d = J @ Sigma_cam @ J^T
    covs2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)

    # Add small regularization for numerical stability
    covs2d[:, 0, 0] = covs2d[:, 0, 0] + 0.3
    covs2d[:, 1, 1] = covs2d[:, 1, 1] + 0.3

    return means2d, covs2d, depths, visible


# ---------------------------------------------------------------------------
# Rasterization: alpha-blended splatting
# ---------------------------------------------------------------------------

def rasterize_gaussians(
    means2d: "torch.Tensor",
    covs2d: "torch.Tensor",
    colors: "torch.Tensor",
    opacities: "torch.Tensor",
    depths: "torch.Tensor",
    visible: "torch.Tensor",
    H: int,
    W: int,
    tile_size: int = 16,
) -> "torch.Tensor":
    """Rasterize 2D Gaussians to an image via alpha blending.

    Uses a simplified tile-based approach: for each pixel, evaluate
    all visible Gaussians sorted by depth and accumulate with
    front-to-back alpha compositing.

    Args:
        means2d: 2D pixel means (N, 2).
        covs2d: 2D covariance matrices (N, 2, 2).
        colors: Per-Gaussian RGB (N, 3).
        opacities: Per-Gaussian opacity (N, 1).
        depths: Depth values for sorting (N,).
        visible: Visibility mask (N,).
        H: Image height.
        W: Image width.
        tile_size: Tile size for spatial bucketing.

    Returns:
        Rendered image (H, W, 3).
    """
    device = means2d.device
    dtype = means2d.dtype

    # Filter to visible Gaussians
    vis_idx = torch.where(visible)[0]
    if vis_idx.numel() == 0:
        return torch.zeros(H, W, 3, device=device, dtype=dtype)

    v_means = means2d[vis_idx]
    v_covs = covs2d[vis_idx]
    v_colors = colors[vis_idx]
    v_opacities = opacities[vis_idx].squeeze(-1)
    v_depths = depths[vis_idx]

    # Sort by depth (front to back)
    sort_idx = torch.argsort(v_depths)
    v_means = v_means[sort_idx]
    v_covs = v_covs[sort_idx]
    v_colors = v_colors[sort_idx]
    v_opacities = v_opacities[sort_idx]

    M = v_means.shape[0]

    # Pre-compute inverse covariance and determinant for Gaussian evaluation
    det = v_covs[:, 0, 0] * v_covs[:, 1, 1] - v_covs[:, 0, 1] * v_covs[:, 1, 0]
    det = det.clamp(min=1e-10)

    inv_cov = torch.zeros_like(v_covs)
    inv_cov[:, 0, 0] = v_covs[:, 1, 1] / det
    inv_cov[:, 0, 1] = -v_covs[:, 0, 1] / det
    inv_cov[:, 1, 0] = -v_covs[:, 1, 0] / det
    inv_cov[:, 1, 1] = v_covs[:, 0, 0] / det

    # Compute per-Gaussian bounding radius (3-sigma ellipse)
    eigenvalues = 0.5 * (
        v_covs[:, 0, 0] + v_covs[:, 1, 1]
        + torch.sqrt(
            (v_covs[:, 0, 0] - v_covs[:, 1, 1]) ** 2
            + 4 * v_covs[:, 0, 1] ** 2
            + 1e-8
        )
    )
    radii = torch.ceil(3.0 * torch.sqrt(eigenvalues.clamp(min=1e-8)))
    radii = radii.clamp(max=max(H, W) * 0.5)

    # Tile-based rasterization
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size

    image = torch.zeros(H, W, 3, device=device, dtype=dtype)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Tile pixel bounds
            px_min = tx * tile_size
            py_min = ty * tile_size
            px_max = min(px_min + tile_size, W)
            py_max = min(py_min + tile_size, H)

            tile_cx = (px_min + px_max) * 0.5
            tile_cy = (py_min + py_max) * 0.5
            tile_r = tile_size * 0.7071  # half-diagonal

            # Find Gaussians that overlap this tile
            dist_to_tile = torch.sqrt(
                (v_means[:, 0] - tile_cx) ** 2
                + (v_means[:, 1] - tile_cy) ** 2
            )
            tile_mask = dist_to_tile < (radii + tile_r)
            tile_idx = torch.where(tile_mask)[0]

            if tile_idx.numel() == 0:
                continue

            t_means = v_means[tile_idx]
            t_inv_cov = inv_cov[tile_idx]
            t_colors = v_colors[tile_idx]
            t_opacities = v_opacities[tile_idx]

            # Generate pixel grid for this tile
            py_coords = torch.arange(py_min, py_max, device=device, dtype=dtype)
            px_coords = torch.arange(px_min, px_max, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(py_coords, px_coords, indexing="ij")
            # (tile_h, tile_w)
            tile_h = py_max - py_min
            tile_w = px_max - px_min

            # Flatten pixel grid: (P, 2)
            pixels = torch.stack([
                grid_x.reshape(-1) + 0.5,
                grid_y.reshape(-1) + 0.5,
            ], dim=-1)
            P = pixels.shape[0]
            K = tile_idx.shape[0]

            # Compute displacement: (P, K, 2)
            dx = pixels[:, None, :] - t_means[None, :, :]  # (P, K, 2)

            # Evaluate Gaussian: exp(-0.5 * d^T @ inv_cov @ d)
            # (P, K, 1, 2) @ (K, 2, 2) -> (P, K, 1, 2)
            Mdx = (dx.unsqueeze(2) @ t_inv_cov.unsqueeze(0)).squeeze(2)  # (P, K, 2)
            exponent = -0.5 * (dx * Mdx).sum(dim=-1)  # (P, K)

            # Clamp exponent for numerical stability
            gauss_weight = torch.exp(exponent.clamp(min=-20.0, max=0.0))

            # Alpha per Gaussian per pixel
            alpha = (t_opacities[None, :] * gauss_weight).clamp(max=0.99)  # (P, K)

            # Front-to-back alpha compositing
            # T_i = prod(1 - alpha_j, j < i)
            one_minus_alpha = 1.0 - alpha  # (P, K)
            # Exclusive cumulative product
            T = torch.ones(P, K, device=device, dtype=dtype)
            if K > 1:
                T[:, 1:] = torch.cumprod(one_minus_alpha[:, :-1], dim=1)

            # Weight = T_i * alpha_i
            weight = T * alpha  # (P, K)

            # Accumulate color: (P, 3)
            pixel_colors = (weight.unsqueeze(-1) * t_colors[None, :, :]).sum(dim=1)

            # Write to image
            image[py_min:py_max, px_min:px_max, :] = pixel_colors.reshape(
                tile_h, tile_w, 3
            )

    return image.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Adaptive density control (densification / pruning)
# ---------------------------------------------------------------------------

def _densify_and_prune(
    model: "GaussianModel",
    grad_accum: "torch.Tensor",
    grad_count: "torch.Tensor",
    grad_threshold: float = 0.0002,
    min_opacity: float = 0.005,
    max_screen_size: float = 20.0,
    scene_extent: float = 1.0,
) -> "GaussianModel":
    """Adaptive density control for Gaussian splatting.

    - Clone small Gaussians with large gradients (under-reconstruction).
    - Split large Gaussians with large gradients (over-reconstruction).
    - Prune low-opacity or oversized Gaussians.

    Args:
        model: Current Gaussian model.
        grad_accum: Accumulated position gradients (N, 3).
        grad_count: Number of gradient accumulations per Gaussian (N,).
        grad_threshold: Gradient magnitude threshold for densification.
        min_opacity: Opacity threshold for pruning.
        max_screen_size: Maximum screen-space footprint for pruning.
        scene_extent: Scene bounding sphere radius.

    Returns:
        Updated model (potentially with different number of Gaussians).
    """
    device = model.positions.device
    N = model.n_gaussians

    # Average gradient magnitude
    avg_grad = (grad_accum / grad_count.clamp(min=1)).norm(dim=-1)

    # Identify Gaussians that need densification
    needs_densify = avg_grad > grad_threshold

    # Split criterion: large scale
    scales = model.get_scales()
    max_scale = scales.max(dim=-1).values
    needs_split = needs_densify & (max_scale > 0.01 * scene_extent)

    # Clone criterion: small scale
    needs_clone = needs_densify & ~needs_split

    # Prune criterion: low opacity or too large
    opacities = model.get_opacities().squeeze(-1)
    needs_prune = opacities < min_opacity

    # Collect new Gaussians from cloning
    clone_idx = torch.where(needs_clone)[0]
    split_idx = torch.where(needs_split)[0]
    keep_mask = ~needs_prune

    # Build new parameter tensors
    new_positions = [model.positions.data[keep_mask]]
    new_scales = [model.scales.data[keep_mask]]
    new_rotations = [model.rotations.data[keep_mask]]
    new_opacities = [model.opacities.data[keep_mask]]
    new_sh = [model.sh_coeffs.data[keep_mask]]

    # Clone: duplicate at same position with small perturbation
    if clone_idx.numel() > 0:
        new_positions.append(
            model.positions.data[clone_idx]
            + torch.randn_like(model.positions.data[clone_idx]) * 0.001
        )
        new_scales.append(model.scales.data[clone_idx])
        new_rotations.append(model.rotations.data[clone_idx])
        new_opacities.append(model.opacities.data[clone_idx])
        new_sh.append(model.sh_coeffs.data[clone_idx])

    # Split: replace with two smaller Gaussians
    if split_idx.numel() > 0:
        for _ in range(2):
            noise = torch.randn_like(model.positions.data[split_idx]) * 0.01
            new_positions.append(model.positions.data[split_idx] + noise)
            new_scales.append(model.scales.data[split_idx] - 0.693)  # halve scale
            new_rotations.append(model.rotations.data[split_idx])
            new_opacities.append(model.opacities.data[split_idx])
            new_sh.append(model.sh_coeffs.data[split_idx])

    # Reconstruct model
    all_pos = torch.cat(new_positions, dim=0)
    model.positions = nn.Parameter(all_pos)
    model.scales = nn.Parameter(torch.cat(new_scales, dim=0))
    model.rotations = nn.Parameter(torch.cat(new_rotations, dim=0))
    model.opacities = nn.Parameter(torch.cat(new_opacities, dim=0))
    model.sh_coeffs = nn.Parameter(torch.cat(new_sh, dim=0))

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def gs_train(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    intrinsics: np.ndarray,
    n_init_points: int = 1000,
    iters: int = 3000,
    lr_position: float = 1.6e-4,
    lr_scale: float = 5e-3,
    lr_rotation: float = 1e-3,
    lr_opacity: float = 5e-2,
    lr_sh: float = 2.5e-3,
    sh_degree: int = 3,
    densify_from: int = 500,
    densify_until: int = 2500,
    densify_every: int = 100,
    opacity_reset_every: int = 3000,
    init_points: Optional[np.ndarray] = None,
    init_colors: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple["GaussianModel", Dict[str, Any]]:
    """Train a 3D Gaussian splatting model.

    Args:
        images: List of training images, each (H, W, 3) float32 in [0, 1].
        poses: List of camera-to-world matrices (4, 4).
        intrinsics: Camera intrinsic matrix (3, 3).
        n_init_points: Number of random initial points (if init_points is None).
        iters: Number of optimization iterations.
        lr_position: Learning rate for positions.
        lr_scale: Learning rate for scales.
        lr_rotation: Learning rate for rotations.
        lr_opacity: Learning rate for opacities.
        lr_sh: Learning rate for SH coefficients.
        sh_degree: Maximum SH degree.
        densify_from: Start densification at this iteration.
        densify_until: Stop densification at this iteration.
        densify_every: Densification interval.
        opacity_reset_every: Opacity reset interval.
        init_points: Optional initial point cloud (M, 3).
        init_colors: Optional initial point colors (M, 3) in [0, 1].
        device: Torch device string.

    Returns:
        Tuple of (trained GaussianModel, training_info dict).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_images = len(images)
    H, W = images[0].shape[:2]

    # Build projection matrices
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    near, far = 0.01, 100.0

    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = 2.0 * fy / H
    proj[0, 2] = 2.0 * cx / W - 1.0
    proj[1, 2] = 2.0 * cy / H - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0

    # Convert poses to view matrices (world-to-camera)
    view_mats = []
    for pose in poses:
        c2w = torch.tensor(pose, device=device, dtype=torch.float32)
        w2c = torch.inverse(c2w)
        view_mats.append(w2c)

    # Convert images to tensors
    image_tensors = [
        torch.tensor(img, device=device, dtype=torch.float32)
        for img in images
    ]

    # Initialize model
    model = GaussianModel(n_points=n_init_points, sh_degree=sh_degree).to(device)

    if init_points is not None:
        pts = torch.tensor(init_points, dtype=torch.float32)
        cols = None
        if init_colors is not None:
            cols = torch.tensor(init_colors, dtype=torch.float32)
        model.init_from_points(pts, cols)

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [model.positions], "lr": lr_position},
        {"params": [model.scales], "lr": lr_scale},
        {"params": [model.rotations], "lr": lr_rotation},
        {"params": [model.opacities], "lr": lr_opacity},
        {"params": [model.sh_coeffs], "lr": lr_sh},
    ])

    # Gradient accumulators for densification
    grad_accum = torch.zeros(model.n_gaussians, 3, device=device)
    grad_count = torch.zeros(model.n_gaussians, device=device)

    # Training info
    losses = []
    best_loss = float("inf")

    for it in range(iters):
        # Pick a random training view
        idx = np.random.randint(n_images)
        gt_image = image_tensors[idx]
        viewmat = view_mats[idx]
        projmat = proj @ viewmat

        # Forward: build covariances, project, compute colors, rasterize
        covs_3d = model.get_covariance_3d()
        means2d, covs2d, depths_val, visible = project_gaussians(
            model.positions, covs_3d, viewmat, projmat, W, H
        )

        # View directions for SH evaluation
        cam_pos = torch.inverse(viewmat)[:3, 3]
        viewdirs = F.normalize(model.positions - cam_pos[None, :], dim=-1)
        colors = model.get_colors(viewdirs)
        opacities_act = model.get_opacities()

        rendered = rasterize_gaussians(
            means2d, covs2d, colors, opacities_act, depths_val, visible, H, W
        )

        # L1 + SSIM-like loss
        l1_loss = (rendered - gt_image).abs().mean()
        loss = 0.8 * l1_loss + 0.2 * (1.0 - _ssim_simple(rendered, gt_image))

        optimizer.zero_grad()
        loss.backward()

        # Accumulate gradients for densification
        if (
            model.positions.grad is not None
            and densify_from <= it < densify_until
        ):
            N_cur = model.n_gaussians
            if grad_accum.shape[0] == N_cur:
                grad_accum += model.positions.grad.data.abs()
                grad_count += 1

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        # Densification
        if (
            densify_from <= it < densify_until
            and it > 0
            and it % densify_every == 0
        ):
            model = _densify_and_prune(model, grad_accum, grad_count)

            # Reset optimizer and accumulators
            optimizer = torch.optim.Adam([
                {"params": [model.positions], "lr": lr_position},
                {"params": [model.scales], "lr": lr_scale},
                {"params": [model.rotations], "lr": lr_rotation},
                {"params": [model.opacities], "lr": lr_opacity},
                {"params": [model.sh_coeffs], "lr": lr_sh},
            ])
            grad_accum = torch.zeros(model.n_gaussians, 3, device=device)
            grad_count = torch.zeros(model.n_gaussians, device=device)

        # Opacity reset
        if it > 0 and it % opacity_reset_every == 0:
            with torch.no_grad():
                model.opacities.fill_(-2.0)

    info = {
        "solver": "gaussian_splatting",
        "iters": iters,
        "final_loss": float(losses[-1]) if losses else float("inf"),
        "best_loss": float(best_loss),
        "n_gaussians": model.n_gaussians,
        "sh_degree": sh_degree,
    }

    return model, info


def _ssim_simple(
    img1: "torch.Tensor", img2: "torch.Tensor", window_size: int = 11
) -> "torch.Tensor":
    """Simplified structural similarity for training loss.

    Computes mean SSIM over the image using a uniform averaging window.

    Args:
        img1: Image (H, W, 3).
        img2: Image (H, W, 3).
        window_size: Averaging window size.

    Returns:
        Scalar SSIM value.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Move channels first for conv2d: (1, 3, H, W)
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)

    # Uniform averaging kernel
    pad = window_size // 2
    kernel = torch.ones(
        1, 1, window_size, window_size, device=x.device, dtype=x.dtype
    ) / (window_size * window_size)

    channels = x.shape[1]
    ssim_vals = []
    for c in range(channels):
        xc = x[:, c:c+1, :, :]
        yc = y[:, c:c+1, :, :]

        mu_x = F.conv2d(xc, kernel, padding=pad)
        mu_y = F.conv2d(yc, kernel, padding=pad)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(xc * xc, kernel, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(yc * yc, kernel, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(xc * yc, kernel, padding=pad) - mu_xy

        ssim_map = (
            (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        ) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        ssim_vals.append(ssim_map.mean())

    return torch.stack(ssim_vals).mean()


# ---------------------------------------------------------------------------
# Rendering a single view from a trained model
# ---------------------------------------------------------------------------

def gs_render(
    model: "GaussianModel",
    pose: np.ndarray,
    intrinsics: np.ndarray,
    H: int,
    W: int,
    device: Optional[str] = None,
) -> np.ndarray:
    """Render a novel view from a trained Gaussian splatting model.

    Args:
        model: Trained GaussianModel.
        pose: Camera-to-world matrix (4, 4).
        intrinsics: Camera intrinsic matrix (3, 3).
        H: Output image height.
        W: Output image width.
        device: Torch device string.

    Returns:
        Rendered image (H, W, 3) float32 in [0, 1].
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    near, far = 0.01, 100.0

    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = 2.0 * fy / H
    proj[0, 2] = 2.0 * cx / W - 1.0
    proj[1, 2] = 2.0 * cy / H - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0

    c2w = torch.tensor(pose, device=device, dtype=torch.float32)
    viewmat = torch.inverse(c2w)
    projmat = proj @ viewmat

    with torch.no_grad():
        covs_3d = model.get_covariance_3d()
        means2d, covs2d, depths_val, visible = project_gaussians(
            model.positions, covs_3d, viewmat, projmat, W, H
        )

        cam_pos = c2w[:3, 3]
        viewdirs = F.normalize(model.positions - cam_pos[None, :], dim=-1)
        colors = model.get_colors(viewdirs)
        opacities_act = model.get_opacities()

        rendered = rasterize_gaussians(
            means2d, covs2d, colors, opacities_act, depths_val, visible, H, W
        )

    return rendered.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------

def run_gaussian_splatting(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Gaussian splatting reconstruction (portfolio interface).

    Expects multi-view images and camera parameters from the physics
    operator or configuration.

    Args:
        y: Multi-view images (N, H, W, 3) or stacked measurements.
        physics: Physics operator with camera info.
        cfg: Configuration with:
            - iters: Training iterations (default: 3000).
            - n_init_points: Initial splat count (default: 1000).
            - sh_degree: SH degree (default: 3).
            - render_pose: Pose for novel view (default: first training pose).
            - weights_path: Path to pretrained model checkpoint.

    Returns:
        Tuple of (rendered_image, info_dict).
    """
    _require_torch()

    iters = cfg.get("iters", 3000)
    n_init_points = cfg.get("n_init_points", 1000)
    sh_degree = cfg.get("sh_degree", 3)
    weights_path = cfg.get("weights_path", None)
    device_str = cfg.get("device", None)

    info: Dict[str, Any] = {"solver": "gaussian_splatting"}

    try:
        # Extract camera parameters from physics or config
        poses = None
        intrinsics = None
        init_points = None

        if hasattr(physics, "poses"):
            poses = physics.poses
        if hasattr(physics, "intrinsics"):
            intrinsics = physics.intrinsics
        if hasattr(physics, "init_points"):
            init_points = physics.init_points

        if hasattr(physics, "info"):
            op_info = physics.info()
            poses = op_info.get("poses", poses)
            intrinsics = op_info.get("intrinsics", intrinsics)
            init_points = op_info.get("init_points", init_points)

        # Override from config
        poses = cfg.get("poses", poses)
        intrinsics = cfg.get("intrinsics", intrinsics)
        init_points = cfg.get("init_points", init_points)

        if poses is None or intrinsics is None:
            info["error"] = "missing camera poses or intrinsics"
            return y[0].astype(np.float32) if y.ndim == 4 else y.astype(np.float32), info

        # Ensure poses is a list of arrays
        if isinstance(poses, np.ndarray) and poses.ndim == 3:
            poses = [poses[i] for i in range(poses.shape[0])]

        # Prepare images
        if y.ndim == 4:
            images = [y[i] for i in range(y.shape[0])]
        elif y.ndim == 3:
            images = [y]
        else:
            info["error"] = "unexpected input shape"
            return y.astype(np.float32), info

        H, W = images[0].shape[:2]

        # Check for pretrained weights
        if weights_path is None:
            default_path = _WEIGHTS_DIR / "model.pth"
            if default_path.exists():
                weights_path = str(default_path)

        if weights_path is not None and Path(weights_path).exists():
            # Load pretrained model
            device = torch.device(
                device_str if device_str else
                ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            checkpoint = torch.load(
                weights_path, map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state = checkpoint["model_state"]
                n_pts = state["positions"].shape[0]
                model = GaussianModel(n_points=n_pts, sh_degree=sh_degree).to(device)
                model.load_state_dict(state, strict=False)
            else:
                model = checkpoint
            info["loaded_weights"] = weights_path
        else:
            # Train from scratch
            model, train_info = gs_train(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                n_init_points=n_init_points,
                iters=iters,
                sh_degree=sh_degree,
                init_points=init_points,
                device=device_str,
            )
            info.update(train_info)

        # Render novel view
        render_pose = cfg.get("render_pose", None)
        if render_pose is None:
            render_pose = poses[0]

        rendered = gs_render(model, render_pose, intrinsics, H, W, device=device_str)
        info["n_gaussians"] = model.n_gaussians

        return rendered, info

    except Exception as e:
        info["error"] = str(e)
        if y.ndim == 4:
            return y[0].astype(np.float32), info
        return y.astype(np.float32), info
