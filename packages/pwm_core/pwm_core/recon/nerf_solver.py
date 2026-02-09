"""NeRF: Neural Radiance Fields for Novel View Synthesis.

MLP-based and hash-grid neural scene representations.

References:
- Mildenhall, B. et al. (2020). "NeRF: Representing Scenes as Neural
  Radiance Fields for View Synthesis", ECCV 2020.
- Muller, T. et al. (2022). "Instant Neural Graphics Primitives with a
  Multiresolution Hash Encoding", SIGGRAPH 2022. (Instant-NGP)
- Barron, J.T. et al. (2022). "Mip-NeRF 360: Unbounded Anti-Aliased
  Neural Radiance Fields", CVPR 2022.

Benchmark:
- NeRF (MLP): ~31 dB PSNR on synthetic scenes
- Instant-NGP: ~33 dB, trains in seconds
- Mip-NeRF 360: ~34 dB on unbounded scenes (needs 24+ GB VRAM)
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


_WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights" / "nerf"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "nerf_solver requires PyTorch.  Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class PositionalEncoding:
    """Sinusoidal positional encoding for coordinates.

    Maps a scalar coordinate *x* to
        [sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(2^{L-1} pi x)]

    Args:
        num_freqs: Number of frequency bands (*L*).
        include_input: If True, prepend the raw coordinate to the encoding.
    """

    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Pre-compute frequency bands: 2^0 ... 2^{L-1}
        self.freq_bands = 2.0 ** np.arange(num_freqs).astype(np.float32)

    @property
    def out_dim(self) -> int:
        d = self.num_freqs * 2
        if self.include_input:
            d += 1
        return d

    # NumPy path (used during data preparation) --------------------------------

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        """Encode with NumPy.  *x* has shape ``(..., D)``."""
        parts: List[np.ndarray] = []
        if self.include_input:
            parts.append(x)
        for freq in self.freq_bands:
            parts.append(np.sin(freq * np.pi * x))
            parts.append(np.cos(freq * np.pi * x))
        return np.concatenate(parts, axis=-1)

    # Torch path (used inside nn.Module forward) --------------------------------

    def encode_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """Encode with PyTorch.  *x* has shape ``(..., D)``."""
        _require_torch()
        parts: list = []
        if self.include_input:
            parts.append(x)
        for freq in self.freq_bands:
            parts.append(torch.sin(freq * np.pi * x))
            parts.append(torch.cos(freq * np.pi * x))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# NeRF MLP (original architecture)
# ---------------------------------------------------------------------------


if HAS_TORCH:

    class NeRFMLP(nn.Module):
        """Original NeRF MLP.

        8-layer, 256-channel MLP with a skip connection at layer 4.

        * Input:  pos(3) -> PE(pos) -> density sigma + 256-d feature
        * Input:  dir(3) -> PE(dir) + feature -> RGB

        Approximately 1.2 M parameters.

        Args:
            pos_freqs: Number of PE frequency bands for position (default 10).
            dir_freqs: Number of PE frequency bands for direction (default 4).
            hidden_dim: Hidden layer width (default 256).
            n_layers: Number of layers before the density head (default 8).
            skip_layer: Index of the skip-connection layer (default 4).
        """

        def __init__(
            self,
            pos_freqs: int = 10,
            dir_freqs: int = 4,
            hidden_dim: int = 256,
            n_layers: int = 8,
            skip_layer: int = 4,
        ):
            super().__init__()
            self.pos_enc = PositionalEncoding(pos_freqs, include_input=True)
            self.dir_enc = PositionalEncoding(dir_freqs, include_input=True)

            pos_in = 3 * self.pos_enc.out_dim  # 3 coords x out_dim
            dir_in = 3 * self.dir_enc.out_dim

            self.skip_layer = skip_layer

            # Build MLP layers for density path
            layers: List[nn.Module] = [nn.Linear(pos_in, hidden_dim), nn.ReLU(True)]
            for i in range(1, n_layers):
                in_dim = hidden_dim + pos_in if i == skip_layer else hidden_dim
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(True))
            self.density_layers = nn.ModuleList(
                [layers[j] for j in range(0, len(layers), 2)]
            )
            self.density_acts = nn.ModuleList(
                [layers[j] for j in range(1, len(layers), 2)]
            )

            # Density head
            self.sigma_head = nn.Linear(hidden_dim, 1)

            # Feature -> color head
            self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
            self.color_layer = nn.Sequential(
                nn.Linear(hidden_dim + dir_in, hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, 3),
                nn.Sigmoid(),
            )

        def forward(
            self,
            pos: "torch.Tensor",
            dirs: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Evaluate radiance field.

            Args:
                pos:  (..., 3)  world-space positions.
                dirs: (..., 3)  unit view directions.

            Returns:
                rgb:   (..., 3)  colour in [0, 1].
                sigma: (..., 1)  volume density (non-negative via softplus).
            """
            # Positional encoding
            pos_enc = self.pos_enc.encode_torch(pos)
            dir_enc = self.dir_enc.encode_torch(dirs)

            h = pos_enc
            for idx, (lin, act) in enumerate(
                zip(self.density_layers, self.density_acts)
            ):
                if idx == self.skip_layer:
                    h = torch.cat([h, pos_enc], dim=-1)
                h = act(lin(h))

            sigma = F.softplus(self.sigma_head(h))  # (..., 1)

            feat = self.feature_proj(h)
            rgb = self.color_layer(torch.cat([feat, dir_enc], dim=-1))  # (..., 3)

            return rgb, sigma

    # -------------------------------------------------------------------
    # Hash Encoding (Instant-NGP style)
    # -------------------------------------------------------------------

    class HashEncoding(nn.Module):
        """Multi-resolution hash grid encoding (Instant-NGP).

        Each of *L* levels stores a hash table of trainable feature vectors.
        Resolution increases geometrically from *base_resolution* to
        *max_resolution*.  Trilinear interpolation is performed within each
        level, and the results are concatenated.

        Args:
            n_levels: Number of resolution levels (default 16).
            n_features_per_level: Feature vector size per level (default 2).
            log2_hashmap_size: log2 of hash table size (default 19 -> 524288).
            base_resolution: Coarsest grid resolution (default 16).
            max_resolution: Finest grid resolution (default 2048).
        """

        def __init__(
            self,
            n_levels: int = 16,
            n_features_per_level: int = 2,
            log2_hashmap_size: int = 19,
            base_resolution: int = 16,
            max_resolution: int = 2048,
        ):
            super().__init__()
            self.n_levels = n_levels
            self.n_features = n_features_per_level
            self.hashmap_size = 2 ** log2_hashmap_size

            # Geometric growth factor per level
            self.growth = np.exp(
                (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
            )
            self.base_resolution = base_resolution

            # Hash tables (one Embedding per level)
            self.tables = nn.ModuleList()
            for _ in range(n_levels):
                table = nn.Embedding(self.hashmap_size, n_features_per_level)
                nn.init.uniform_(table.weight, -1e-4, 1e-4)
                self.tables.append(table)

            # Prime numbers for spatial hashing
            self.primes = torch.tensor(
                [1, 2654435761, 805459861], dtype=torch.long
            )

        @property
        def out_dim(self) -> int:
            return self.n_levels * self.n_features

        def _hash(self, coords_int: "torch.Tensor") -> "torch.Tensor":
            """Spatial hash function.  coords_int: (..., 3) long tensor."""
            primes = self.primes.to(coords_int.device)
            h = torch.zeros(coords_int.shape[:-1], dtype=torch.long,
                            device=coords_int.device)
            for d in range(3):
                h = h ^ (coords_int[..., d] * primes[d])
            return h % self.hashmap_size

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode positions.

            Args:
                x: (..., 3)  positions normalised to [0, 1].

            Returns:
                (..., n_levels * n_features)  feature vector.
            """
            outputs: list = []
            for lv, table in enumerate(self.tables):
                res = int(self.base_resolution * (self.growth ** lv))
                # Scale to voxel coordinates
                scaled = x * res
                floor = torch.floor(scaled).long()
                frac = scaled - floor.float()

                # 8 corners for trilinear interpolation
                offsets = torch.tensor(
                    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                    device=x.device, dtype=torch.long,
                )

                corner_features = []
                for off in offsets:
                    corner = floor + off  # (..., 3)
                    idx = self._hash(corner)  # (...)
                    feat = table(idx)  # (..., n_features)
                    corner_features.append(feat)

                # Trilinear weights
                wx = frac[..., 0:1]
                wy = frac[..., 1:2]
                wz = frac[..., 2:3]

                c000, c001, c010, c011 = corner_features[:4]
                c100, c101, c110, c111 = corner_features[4:]

                c00 = c000 * (1 - wz) + c001 * wz
                c01 = c010 * (1 - wz) + c011 * wz
                c10 = c100 * (1 - wz) + c101 * wz
                c11 = c110 * (1 - wz) + c111 * wz

                c0 = c00 * (1 - wy) + c01 * wy
                c1 = c10 * (1 - wy) + c11 * wy

                interp = c0 * (1 - wx) + c1 * wx  # (..., n_features)
                outputs.append(interp)

            return torch.cat(outputs, dim=-1)

    # -------------------------------------------------------------------
    # Instant-NGP Model
    # -------------------------------------------------------------------

    class InstantNGP(nn.Module):
        """Instant-NGP variant: hash encoding + small MLP.

        Approximately 5 M parameters (mostly in hash tables).

        Args:
            n_levels: Hash encoding levels.
            n_features_per_level: Feature dim per level.
            log2_hashmap_size: Hash table size (log2).
            base_resolution: Coarsest resolution.
            max_resolution: Finest resolution.
            hidden_dim: MLP hidden width (default 64).
            dir_freqs: PE frequencies for direction (default 4).
        """

        def __init__(
            self,
            n_levels: int = 16,
            n_features_per_level: int = 2,
            log2_hashmap_size: int = 19,
            base_resolution: int = 16,
            max_resolution: int = 2048,
            hidden_dim: int = 64,
            dir_freqs: int = 4,
        ):
            super().__init__()
            self.hash_enc = HashEncoding(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                max_resolution=max_resolution,
            )
            self.dir_enc = PositionalEncoding(dir_freqs, include_input=True)

            hash_dim = self.hash_enc.out_dim
            dir_dim = 3 * self.dir_enc.out_dim

            # Density MLP (small)
            self.density_net = nn.Sequential(
                nn.Linear(hash_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
            )
            self.sigma_head = nn.Linear(hidden_dim, 1)

            # Colour MLP
            self.color_net = nn.Sequential(
                nn.Linear(hidden_dim + dir_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 3),
                nn.Sigmoid(),
            )

        def forward(
            self,
            pos: "torch.Tensor",
            dirs: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Evaluate radiance field.

            Args:
                pos:  (..., 3)  positions in [0, 1].
                dirs: (..., 3)  unit view directions.

            Returns:
                rgb:   (..., 3)
                sigma: (..., 1)
            """
            h_enc = self.hash_enc(pos)
            h = self.density_net(h_enc)

            sigma = F.softplus(self.sigma_head(h))

            dir_enc = self.dir_enc.encode_torch(dirs)
            rgb = self.color_net(torch.cat([h, dir_enc], dim=-1))

            return rgb, sigma


# ---------------------------------------------------------------------------
# Volume Rendering
# ---------------------------------------------------------------------------


def volume_render(
    rays_o: "torch.Tensor",
    rays_d: "torch.Tensor",
    model: "nn.Module",
    near: float = 0.0,
    far: float = 1.0,
    n_samples: int = 64,
    perturb: bool = True,
    white_bg: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Render rays through a radiance field via alpha compositing.

    Implements stratified sampling along each ray, queries *model* for
    (density, colour), and accumulates via the standard NeRF quadrature.

    Args:
        rays_o: (N, 3) ray origins.
        rays_d: (N, 3) ray directions (need not be unit-length).
        model:  callable(pos, dirs) -> (rgb, sigma).
        near:   near plane distance.
        far:    far plane distance.
        n_samples: Number of samples per ray.
        perturb: If True, apply stratified jitter to sample positions.
        white_bg: If True, composite over a white background.

    Returns:
        rgb_map:   (N, 3)  rendered colours.
        depth_map: (N,)    expected depth.
        acc_map:   (N,)    accumulated opacity (alpha).
    """
    _require_torch()
    device = rays_o.device
    N = rays_o.shape[0]

    # Stratified samples in [near, far]
    t_vals = torch.linspace(near, far, n_samples, device=device)  # (S,)
    t_vals = t_vals.unsqueeze(0).expand(N, -1)  # (N, S)

    if perturb:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    # 3D sample positions:  (N, S, 3)
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)

    # Unit direction (broadcast to each sample)
    dirs = F.normalize(rays_d, dim=-1)  # (N, 3)
    dirs_exp = dirs.unsqueeze(1).expand_as(pts)  # (N, S, 3)

    # Flatten, evaluate model, reshape
    flat_pts = pts.reshape(-1, 3)
    flat_dirs = dirs_exp.reshape(-1, 3)
    rgb, sigma = model(flat_pts, flat_dirs)  # (-1, 3), (-1, 1)
    rgb = rgb.reshape(N, n_samples, 3)
    sigma = sigma.reshape(N, n_samples)

    # Delta distances between samples
    deltas = t_vals[..., 1:] - t_vals[..., :-1]  # (N, S-1)
    # Last delta is "infinite" (large)
    deltas = torch.cat(
        [deltas, torch.full((N, 1), 1e10, device=device)], dim=-1
    )  # (N, S)

    # Alpha compositing
    alpha = 1.0 - torch.exp(-sigma * deltas)  # (N, S)
    # Transmittance: T_i = prod_{j<i} (1 - alpha_j)
    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones(N, 1, device=device), 1.0 - alpha + 1e-10], dim=-1
        ),
        dim=-1,
    )[..., :-1]  # (N, S)

    weights = alpha * transmittance  # (N, S)

    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (N, 3)
    depth_map = (weights * t_vals).sum(dim=1)  # (N,)
    acc_map = weights.sum(dim=1)  # (N,)

    if white_bg:
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))

    return rgb_map, depth_map, acc_map


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------


def _get_rays(
    H: int,
    W: int,
    focal: float,
    pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rays for a pinhole camera.

    Args:
        H, W: Image height and width in pixels.
        focal: Focal length in pixels.
        pose: (4, 4) camera-to-world matrix.

    Returns:
        rays_o: (H*W, 3)
        rays_d: (H*W, 3)
    """
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )
    dirs = np.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -np.ones_like(i),
        ],
        axis=-1,
    )  # (H, W, 3) in camera space

    # Rotate to world
    rot = pose[:3, :3]  # (3, 3)
    rays_d = (dirs[..., None, :] @ rot.T).squeeze(-2)  # (H, W, 3)
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape).copy()  # (H, W, 3)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def _intrinsics_to_focal(intrinsics: np.ndarray) -> float:
    """Extract focal length from a (3,3) intrinsic matrix or scalar."""
    if np.isscalar(intrinsics) or intrinsics.ndim == 0:
        return float(intrinsics)
    return float(intrinsics[0, 0])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def nerf_train(
    images: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    model: str = "instant_ngp",
    iters: int = 1000,
    lr: float = 5e-4,
    batch_rays: int = 1024,
    n_samples: int = 64,
    near: float = 0.0,
    far: float = 1.0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple["nn.Module", Dict[str, Any]]:
    """Train a NeRF model from posed images.

    Args:
        images:     (N, H, W, 3) float32 images in [0, 1].
        poses:      (N, 4, 4) camera-to-world matrices.
        intrinsics: (3, 3) intrinsic matrix **or** scalar focal length.
        model:      ``"nerf"`` (original MLP) or ``"instant_ngp"`` (hash grid).
        iters:      Number of optimisation iterations.
        lr:         Learning rate.
        batch_rays: Number of rays per mini-batch.
        n_samples:  Samples per ray.
        near:       Near plane.
        far:        Far plane.
        device:     Torch device (auto-detect if None).
        verbose:    If True, print loss every 100 iterations.

    Returns:
        Tuple of (trained_model, info_dict).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    N_imgs, H, W, _ = images.shape
    focal = _intrinsics_to_focal(intrinsics)

    # Build all rays
    all_rays_o_list: List[np.ndarray] = []
    all_rays_d_list: List[np.ndarray] = []
    all_rgb_list: List[np.ndarray] = []
    for idx in range(N_imgs):
        ro, rd = _get_rays(H, W, focal, poses[idx])
        all_rays_o_list.append(ro)
        all_rays_d_list.append(rd)
        all_rgb_list.append(images[idx].reshape(-1, 3))

    all_rays_o = torch.from_numpy(np.concatenate(all_rays_o_list, axis=0)).float().to(device)
    all_rays_d = torch.from_numpy(np.concatenate(all_rays_d_list, axis=0)).float().to(device)
    all_rgb = torch.from_numpy(np.concatenate(all_rgb_list, axis=0)).float().to(device)
    total_rays = all_rays_o.shape[0]

    # Create model
    if model == "instant_ngp":
        net = InstantNGP().to(device)
    else:
        net = NeRFMLP().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    info: Dict[str, Any] = {
        "model_type": model,
        "iters": iters,
        "lr": lr,
        "losses": [],
    }

    net.train()
    for step in range(iters):
        # Random ray batch
        idx = torch.randint(0, total_rays, (batch_rays,), device=device)
        ro_batch = all_rays_o[idx]
        rd_batch = all_rays_d[idx]
        gt_batch = all_rgb[idx]

        rgb_pred, _, _ = volume_render(
            ro_batch, rd_batch, net,
            near=near, far=far,
            n_samples=n_samples,
            perturb=True,
        )

        loss = F.mse_loss(rgb_pred, gt_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        info["losses"].append(loss_val)

        if verbose and (step % 100 == 0 or step == iters - 1):
            psnr_est = -10.0 * np.log10(max(loss_val, 1e-10))
            print(f"  [nerf_train] step {step:>5d}/{iters}  "
                  f"loss={loss_val:.6f}  PSNR~{psnr_est:.2f} dB")

    net.eval()
    final_loss = info["losses"][-1] if info["losses"] else float("inf")
    info["final_loss"] = final_loss
    info["final_psnr_est"] = -10.0 * np.log10(max(final_loss, 1e-10))

    return net, info


# ---------------------------------------------------------------------------
# Rendering a single view
# ---------------------------------------------------------------------------


def nerf_render(
    model: "nn.Module",
    pose: np.ndarray,
    intrinsics: np.ndarray,
    H: int,
    W: int,
    near: float = 0.0,
    far: float = 1.0,
    n_samples: int = 64,
    chunk: int = 4096,
    device: Optional[str] = None,
) -> np.ndarray:
    """Render a single view from a trained NeRF model.

    Args:
        model:      Trained NeRFMLP or InstantNGP.
        pose:       (4, 4) camera-to-world matrix.
        intrinsics: (3, 3) intrinsic matrix **or** scalar focal length.
        H, W:       Output image size in pixels.
        near:       Near plane distance.
        far:        Far plane distance.
        n_samples:  Samples per ray.
        chunk:      Max rays processed at once (memory management).
        device:     Torch device.

    Returns:
        Rendered image as (H, W, 3) float32 in [0, 1].
    """
    _require_torch()

    if device is None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    device = torch.device(device)

    focal = _intrinsics_to_focal(intrinsics)
    rays_o, rays_d = _get_rays(H, W, focal, pose)
    rays_o_t = torch.from_numpy(rays_o).float().to(device)
    rays_d_t = torch.from_numpy(rays_d).float().to(device)

    model.eval()
    rgb_chunks: list = []
    with torch.no_grad():
        for i in range(0, rays_o_t.shape[0], chunk):
            ro = rays_o_t[i: i + chunk]
            rd = rays_d_t[i: i + chunk]
            rgb, _, _ = volume_render(
                ro, rd, model,
                near=near, far=far,
                n_samples=n_samples,
                perturb=False,
            )
            rgb_chunks.append(rgb.cpu())

    rendered = torch.cat(rgb_chunks, dim=0).numpy()
    rendered = np.clip(rendered.reshape(H, W, 3), 0.0, 1.0)
    return rendered.astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------


def run_nerf(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible NeRF reconstruction entry point.

    Expects *physics* to carry camera poses and intrinsics (e.g. from
    ``NeRFOperator``).  Falls back gracefully when metadata is missing.

    Args:
        y:       Multi-view images (N, H, W, 3) **or** (N, H, W).
        physics: Physics operator with pose/intrinsic metadata.
        cfg:     Configuration dict with optional keys:
                 - model: ``"nerf"`` or ``"instant_ngp"`` (default).
                 - iters: Training iterations (default 1000).
                 - lr: Learning rate (default 5e-4).
                 - near / far: Scene bounds (default 0 / 1).
                 - n_samples: Samples per ray (default 64).
                 - render_idx: Index of view to render (default 0).
                 - weights_path: Path to pre-trained weights.

    Returns:
        Tuple of (rendered_image, info_dict).
    """
    _require_torch()

    model_type = cfg.get("model", "instant_ngp")
    iters = cfg.get("iters", 1000)
    lr = cfg.get("lr", 5e-4)
    near = cfg.get("near", 0.0)
    far = cfg.get("far", 1.0)
    n_samples = cfg.get("n_samples", 64)
    render_idx = cfg.get("render_idx", 0)
    weights_path = cfg.get("weights_path", None)
    device_str = cfg.get("device", None)

    info: Dict[str, Any] = {
        "solver": "nerf",
        "model": model_type,
    }

    try:
        # ---- Reshape input ------------------------------------------------
        if y.ndim == 3:
            # (N, H, W) grayscale -> (N, H, W, 3)
            y = np.stack([y, y, y], axis=-1)
        if y.ndim == 2:
            # Single grayscale image
            y = y[np.newaxis, :, :, np.newaxis]
            y = np.broadcast_to(y, y.shape[:3] + (3,)).copy()

        N_imgs, H, W, C = y.shape
        images = y.astype(np.float32)
        if images.max() > 1.5:
            images = images / 255.0

        # ---- Retrieve poses and intrinsics from physics -------------------
        poses = None
        intrinsics_val: Any = None

        if hasattr(physics, "info"):
            op_info = physics.info()
            if "poses" in op_info:
                poses = np.array(op_info["poses"])
            if "intrinsics" in op_info:
                intrinsics_val = np.array(op_info["intrinsics"])

        if poses is None and hasattr(physics, "poses"):
            poses = np.array(physics.poses)

        if poses is None:
            # Generate default circular poses
            poses = _default_circular_poses(N_imgs, radius=2.0)
            info["poses_source"] = "default_circular"

        if intrinsics_val is None and hasattr(physics, "intrinsics"):
            intrinsics_val = np.array(physics.intrinsics)

        if intrinsics_val is None:
            # Default: focal = max(H, W)
            intrinsics_val = np.array(max(H, W), dtype=np.float32)
            info["intrinsics_source"] = "default"

        # ---- Load or train ------------------------------------------------
        if weights_path is not None and Path(weights_path).exists():
            # Load pre-trained model
            if device_str is None:
                device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)

            if model_type == "instant_ngp":
                net = InstantNGP().to(device)
            else:
                net = NeRFMLP().to(device)

            state = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = {
                    k.replace("module.", ""): v for k, v in state["state_dict"].items()
                }
            net.load_state_dict(state, strict=False)
            net.eval()
            info["weights"] = str(weights_path)
            train_info: Dict[str, Any] = {}
        else:
            net, train_info = nerf_train(
                images=images,
                poses=poses,
                intrinsics=intrinsics_val,
                model=model_type,
                iters=iters,
                lr=lr,
                n_samples=n_samples,
                near=near,
                far=far,
                device=device_str,
                verbose=cfg.get("verbose", False),
            )
            info.update(train_info)

        # ---- Render requested view ----------------------------------------
        render_pose = poses[min(render_idx, len(poses) - 1)]
        rendered = nerf_render(
            model=net,
            pose=render_pose,
            intrinsics=intrinsics_val,
            H=H,
            W=W,
            near=near,
            far=far,
            n_samples=n_samples,
            device=device_str,
        )

        return rendered, info

    except Exception as e:
        info["error"] = str(e)
        # Graceful fallback: return first input view
        if y.ndim >= 3:
            fallback = y[0] if y.ndim == 4 else y
            return fallback.astype(np.float32), info
        return y.astype(np.float32), info


# ---------------------------------------------------------------------------
# Helper: default camera poses
# ---------------------------------------------------------------------------


def _default_circular_poses(
    n_views: int,
    radius: float = 2.0,
    height: float = 0.5,
) -> np.ndarray:
    """Generate default camera-to-world matrices on a circle.

    Cameras point toward the origin from equally-spaced azimuth angles.

    Args:
        n_views: Number of views.
        radius:  Camera orbit radius.
        height:  Camera height above the XZ plane.

    Returns:
        (n_views, 4, 4) float32 array of camera-to-world matrices.
    """
    poses = np.zeros((n_views, 4, 4), dtype=np.float32)
    for i in range(n_views):
        angle = 2.0 * np.pi * i / n_views
        # Camera position
        cx = radius * np.cos(angle)
        cz = radius * np.sin(angle)
        cy = height
        pos = np.array([cx, cy, cz], dtype=np.float32)

        # Look-at direction (toward origin)
        forward = -pos / (np.linalg.norm(pos) + 1e-8)
        # World up
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        # Camera-to-world (OpenGL convention: -Z forward)
        poses[i, :3, 0] = right
        poses[i, :3, 1] = up
        poses[i, :3, 2] = -forward
        poses[i, :3, 3] = pos
        poses[i, 3, 3] = 1.0

    return poses
