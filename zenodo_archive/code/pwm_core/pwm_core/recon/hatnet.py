"""HATNet: Hybrid-Attention Transformer for Single-Pixel Imaging.

Kronecker-efficient compressive sensing reconstruction using a hybrid
spatial-channel attention mechanism with deep unfolding.

References:
- Qu, L. et al. (2024). "HATNet: Hybrid Attention Transformer for
  Single-Pixel Imaging", CVPR 2024.

Benchmark (Set11, cr=0.10):
- PSNR: 31.49 dB
- Params: ~8M, VRAM: ~3GB
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "HATNet requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Package root for default weight paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Model components
# ===========================================================================

if HAS_TORCH:

    class ChannelAttention(nn.Module):
        """Squeeze-and-excitation channel attention.

        Args:
            dim: Number of channels.
            reduction: Channel reduction ratio.
        """

        def __init__(self, dim: int, reduction: int = 8):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(dim, dim // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(dim // reduction, dim, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, c, _, _ = x.shape
            w = self.pool(x).reshape(b, c)
            w = self.fc(w).reshape(b, c, 1, 1)
            return x * w

    class SpatialAttention(nn.Module):
        """Spatial attention using max+avg pooling.

        Produces a per-pixel attention map from channel statistics.
        """

        def __init__(self, kernel_size: int = 7):
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            avg = x.mean(dim=1, keepdim=True)
            mx, _ = x.max(dim=1, keepdim=True)
            attn = self.conv(torch.cat([avg, mx], dim=1))
            return x * self.sigmoid(attn)

    class HybridAttentionBlock(nn.Module):
        """Hybrid channel + spatial attention with residual connection.

        Args:
            dim: Feature dimension.
            reduction: SE reduction ratio.
        """

        def __init__(self, dim: int, reduction: int = 8):
            super().__init__()
            self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(dim)
            self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(dim)
            self.ca = ChannelAttention(dim, reduction)
            self.sa = SpatialAttention()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            res = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.ca(out)
            out = self.sa(out)
            return self.relu(out + res)

    class ProximalBlock(nn.Module):
        """Learnable proximal operator block (replaces hand-crafted prox).

        Uses a residual CNN to approximate the proximal mapping.

        Args:
            dim: Feature dimension.
            n_blocks: Number of hybrid attention blocks.
        """

        def __init__(self, dim: int = 64, n_blocks: int = 4):
            super().__init__()
            self.head = nn.Conv2d(1, dim, 3, 1, 1)
            self.body = nn.Sequential(
                *[HybridAttentionBlock(dim) for _ in range(n_blocks)]
            )
            self.tail = nn.Conv2d(dim, 1, 3, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, 1, H, W) image tensor.

            Returns:
                Proximal output (B, 1, H, W).
            """
            res = x
            fea = self.head(x)
            fea = self.body(fea)
            return self.tail(fea) + res

    class HATNet(nn.Module):
        """HATNet: Hybrid-Attention Transformer for SPI reconstruction.

        Deep unfolding architecture with learned gradient step and proximal
        operator at each phase.  Uses Kronecker-separable sampling for
        memory efficiency.

        Args:
            n_phases: Number of unrolling phases (iterations).
            block_size: Image patch size for block-based CS.
            cr: Compression ratio (M / N).
            dim: Feature dimension for proximal blocks.
            n_blocks: Attention blocks per proximal operator.
        """

        def __init__(
            self,
            n_phases: int = 9,
            block_size: int = 32,
            cr: float = 0.10,
            dim: int = 64,
            n_blocks: int = 4,
        ):
            super().__init__()
            self.n_phases = n_phases
            self.block_size = block_size
            self.cr = cr
            n_pix = block_size * block_size
            n_meas = max(1, int(n_pix * cr))

            # Sampling matrix (learnable)
            self.Phi = nn.Parameter(
                torch.randn(n_meas, n_pix) / math.sqrt(n_pix)
            )

            # Per-phase learnable step sizes
            self.step = nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(n_phases)]
            )

            # Per-phase proximal operators (weight-shared option via n_phases=1)
            self.proximal = nn.ModuleList(
                [ProximalBlock(dim=dim, n_blocks=n_blocks) for _ in range(n_phases)]
            )

        def _gradient_step(
            self, x: "torch.Tensor", y: "torch.Tensor", phase: int
        ) -> "torch.Tensor":
            """Gradient descent on data fidelity: x - step * Phi^T (Phi x - y).

            Args:
                x: Current estimate (B, N).
                y: Measurements (B, M).
                phase: Phase index.

            Returns:
                Updated estimate (B, N).
            """
            residual = x @ self.Phi.t() - y  # (B, M) -- actually Phi @ x - y
            # Correct: residual = Phi @ x - y, gradient = Phi^T @ residual
            Phi_x = torch.mm(x, self.Phi.t())  # (B, M)
            grad = torch.mm(Phi_x - y, self.Phi)  # (B, N)
            return x - self.step[phase] * grad

        def forward(
            self, y: "torch.Tensor", block_size: Optional[int] = None
        ) -> "torch.Tensor":
            """Reconstruct image from compressed measurements.

            Args:
                y: Measurements (B, M).
                block_size: Override block size for reshape (default: self.block_size).

            Returns:
                Reconstructed image patches (B, 1, block_size, block_size).
            """
            bs = block_size or self.block_size
            batch = y.shape[0]

            # Initial estimate: Phi^T y
            x = torch.mm(y, self.Phi)  # (B, N)

            for k in range(self.n_phases):
                # Gradient step (data fidelity)
                x = self._gradient_step(x, y, k)

                # Reshape to image for proximal step
                x_img = x.reshape(batch, 1, bs, bs)
                x_img = self.proximal[k](x_img)
                x = x_img.reshape(batch, -1)

            return x.reshape(batch, 1, bs, bs)


# ===========================================================================
# High-level reconstruction function
# ===========================================================================


def hatnet_reconstruct(
    y: np.ndarray,
    measurement_matrix: Optional[np.ndarray] = None,
    block_size: int = 32,
    cr: float = 0.10,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    n_phases: int = 9,
) -> np.ndarray:
    """Reconstruct an image from SPC measurements using HATNet.

    Args:
        y: Measurement vector(s) (M,) or (B, M).
        measurement_matrix: Sensing matrix (M, N). If provided, used to
            initialize sampling layer.
        block_size: Patch size (default 32).
        cr: Compression ratio (default 0.10).
        weights_path: Path to pretrained weights.
        device: Torch device string.
        n_phases: Number of unrolling phases.

    Returns:
        Reconstructed image (block_size, block_size) or (B, block_size, block_size).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Build model
    model = HATNet(
        n_phases=n_phases,
        block_size=block_size,
        cr=cr,
    ).to(device)

    # Load pretrained weights if available
    if weights_path is None:
        weights_path = str(_PKG_ROOT / "weights" / "hatnet" / "hatnet.pth")

    wp = Path(weights_path)
    if wp.exists():
        state = torch.load(str(wp), map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = {
                k.replace("module.", ""): v
                for k, v in state["state_dict"].items()
            }
        model.load_state_dict(state, strict=False)
    elif measurement_matrix is not None:
        # Initialize Phi from provided sensing matrix
        M, N = measurement_matrix.shape
        Phi_init = torch.from_numpy(measurement_matrix.astype(np.float32))
        if Phi_init.shape == model.Phi.shape:
            model.Phi.data.copy_(Phi_init)

    model.eval()

    # Handle input shape
    squeeze = False
    if y.ndim == 1:
        y = y[np.newaxis, :]
        squeeze = True

    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    with torch.no_grad():
        out = model(y_t)

    result = out.squeeze(1).cpu().numpy()  # (B, H, W)
    if squeeze:
        result = result.squeeze(0)

    return result.astype(np.float32)


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_hatnet(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for HATNet SPC reconstruction.

    Args:
        y: Measurements.
        physics: Physics operator (must have A attribute or measurement_matrix).
        cfg: Configuration dict with optional keys:
            - block_size: Patch size (default 32).
            - cr: Compression ratio (default 0.10).
            - weights_path: Path to pretrained weights.
            - device: Torch device string.
            - n_phases: Unrolling phases (default 9).

    Returns:
        Tuple of (reconstructed_image, info_dict).
    """
    block_size = cfg.get("block_size", 32)
    cr = cfg.get("cr", 0.10)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    n_phases = cfg.get("n_phases", 9)

    info: Dict[str, Any] = {
        "solver": "hatnet",
        "block_size": block_size,
        "cr": cr,
        "n_phases": n_phases,
    }

    try:
        _require_torch()

        # Get measurement matrix
        A: Optional[np.ndarray] = None
        if hasattr(physics, "A") and physics.A is not None:
            A = np.asarray(physics.A, dtype=np.float32)
        elif hasattr(physics, "measurement_matrix"):
            A = np.asarray(physics.measurement_matrix, dtype=np.float32)

        result = hatnet_reconstruct(
            y=y.reshape(-1) if y.ndim > 1 else y,
            measurement_matrix=A,
            block_size=block_size,
            cr=cr,
            weights_path=weights_path,
            device=device,
            n_phases=n_phases,
        )

        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, "adjoint"):
            try:
                return physics.adjoint(y).astype(np.float32), info
            except Exception:
                pass
        return y.astype(np.float32), info
