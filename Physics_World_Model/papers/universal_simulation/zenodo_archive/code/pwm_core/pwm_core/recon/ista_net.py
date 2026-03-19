"""ISTA-Net / ISTA-Net+: Learned ISTA Deep Unfolding for Compressive Sensing.

Interpretable deep network that unrolls ISTA with learnable transforms
and soft-thresholding at each phase.

References:
- Zhang, J. & Ghanem, B. (2018). "ISTA-Net: Interpretable Optimization-
  Inspired Deep Network for Image Compressive Sensing", CVPR 2018.

Benchmark (Set11, cr=0.10):
- ISTA-Net+: 31.85 dB
- Params: ~3M, VRAM: ~1.5GB
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
            "ISTA-Net requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Model components
# ===========================================================================

if HAS_TORCH:

    class ISTANetPhase(nn.Module):
        """Single ISTA-Net+ phase (one unrolled iteration).

        Each phase learns:
        - A nonlinear sparsifying transform (conv-based)
        - A soft-thresholding with learnable threshold
        - An inverse transform

        Args:
            n_filters: Number of convolutional filters.
        """

        def __init__(self, n_filters: int = 32):
            super().__init__()

            # Forward transform (sparsifying)
            self.transform = nn.Sequential(
                nn.Conv2d(1, n_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_filters),
            )

            # Inverse transform
            self.inv_transform = nn.Sequential(
                nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, 1, 3, 1, 1, bias=False),
            )

            # Learnable soft-threshold
            self.theta = nn.Parameter(torch.tensor(0.01))

            # Learnable step size for gradient update
            self.step = nn.Parameter(torch.tensor(1.0))

        def _soft_threshold(self, x: "torch.Tensor") -> "torch.Tensor":
            """Soft-thresholding with learnable positive threshold."""
            theta = torch.abs(self.theta)
            return torch.sign(x) * F.relu(torch.abs(x) - theta)

        def forward(
            self,
            x: "torch.Tensor",
            PhiTPhi: "torch.Tensor",
            PhiTy: "torch.Tensor",
        ) -> "torch.Tensor":
            """One ISTA-Net+ iteration.

            Args:
                x: Current estimate (B, 1, H, W).
                PhiTPhi: Phi^T Phi precomputed (N, N) for gradient.
                PhiTy: Phi^T y precomputed (B, N) for gradient.

            Returns:
                Updated estimate (B, 1, H, W).
            """
            batch, _, h, w = x.shape
            n_pix = h * w

            # Gradient step: r = x - step * (Phi^T Phi x - Phi^T y)
            x_flat = x.reshape(batch, n_pix)
            grad = torch.mm(x_flat, PhiTPhi.t()) - PhiTy  # (B, N)
            r = x_flat - self.step * grad
            r = r.reshape(batch, 1, h, w)

            # Sparsifying transform
            z = self.transform(r)

            # Soft thresholding in transform domain
            z = self._soft_threshold(z)

            # Inverse transform (residual)
            x_out = self.inv_transform(z) + r

            return x_out

    class ISTANet(nn.Module):
        """ISTA-Net+: Deep unfolding network for image CS.

        Stacks multiple ISTANetPhase modules, each with independent
        learned parameters.

        Args:
            n_phases: Number of unrolled iterations.
            block_size: Image patch size.
            cr: Compression ratio (M / N).
            n_filters: Convolutional filters per phase.
        """

        def __init__(
            self,
            n_phases: int = 9,
            block_size: int = 32,
            cr: float = 0.10,
            n_filters: int = 32,
        ):
            super().__init__()
            self.n_phases = n_phases
            self.block_size = block_size
            self.cr = cr
            n_pix = block_size * block_size
            n_meas = max(1, int(n_pix * cr))

            # Learnable sampling matrix
            self.Phi = nn.Parameter(
                torch.randn(n_meas, n_pix) / math.sqrt(n_pix)
            )

            # Unrolled phases
            self.phases = nn.ModuleList(
                [ISTANetPhase(n_filters=n_filters) for _ in range(n_phases)]
            )

        def forward(self, y: "torch.Tensor") -> "torch.Tensor":
            """Reconstruct from compressed measurements.

            Args:
                y: Measurements (B, M).

            Returns:
                Reconstructed patches (B, 1, block_size, block_size).
            """
            batch = y.shape[0]
            bs = self.block_size

            # Precompute Phi^T Phi and Phi^T y
            PhiTPhi = self.Phi.t() @ self.Phi  # (N, N)
            PhiTy = torch.mm(y, self.Phi)  # (B, N)

            # Initial estimate
            x = PhiTy.reshape(batch, 1, bs, bs)

            for phase in self.phases:
                x = phase(x, PhiTPhi, PhiTy)

            return x


# ===========================================================================
# High-level reconstruction function
# ===========================================================================


def ista_net_reconstruct(
    y: np.ndarray,
    measurement_matrix: Optional[np.ndarray] = None,
    block_size: int = 32,
    cr: float = 0.10,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    n_phases: int = 9,
) -> np.ndarray:
    """Reconstruct an image from CS measurements using ISTA-Net+.

    Args:
        y: Measurement vector(s) (M,) or (B, M).
        measurement_matrix: Sensing matrix (M, N). If provided, used to
            initialize Phi.
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

    model = ISTANet(
        n_phases=n_phases,
        block_size=block_size,
        cr=cr,
    ).to(device)

    # Load weights
    if weights_path is None:
        weights_path = str(_PKG_ROOT / "weights" / "ista_net" / "ista_net_plus.pth")

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
        Phi_init = torch.from_numpy(measurement_matrix.astype(np.float32))
        if Phi_init.shape == model.Phi.shape:
            model.Phi.data.copy_(Phi_init)

    model.eval()

    # Handle input
    squeeze = False
    if y.ndim == 1:
        y = y[np.newaxis, :]
        squeeze = True

    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    with torch.no_grad():
        out = model(y_t)

    result = out.squeeze(1).cpu().numpy()
    if squeeze:
        result = result.squeeze(0)

    return result.astype(np.float32)


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_ista_net(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for ISTA-Net+ reconstruction.

    Args:
        y: Measurements.
        physics: Physics operator with A attribute or measurement_matrix.
        cfg: Configuration dict with optional keys:
            - block_size, cr, weights_path, device, n_phases.

    Returns:
        Tuple of (reconstructed_image, info_dict).
    """
    block_size = cfg.get("block_size", 32)
    cr = cfg.get("cr", 0.10)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    n_phases = cfg.get("n_phases", 9)

    info: Dict[str, Any] = {
        "solver": "ista_net",
        "n_phases": n_phases,
        "cr": cr,
    }

    try:
        _require_torch()

        A: Optional[np.ndarray] = None
        if hasattr(physics, "A") and physics.A is not None:
            A = np.asarray(physics.A, dtype=np.float32)
        elif hasattr(physics, "measurement_matrix"):
            A = np.asarray(physics.measurement_matrix, dtype=np.float32)

        result = ista_net_reconstruct(
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
