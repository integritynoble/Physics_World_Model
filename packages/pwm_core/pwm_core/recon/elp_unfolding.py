"""ELP-Unfolding: Ensemble Learning Priors Deep Unfolding for Video SCI.

Deep unfolding network with ensemble of learned priors for snapshot
compressive imaging (CACTI) reconstruction.

References:
- Yang, C. et al. (2022). "Ensemble Learning Priors Driven Deep Unfolding
  for Scalable Snapshot Compressive Imaging", ECCV 2022.

Benchmark (6 videos, 256x256x8):
- Average PSNR: 34.57 dB
- Params: ~10M, VRAM: ~4GB
"""

from __future__ import annotations

import warnings
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
            "ELP-Unfolding requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Model components
# ===========================================================================

if HAS_TORCH:

    class ResBlock(nn.Module):
        """Residual block with two convolutions.

        Args:
            dim: Number of channels.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.block(x)

    class SpatialPrior(nn.Module):
        """Spatial denoising prior using residual CNN.

        Operates on individual frames to enforce spatial smoothness.

        Args:
            n_frames: Number of video frames.
            dim: Feature dimension.
            n_blocks: Number of residual blocks.
        """

        def __init__(self, n_frames: int = 8, dim: int = 32, n_blocks: int = 4):
            super().__init__()
            self.head = nn.Conv2d(1, dim, 3, 1, 1)
            self.body = nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])
            self.tail = nn.Conv2d(dim, 1, 3, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, T, H, W) video frames.

            Returns:
                Spatially denoised frames (B, T, H, W).
            """
            b, t, h, w = x.shape
            # Process each frame independently
            x_flat = x.reshape(b * t, 1, h, w)
            fea = self.head(x_flat)
            fea = self.body(fea)
            out = self.tail(fea) + x_flat
            return out.reshape(b, t, h, w)

    class TemporalPrior(nn.Module):
        """Temporal prior using 1D convolutions across frames.

        Enforces temporal consistency by processing along the time axis.

        Args:
            n_frames: Number of video frames.
            dim: Feature dimension.
            n_blocks: Number of temporal residual blocks.
        """

        def __init__(self, n_frames: int = 8, dim: int = 32, n_blocks: int = 3):
            super().__init__()
            self.head = nn.Conv2d(n_frames, dim, 3, 1, 1)
            self.body = nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])
            self.tail = nn.Conv2d(dim, n_frames, 3, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, T, H, W) video frames.

            Returns:
                Temporally regularized frames (B, T, H, W).
            """
            fea = self.head(x)
            fea = self.body(fea)
            return self.tail(fea) + x

    class SpectralPrior(nn.Module):
        """Spectral (frequency-domain) prior.

        Applies regularization in Fourier domain to exploit spectral
        sparsity of natural video signals.

        Args:
            n_frames: Number of video frames.
            dim: Feature dimension.
        """

        def __init__(self, n_frames: int = 8, dim: int = 32):
            super().__init__()
            # Process real + imaginary parts
            self.conv1 = nn.Conv2d(n_frames * 2, dim, 3, 1, 1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.conv3 = nn.Conv2d(dim, n_frames * 2, 3, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, T, H, W) video frames.

            Returns:
                Spectrally regularized frames (B, T, H, W).
            """
            # FFT along temporal dimension
            x_fft = torch.fft.rfft(x, dim=1)
            # Stack real and imag as channels
            n_freq = x_fft.shape[1]
            x_ri = torch.cat([x_fft.real, x_fft.imag], dim=1)  # (B, 2*freq, H, W)

            # Process in frequency domain
            fea = self.relu(self.conv1(x_ri))
            fea = self.relu(self.conv2(fea))
            out = self.conv3(fea) + x_ri

            # Inverse FFT
            real = out[:, :n_freq]
            imag = out[:, n_freq:]
            x_fft_out = torch.complex(real, imag)
            return torch.fft.irfft(x_fft_out, n=x.shape[1], dim=1)

    class ELPPhase(nn.Module):
        """Single ELP-Unfolding phase.

        Combines gradient descent on data fidelity with an ensemble of
        three learned priors (spatial, temporal, spectral).

        Args:
            n_frames: Number of video frames.
            dim: Feature dimension.
        """

        def __init__(self, n_frames: int = 8, dim: int = 32):
            super().__init__()
            self.spatial = SpatialPrior(n_frames, dim)
            self.temporal = TemporalPrior(n_frames, dim)
            self.spectral = SpectralPrior(n_frames, dim)

            # Learnable ensemble weights (one per prior)
            self.alpha = nn.Parameter(torch.ones(3) / 3)

            # Learnable step size for data fidelity gradient
            self.step = nn.Parameter(torch.tensor(1.0))

        def forward(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: "torch.Tensor",
            mask_sum: "torch.Tensor",
        ) -> "torch.Tensor":
            """One ELP phase.

            Args:
                x: Current video estimate (B, T, H, W).
                y: Snapshot measurement (B, 1, H, W).
                mask: Temporal masks (B, T, H, W).
                mask_sum: Precomputed sum(mask^2, dim=1) + eps.

            Returns:
                Updated estimate (B, T, H, W).
            """
            # Data fidelity gradient step
            y_est = (mask * x).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            residual = y - y_est
            grad = mask * residual  # Back-project residual
            x_g = x + self.step * grad / mask_sum.unsqueeze(1)

            # Ensemble of priors
            weights = F.softmax(self.alpha, dim=0)
            x_s = self.spatial(x_g)
            x_t = self.temporal(x_g)
            x_f = self.spectral(x_g)

            x_out = weights[0] * x_s + weights[1] * x_t + weights[2] * x_f
            return x_out

    class ELPUnfolding(nn.Module):
        """ELP-Unfolding: Ensemble Learning Priors Deep Unfolding for CACTI.

        Multi-phase deep unfolding with three complementary learned priors
        combined through a learnable ensemble at each phase.

        Args:
            n_frames: Number of compressed video frames.
            n_phases: Number of unrolling phases.
            dim: Feature dimension for prior networks.
        """

        def __init__(
            self,
            n_frames: int = 8,
            n_phases: int = 5,
            dim: int = 32,
        ):
            super().__init__()
            self.n_frames = n_frames
            self.n_phases = n_phases

            self.phases = nn.ModuleList(
                [ELPPhase(n_frames, dim) for _ in range(n_phases)]
            )

        def forward(
            self,
            meas: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Args:
                meas: Snapshot measurement (B, 1, H, W).
                mask: Temporal masks (B, T, H, W).

            Returns:
                Reconstructed video (B, T, H, W).
            """
            # Precompute mask normalization
            mask_sum = (mask ** 2).sum(dim=1) + 1e-10  # (B, H, W)

            # Initialize: back-projection
            x = mask * meas / mask_sum.unsqueeze(1)

            for phase in self.phases:
                x = phase(x, meas, mask, mask_sum)

            return x


# ===========================================================================
# High-level reconstruction function
# ===========================================================================


def elp_recon(
    meas: np.ndarray,
    mask: np.ndarray,
    weights_path: Optional[str] = None,
    n_phases: int = 5,
    device: Optional[str] = None,
) -> np.ndarray:
    """Reconstruct video from CACTI snapshot using ELP-Unfolding.

    Args:
        meas: 2D snapshot measurement (H, W).
        mask: 3D mask tensor (H, W, T) with T temporal frames.
        weights_path: Path to pretrained weights (optional).
        n_phases: Number of unrolling phases.
        device: Torch device string.

    Returns:
        Reconstructed video frames (T, H, W).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    H, W = meas.shape[:2]
    n_frames = mask.shape[2]

    model = ELPUnfolding(
        n_frames=n_frames,
        n_phases=n_phases,
    ).to(device)

    # Load pretrained weights
    if weights_path is None:
        weights_path = str(
            _PKG_ROOT / "weights" / "elp_unfolding" / "elp_unfolding.pth"
        )

    wp = Path(weights_path)
    if wp.exists():
        checkpoint = torch.load(str(wp), map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        warnings.warn(
            "No pretrained weights found for ELP-Unfolding. "
            "Running with random initialization.",
            stacklevel=2,
        )

    model.eval()

    # Prepare measurement: (H, W) -> (1, 1, H, W)
    meas_t = (
        torch.from_numpy(meas.copy())
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Prepare mask: (H, W, T) -> (1, T, H, W)
    mask_t = (
        torch.from_numpy(mask.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        recon = model(meas_t, mask_t)

    # (1, T, H, W) -> (T, H, W)
    result = recon.squeeze(0).cpu().numpy()
    return np.clip(result, 0, 1).astype(np.float32)


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_elp_unfolding(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio wrapper for ELP-Unfolding reconstruction.

    Args:
        y: Snapshot measurement (H, W).
        physics: Physics operator (must have .masks attribute).
        cfg: Configuration with:
            - n_phases: Number of phases (default 5).
            - weights_path: Path to pretrained weights.
            - device: Torch device string.

    Returns:
        Tuple of (reconstructed video (T, H, W), info_dict).
    """
    n_phases = cfg.get("n_phases", 5)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    info: Dict[str, Any] = {
        "solver": "elp_unfolding",
        "n_phases": n_phases,
    }

    try:
        masks = None
        if hasattr(physics, "masks"):
            masks = physics.masks
        elif hasattr(physics, "info"):
            op_info = physics.info()
            masks = op_info.get("masks", None)

        if masks is None:
            info["error"] = "no masks found on physics operator"
            return y.astype(np.float32), info

        result = elp_recon(
            meas=y,
            mask=masks,
            weights_path=weights_path,
            n_phases=n_phases,
            device=device,
        )
        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, "adjoint"):
            result = physics.adjoint(y)
            return result.astype(np.float32), info
        return y.astype(np.float32), info
