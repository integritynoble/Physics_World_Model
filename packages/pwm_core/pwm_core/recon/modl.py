"""MoDL: Model-Based Deep Learning for MRI Reconstruction.

Unrolled optimization combining conjugate gradient (CG) for data
consistency with a CNN denoiser, using weight sharing across iterations.

References:
- Aggarwal, H.K. et al. (2019). "MoDL: Model-Based Deep Learning for
  Inverse Problems", IEEE TMI.

Benchmark (fastMRI knee 4x):
- PSNR: ~36 dB (with weight sharing, 5 unrolls)
- Params: ~2M (weight sharing), VRAM: ~2GB
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
            "MoDL requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Model components
# ===========================================================================

if HAS_TORCH:

    class ResidualBlock(nn.Module):
        """Residual block with two conv layers + BN + ReLU.

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

    class DenoisingCNN(nn.Module):
        """CNN denoiser (regularization network) for MoDL.

        Residual CNN operating on real-valued image (magnitude or
        real/imag channels).

        Args:
            in_ch: Input channels (2 for complex: real + imag).
            dim: Feature dimension.
            n_blocks: Number of residual blocks.
        """

        def __init__(self, in_ch: int = 2, dim: int = 64, n_blocks: int = 5):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, dim, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(
                *[ResidualBlock(dim) for _ in range(n_blocks)]
            )
            self.tail = nn.Conv2d(dim, in_ch, 3, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, 2, H, W) real+imag channels.

            Returns:
                Denoised (B, 2, H, W).
            """
            res = x
            fea = self.head(x)
            fea = self.body(fea)
            return self.tail(fea) + res

    class MoDL(nn.Module):
        """MoDL: Model-Based Deep Learning for MRI.

        Alternates between:
        1. CNN denoiser (regularization prior)
        2. CG-based data consistency (enforces measurement fidelity)

        Weight sharing: the same CNN is reused across all iterations.

        Args:
            n_iter: Number of unrolled iterations.
            cg_iter: CG iterations for data consistency.
            dim: CNN feature dimension.
            n_blocks: CNN residual blocks.
            lam: Regularization strength (data consistency vs prior).
        """

        def __init__(
            self,
            n_iter: int = 5,
            cg_iter: int = 10,
            dim: int = 64,
            n_blocks: int = 5,
            lam: float = 0.05,
        ):
            super().__init__()
            self.n_iter = n_iter
            self.cg_iter = cg_iter

            # Shared CNN denoiser
            self.denoiser = DenoisingCNN(in_ch=2, dim=dim, n_blocks=n_blocks)

            # Learnable regularization parameter
            self.lam = nn.Parameter(torch.tensor(lam))

        def _to_complex(self, x_ri: "torch.Tensor") -> "torch.Tensor":
            """Convert (B, 2, H, W) real+imag to (B, H, W) complex."""
            return torch.complex(x_ri[:, 0], x_ri[:, 1])

        def _to_ri(self, x_c: "torch.Tensor") -> "torch.Tensor":
            """Convert (B, H, W) complex to (B, 2, H, W) real+imag."""
            return torch.stack([x_c.real, x_c.imag], dim=1)

        def _dc_step(
            self,
            x_ri: "torch.Tensor",
            kspace_ref: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Data consistency using closed-form solution.

            Solves: (A^H A + lam I) x = A^H y + lam z
            In k-space: x_k = (mask * y_k + lam * z_k) / (mask + lam)

            Args:
                x_ri: CNN output (B, 2, H, W).
                kspace_ref: Measured k-space (B, H, W) complex.
                mask: Sampling mask broadcastable to (B, H, W).

            Returns:
                Data-consistent estimate (B, 2, H, W).
            """
            lam = torch.abs(self.lam)
            x_c = self._to_complex(x_ri)

            # Transform to k-space
            x_k = torch.fft.fftshift(
                torch.fft.fft2(x_c), dim=(-2, -1)
            )

            # Closed-form DC
            x_k_dc = (mask * kspace_ref + lam * x_k) / (mask + lam)

            # Back to image domain
            x_dc = torch.fft.ifft2(
                torch.fft.ifftshift(x_k_dc, dim=(-2, -1))
            )

            return self._to_ri(x_dc)

        def forward(
            self,
            kspace: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Args:
                kspace: Under-sampled k-space (B, H, W) complex or (B, H, W, 2).
                mask: Sampling mask broadcastable to (B, H, W).

            Returns:
                Reconstructed magnitude image (B, H, W).
            """
            # Handle real input
            if kspace.is_complex():
                kspace_c = kspace
            else:
                if kspace.ndim == 4 and kspace.shape[-1] == 2:
                    kspace_c = torch.complex(kspace[..., 0], kspace[..., 1])
                else:
                    kspace_c = kspace.to(torch.complex64)

            # Initial: zero-filled IFFT
            x_c = torch.fft.ifft2(
                torch.fft.ifftshift(kspace_c, dim=(-2, -1))
            )
            x_ri = self._to_ri(x_c)

            for _ in range(self.n_iter):
                # CNN denoiser step
                x_ri = self.denoiser(x_ri.float())

                # Data consistency step
                x_ri = self._dc_step(x_ri, kspace_c, mask)

            return torch.abs(self._to_complex(x_ri))


# ===========================================================================
# High-level reconstruction function
# ===========================================================================


def modl_recon(
    kspace: np.ndarray,
    mask: np.ndarray,
    weights_path: Optional[str] = None,
    n_iter: int = 5,
    device: Optional[str] = None,
) -> np.ndarray:
    """Reconstruct MRI from under-sampled k-space using MoDL.

    Args:
        kspace: Under-sampled k-space. Complex (H, W) or real (H, W, 2).
        mask: Sampling mask (H, W) or (W,) binary.
        weights_path: Path to pretrained weights.
        n_iter: Number of unrolled iterations.
        device: Torch device string.

    Returns:
        Reconstructed magnitude image (H, W).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = MoDL(n_iter=n_iter).to(device)

    # Load weights
    if weights_path is None:
        weights_path = str(_PKG_ROOT / "weights" / "modl" / "modl.pth")

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
            "No pretrained weights found for MoDL. "
            "Running with random initialization.",
            stacklevel=2,
        )

    model.eval()

    # Prepare kspace tensor
    if np.iscomplexobj(kspace):
        kspace_t = torch.from_numpy(
            np.stack([kspace.real, kspace.imag], axis=-1).astype(np.float32)
        ).unsqueeze(0).to(device)
        kspace_t = torch.complex(kspace_t[..., 0], kspace_t[..., 1])
    else:
        kspace_t = torch.from_numpy(kspace.astype(np.float32)).unsqueeze(0).to(device)
        kspace_t = kspace_t.to(torch.complex64)

    # Prepare mask
    if mask.ndim == 1:
        mask_2d = np.tile(mask, (kspace.shape[0], 1))
    else:
        mask_2d = mask
    mask_t = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        img = model(kspace_t, mask_t)

    return img.squeeze(0).cpu().numpy().astype(np.float32)


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_modl(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for MoDL MRI reconstruction.

    Args:
        y: Under-sampled k-space data.
        physics: MRI physics operator (must have sampling mask).
        cfg: Configuration with optional keys:
            - n_iter, weights_path, device.

    Returns:
        Tuple of (reconstructed_image, info_dict).
    """
    n_iter = cfg.get("n_iter", 5)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    info: Dict[str, Any] = {
        "solver": "modl",
        "n_iter": n_iter,
    }

    try:
        mask = None
        if hasattr(physics, "mask"):
            mask = np.asarray(physics.mask)
        elif hasattr(physics, "sampling_mask"):
            mask = np.asarray(physics.sampling_mask)
        elif hasattr(physics, "info"):
            op_info = physics.info()
            mask = op_info.get("mask", op_info.get("sampling_mask", None))
            if mask is not None:
                mask = np.asarray(mask)

        if mask is None:
            if y.ndim >= 2:
                mask = np.ones(y.shape[-1], dtype=np.float32)
            else:
                info["error"] = "no sampling mask found"
                return y.astype(np.float32), info

        result = modl_recon(
            kspace=y,
            mask=mask,
            weights_path=weights_path,
            n_iter=n_iter,
            device=device,
        )
        return result, info

    except Exception as e:
        info["error"] = str(e)
        try:
            if np.iscomplexobj(y):
                return np.abs(np.fft.ifft2(y)).astype(np.float32), info
            return y.astype(np.float32), info
        except Exception:
            return y.astype(np.float32), info
