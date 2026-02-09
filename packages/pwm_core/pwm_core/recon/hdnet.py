"""HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging.

Dual-domain (spatial + frequency) learning for CASSI reconstruction.

References:
- Hu, X. et al. (2022). "HDNet: High-resolution Dual-domain Learning for
  Spectral Compressive Imaging", CVPR 2022.

Benchmark (KAIST, 256x256x28):
- HDNet: 34.97 dB PSNR, 0.952 SSIM
- Params: 2.37M, FLOPs: 154.76G
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

try:
    from pwm_core.recon.mst import shift_back_meas_torch, shift_torch
except ImportError:
    shift_back_meas_torch = shift_torch = None


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("HDNet requires PyTorch. Install with: pip install torch")


# ============================================================================
# Model components
# ============================================================================


if HAS_TORCH:

    class ChannelAttention(nn.Module):
        """Squeeze-excitation style channel attention.

        Applies global average pooling followed by two FC layers to produce
        per-channel scaling factors, then re-weights the input features.

        Args:
            dim: Number of input channels.
            reduction: Channel reduction ratio for the bottleneck.
        """

        def __init__(self, dim: int, reduction: int = 8):
            super().__init__()
            mid = max(dim // reduction, 4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(dim, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, dim, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [B, C, H, W] -> [B, C, H, W]"""
            b, c, _, _ = x.shape
            w = self.pool(x).view(b, c)
            w = self.fc(w).view(b, c, 1, 1)
            return x * w

    class SpatialBlock(nn.Module):
        """Residual block with channel attention for spatial-domain processing.

        Two 3x3 convolutions with LeakyReLU activation, followed by channel
        attention and a residual connection.

        Args:
            dim: Number of channels.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.ca = ChannelAttention(dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [B, C, H, W] -> [B, C, H, W]"""
            residual = x
            out = self.act(self.conv1(x))
            out = self.conv2(out)
            out = self.ca(out)
            return out + residual

    class FrequencyBlock(nn.Module):
        """Frequency-domain processing block using FFT.

        Transforms features to the frequency domain via rfft2, applies
        learnable frequency-domain weights and a 1x1 convolution, then
        transforms back via irfft2.

        Args:
            dim: Number of channels.
        """

        def __init__(self, dim: int):
            super().__init__()
            # Learnable frequency-domain weight (applied to magnitude)
            self.freq_weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            # 1x1 conv in frequency domain (operates on real/imag concatenated)
            self.freq_conv = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0, bias=True)
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            # Post-IDCT 1x1 projection
            self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [B, C, H, W] -> [B, C, H, W]"""
            b, c, h, w = x.shape
            residual = x

            # Forward FFT (rfft2 as DCT approximation)
            x_freq = torch.fft.rfft2(x, norm="ortho")

            # Decompose into real and imaginary, apply learnable transform
            x_real = x_freq.real * self.freq_weight
            x_imag = x_freq.imag * self.freq_weight

            # Concatenate real/imag -> 1x1 conv -> split back
            x_cat = torch.cat([x_real, x_imag], dim=1)
            x_cat = self.act(self.freq_conv(x_cat))
            x_real2, x_imag2 = x_cat.chunk(2, dim=1)

            # Reconstruct complex tensor and inverse FFT
            x_freq2 = torch.complex(x_real2, x_imag2)
            x_spatial = torch.fft.irfft2(x_freq2, s=(h, w), norm="ortho")

            # Projection + residual
            out = self.proj(x_spatial)
            return out + residual

    class DualDomainBlock(nn.Module):
        """Combines spatial and frequency branches with 1x1 fusion conv.

        Runs the input through a SpatialBlock and a FrequencyBlock in
        parallel, concatenates their outputs, and fuses with a 1x1
        convolution to produce a single output of the same dimension.

        Args:
            dim: Number of channels.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.spatial = SpatialBlock(dim)
            self.frequency = FrequencyBlock(dim)
            self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [B, C, H, W] -> [B, C, H, W]"""
            s_out = self.spatial(x)
            f_out = self.frequency(x)
            fused = self.act(self.fusion(torch.cat([s_out, f_out], dim=1)))
            return fused

    class HDNet(nn.Module):
        """HDNet: High-resolution Dual-domain Learning for CASSI.

        Encoder-decoder architecture with dual-domain (spatial + frequency)
        attention blocks. Takes a shifted measurement initialization and
        mask as input and outputs the reconstructed HSI cube.

        Args:
            dim: Base feature dimension.
            n_blocks: Number of DualDomainBlocks in the body.
            nC: Number of spectral channels (28 for KAIST).
        """

        def __init__(self, dim: int = 64, n_blocks: int = 4, nC: int = 28):
            super().__init__()
            self.dim = dim
            self.nC = nC
            self.n_blocks = n_blocks

            # Input: nC channels (shifted meas init) + nC channels (mask3d)
            in_channels = nC * 2

            # Stem: project to feature space
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )

            # Encoder: downsample path
            self.encoder_blocks = nn.ModuleList()
            self.downsamples = nn.ModuleList()
            current_dim = dim
            n_enc = n_blocks // 2  # half blocks for encoder
            n_enc = max(n_enc, 1)

            for i in range(n_enc):
                self.encoder_blocks.append(DualDomainBlock(current_dim))
                next_dim = min(current_dim * 2, dim * 4)
                self.downsamples.append(
                    nn.Conv2d(current_dim, next_dim, 4, 2, 1, bias=False)
                )
                current_dim = next_dim

            # Bottleneck
            self.bottleneck = nn.Sequential(
                DualDomainBlock(current_dim),
                DualDomainBlock(current_dim),
            )

            # Decoder: upsample path
            self.upsamples = nn.ModuleList()
            self.fusions = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()

            for i in range(n_enc):
                prev_dim = current_dim
                next_dim = current_dim // 2 if current_dim > dim else dim
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        prev_dim, next_dim,
                        kernel_size=2, stride=2, padding=0, bias=False,
                    )
                )
                # Fusion: skip connection concat -> 1x1 conv
                self.fusions.append(
                    nn.Conv2d(next_dim * 2, next_dim, 1, 1, 0, bias=False)
                )
                self.decoder_blocks.append(DualDomainBlock(next_dim))
                current_dim = next_dim

            # Head: project back to nC channels
            self.head = nn.Sequential(
                nn.Conv2d(current_dim, current_dim, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(current_dim, nC, 3, 1, 1, bias=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: [B, nC*2, H, W] where first nC channels are the shifted
                   measurement initialization and last nC channels are mask3d.

            Returns:
                Reconstructed HSI: [B, nC, H, W]
            """
            b, c, h_inp, w_inp = x.shape

            # Pad to multiple of 2^n_enc for the encoder/decoder
            n_enc = len(self.encoder_blocks)
            pad_factor = 2 ** n_enc
            pad_h = (pad_factor - h_inp % pad_factor) % pad_factor
            pad_w = (pad_factor - w_inp % pad_factor) % pad_factor
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

            # Keep the measurement init for residual
            x_init = x[:, : self.nC, :, :]

            # Stem
            fea = self.stem(x)

            # Encoder
            enc_features = []
            for enc_block, down in zip(self.encoder_blocks, self.downsamples):
                fea = enc_block(fea)
                enc_features.append(fea)
                fea = down(fea)

            # Bottleneck
            fea = self.bottleneck(fea)

            # Decoder
            for i, (up, fusion, dec_block) in enumerate(
                zip(self.upsamples, self.fusions, self.decoder_blocks)
            ):
                fea = up(fea)
                skip = enc_features[n_enc - 1 - i]
                # Handle size mismatch from padding
                if fea.shape[2:] != skip.shape[2:]:
                    fea = F.interpolate(
                        fea, size=skip.shape[2:], mode="bilinear",
                        align_corners=False,
                    )
                fea = fusion(torch.cat([fea, skip], dim=1))
                fea = dec_block(fea)

            # Head + residual
            out = self.head(fea) + x_init

            # Remove padding
            return out[:, :, :h_inp, :w_inp]


# ============================================================================
# High-level reconstruction function
# ============================================================================


def _find_weights(weights_path: Optional[str]) -> Optional[str]:
    """Search standard locations for HDNet weights.

    Checks:
    1. Direct path if given.
    2. {pkg_root}/weights/hdnet/hdnet.pth

    Returns:
        Resolved path string, or None if not found.
    """
    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            return str(p)

    # Search relative to this package
    pkg_root = Path(__file__).resolve().parent.parent
    candidate = pkg_root / "weights" / "hdnet" / "hdnet.pth"
    if candidate.exists():
        return str(candidate)

    return None


def hdnet_recon_cassi(
    meas_2d: np.ndarray,
    mask_3d: np.ndarray,
    nC: int = 28,
    step: int = 2,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    dim: int = 64,
    n_blocks: int = 4,
) -> np.ndarray:
    """Reconstruct CASSI hyperspectral cube using HDNet.

    Args:
        meas_2d: 2D measurement [H, W_ext] where W_ext = W + (nC-1)*step.
        mask_3d: 3D mask [H, W, nC] (tiled coded aperture).
        nC: Number of spectral bands.
        step: CASSI dispersion step (pixels per band).
        weights_path: Path to pretrained weights (.pth). Searched in
            standard locations if not given directly.
        device: Torch device string (default: auto-detect).
        dim: Base feature dimension for HDNet (default: 64).
        n_blocks: Number of DualDomainBlocks (default: 4).

    Returns:
        Reconstructed HSI cube [H, W, nC], float32 in [0, 1].
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    H, W = mask_3d.shape[:2]

    # Build model
    model = HDNet(dim=dim, n_blocks=n_blocks, nC=nC).to(device)

    # Load pretrained weights if available
    resolved = _find_weights(weights_path)
    if resolved is not None:
        checkpoint = torch.load(resolved, map_location=device, weights_only=False)
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
            "HDNet: no pretrained weights found; using random initialization. "
            "Results will be poor without training.",
            stacklevel=2,
        )

    model.eval()

    # Prepare mask: [H, W, nC] -> [1, nC, H, W]
    mask_3d_t = (
        torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Prepare initial estimate from measurement via shift_back
    if shift_back_meas_torch is not None:
        meas_t = (
            torch.from_numpy(meas_2d.copy()).unsqueeze(0).float().to(device)
        )
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
        x_init = x_init / nC * 2  # Scaling consistent with MST
    else:
        # Fallback: naive per-band extraction (no dispersion correction)
        x_init_np = np.zeros((H, W, nC), dtype=np.float32)
        for i in range(nC):
            x_init_np[:, :, i] = meas_2d[:, step * i : step * i + W]
        x_init_np = x_init_np / nC * 2
        x_init = (
            torch.from_numpy(x_init_np.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .float()
            .to(device)
        )

    # Concatenate init + mask -> [B, nC*2, H, W]
    model_input = torch.cat([x_init, mask_3d_t], dim=1)

    # Forward pass
    with torch.no_grad():
        recon = model(model_input)

    # Convert to numpy [H, W, nC]
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(recon, 0, 1).astype(np.float32)


def hdnet_train_quick(
    measurements: list,
    ground_truths: list,
    masks_3d: list,
    nC: int = 28,
    step: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[str] = None,
    dim: int = 64,
    n_blocks: int = 4,
) -> "HDNet":
    """Quick-train HDNet on (measurement, ground_truth) pairs.

    Args:
        measurements: List of 2D measurements [H, W_ext].
        ground_truths: List of HSI cubes [H, W, nC].
        masks_3d: List of 3D masks [H, W, nC].
        nC: Number of spectral bands.
        step: CASSI dispersion step.
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device string.
        dim: Base feature dimension.
        n_blocks: Number of DualDomainBlocks.

    Returns:
        Trained HDNet model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = HDNet(dim=dim, n_blocks=n_blocks, nC=nC).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for meas, gt, mask in zip(measurements, ground_truths, masks_3d):
            H, W = mask.shape[:2]
            # Prepare mask: [H, W, nC] -> [1, nC, H, W]
            mask_t = torch.from_numpy(mask.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device)
            # Prepare initial estimate
            if shift_back_meas_torch is not None:
                meas_t = torch.from_numpy(meas.copy()).unsqueeze(0).float().to(device)
                x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
                x_init = x_init / nC * 2
            else:
                x_init_np = np.zeros((H, W, nC), dtype=np.float32)
                for i in range(nC):
                    x_init_np[:, :, i] = meas[:, step * i:step * i + W]
                x_init_np = x_init_np / nC * 2
                x_init = torch.from_numpy(x_init_np.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device)
            # Ground truth: [H, W, nC] -> [1, nC, H, W]
            gt_t = torch.from_numpy(gt.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device)
            # Concatenate init + mask
            model_input = torch.cat([x_init, mask_t], dim=1)
            optimizer.zero_grad()
            pred = model(model_input)
            loss = loss_fn(pred, gt_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_hdnet(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run HDNet CASSI reconstruction (portfolio integration wrapper).

    Extracts mask and measurement from the physics operator, runs HDNet,
    and returns the result with an info dict.

    Args:
        y: Measurements (2D CASSI snapshot).
        physics: Physics operator with mask / info attributes.
        cfg: Configuration with optional keys:
            - weights_path: Path to pretrained HDNet weights.
            - device: Torch device string.
            - dim: Feature dimension (default: 64).
            - n_blocks: Number of DualDomainBlocks (default: 4).
            - nC: Number of spectral bands (default: 28).
            - step: Dispersion step (default: 2).

    Returns:
        Tuple of (reconstructed HSI [H, W, nC], info_dict).
    """
    nC = cfg.get("nC", 28)
    step = cfg.get("step", 2)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    dim = cfg.get("dim", 64)
    n_blocks = cfg.get("n_blocks", 4)

    info: Dict[str, Any] = {
        "solver": "hdnet",
        "nC": nC,
        "step": step,
        "dim": dim,
        "n_blocks": n_blocks,
    }

    try:
        # Extract mask from physics operator
        mask_2d = None
        if hasattr(physics, "mask"):
            mask_2d = physics.mask
        elif hasattr(physics, "info"):
            op_info = physics.info()
            mask_2d = op_info.get("mask", None)

        if mask_2d is None:
            info["error"] = "no mask available from physics operator"
            return y.astype(np.float32), info

        # Build 3D mask: [H, W] -> [H, W, nC]
        if mask_2d.ndim == 2:
            mask_3d = np.tile(mask_2d[:, :, np.newaxis], (1, 1, nC))
        else:
            mask_3d = mask_2d  # Already 3D

        result = hdnet_recon_cassi(
            meas_2d=y,
            mask_3d=mask_3d,
            nC=nC,
            step=step,
            weights_path=weights_path,
            device=device,
            dim=dim,
            n_blocks=n_blocks,
        )

        info["shape"] = list(result.shape)
        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Fallback: adjoint if available
        if hasattr(physics, "adjoint"):
            result = physics.adjoint(y)
            return result.astype(np.float32), info
        return y.astype(np.float32), info
