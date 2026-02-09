"""FlatNet: Photorealistic Lensless Image Reconstruction.

Two-stage (inversion + U-Net refinement) for lensless cameras.

References:
- Khan, S.S. et al. (2020). "FlatNet: Towards Photorealistic Scene
  Reconstruction from Lensless Measurements", IEEE TPAMI.

Benchmark:
- Photorealistic quality from DiffuserCam measurements
- Params: ~59M, VRAM: ~4GB (inference 256x256)
"""

from __future__ import annotations

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
            "FlatNet requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Weight path default
# ============================================================================

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "flatnet" / "flatnet.pth"


# ============================================================================
# Stage 1: Inversion Layer (learned Wiener-like filter)
# ============================================================================

if HAS_TORCH:

    class InversionLayer(nn.Module):
        """Learned Wiener-like inversion in Fourier domain.

        Learns a complex-valued filter that inverts the PSF transfer
        function, producing an initial (noisy) reconstruction from the
        lensless measurement.

        Args:
            spatial_size: (H, W) spatial dimensions of input.
            out_channels: number of output channels (3 for RGB).
        """

        def __init__(
            self,
            spatial_size: Tuple[int, int] = (256, 256),
            out_channels: int = 3,
        ):
            super().__init__()
            H, W = spatial_size
            # Learnable filter in rfft2 frequency domain.
            # rfft2 output has shape (H, W//2+1) complex values.
            freq_h, freq_w = H, W // 2 + 1
            # One filter per output channel, applied to single-channel input.
            self.filter_real = nn.Parameter(
                torch.randn(out_channels, freq_h, freq_w) * 0.02
            )
            self.filter_imag = nn.Parameter(
                torch.randn(out_channels, freq_h, freq_w) * 0.02
            )
            self.out_channels = out_channels
            self.spatial_size = spatial_size

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Apply learned Wiener-like inversion.

            Args:
                x: measurement tensor [B, 1, H, W].

            Returns:
                Initial reconstruction [B, out_channels, H, W].
            """
            # Squeeze channel dim for rfft2
            x_2d = x[:, 0, :, :]  # [B, H, W]

            # Forward FFT
            X_freq = torch.fft.rfft2(x_2d)  # [B, H, W//2+1] complex

            # Apply learned filter per output channel
            # filter_complex: [C, H, W//2+1]
            filt = torch.complex(self.filter_real, self.filter_imag)

            # Broadcast multiply: [B, 1, H, W'] * [1, C, H, W'] -> [B, C, H, W']
            Y_freq = X_freq.unsqueeze(1) * filt.unsqueeze(0)

            # Inverse FFT
            y = torch.fft.irfft2(Y_freq, s=self.spatial_size)  # [B, C, H, W]
            return y

    # ========================================================================
    # Stage 2: Perceptual U-Net refinement
    # ========================================================================

    class _ResBlock(nn.Module):
        """Residual convolutional block with two 3x3 convs + BN + ReLU."""

        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.relu(self.block(x) + x)

    class _DownBlock(nn.Module):
        """Encoder block: conv + residual block + max-pool."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.res = _ResBlock(out_ch)
            self.pool = nn.MaxPool2d(2)

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Returns (pooled_feature, skip_connection)."""
            feat = self.res(self.conv(x))
            return self.pool(feat), feat

    class _UpBlock(nn.Module):
        """Decoder block: upsample + concat skip + conv + residual block."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            # in_ch comes from concat of upsampled + skip
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.res = _ResBlock(out_ch)

        def forward(
            self, x: "torch.Tensor", skip: "torch.Tensor"
        ) -> "torch.Tensor":
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.res(self.conv(x))

    class PerceptualUNet(nn.Module):
        """Large U-Net for perceptual refinement of initial reconstruction.

        Architecture follows FlatNet paper: 5-level U-Net with residual
        blocks and channel progression [64, 128, 256, 512, 1024].

        Args:
            in_channels: input channels (matches InversionLayer output).
            out_channels: output channels (3 for RGB).
            base_ch: base channel count (64).

        Total params: ~55M.
        """

        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            base_ch: int = 64,
        ):
            super().__init__()
            ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
            # ch = [64, 128, 256, 512, 1024]

            # Encoder
            self.down1 = _DownBlock(in_channels, ch[0])
            self.down2 = _DownBlock(ch[0], ch[1])
            self.down3 = _DownBlock(ch[1], ch[2])
            self.down4 = _DownBlock(ch[2], ch[3])

            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv2d(ch[3], ch[4], 3, padding=1, bias=False),
                nn.BatchNorm2d(ch[4]),
                nn.ReLU(inplace=True),
                _ResBlock(ch[4]),
            )

            # Decoder (concat doubles channels before conv)
            self.up4 = _UpBlock(ch[4] + ch[3], ch[3])
            self.up3 = _UpBlock(ch[3] + ch[2], ch[2])
            self.up2 = _UpBlock(ch[2] + ch[1], ch[1])
            self.up1 = _UpBlock(ch[1] + ch[0], ch[0])

            # Output head
            self.head = nn.Conv2d(ch[0], out_channels, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Refine initial reconstruction.

            Args:
                x: initial reconstruction [B, in_channels, H, W].

            Returns:
                Refined image [B, out_channels, H, W].
            """
            # Encoder
            d1, s1 = self.down1(x)
            d2, s2 = self.down2(d1)
            d3, s3 = self.down3(d2)
            d4, s4 = self.down4(d3)

            # Bottleneck
            bn = self.bottleneck(d4)

            # Decoder
            u4 = self.up4(bn, s4)
            u3 = self.up3(u4, s3)
            u2 = self.up2(u3, s2)
            u1 = self.up1(u2, s1)

            return self.head(u1)

    # ========================================================================
    # Full FlatNet model
    # ========================================================================

    class FlatNet(nn.Module):
        """FlatNet: two-stage lensless image reconstruction.

        Stage 1 (InversionLayer): learned Wiener-like deconvolution in
            Fourier domain, producing an initial noisy RGB estimate.
        Stage 2 (PerceptualUNet): large U-Net that refines the initial
            estimate into a photorealistic RGB image.

        Args:
            spatial_size: (H, W) expected input spatial dimensions.
            out_channels: number of output channels (3 for RGB).
            base_ch: base channel count for U-Net (64).

        Total params: ~59M (4M inversion + 55M U-Net).
        """

        def __init__(
            self,
            spatial_size: Tuple[int, int] = (256, 256),
            out_channels: int = 3,
            base_ch: int = 64,
        ):
            super().__init__()
            self.inversion = InversionLayer(
                spatial_size=spatial_size, out_channels=out_channels
            )
            self.refinement = PerceptualUNet(
                in_channels=out_channels,
                out_channels=out_channels,
                base_ch=base_ch,
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Full two-stage reconstruction.

            Args:
                x: lensless measurement [B, 1, H, W].

            Returns:
                Reconstructed RGB image [B, 3, H, W].
            """
            x_init = self.inversion(x)
            x_refined = self.refinement(x_init)
            return x_refined


# ============================================================================
# High-level reconstruction function
# ============================================================================


def flatnet_reconstruct(
    measurement: np.ndarray,
    psf: Optional[np.ndarray] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    spatial_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Reconstruct an RGB image from a lensless measurement using FlatNet.

    Args:
        measurement: 2D lensless sensor measurement [H, W].
        psf: point spread function (unused by the network directly, but
             kept for API consistency with other solvers). The PSF
             information is implicitly learned in InversionLayer weights.
        weights_path: path to pretrained ``flatnet.pth`` checkpoint.
            Defaults to ``{pkg_root}/weights/flatnet/flatnet.pth``.
        device: torch device string (e.g. ``"cuda:0"``). Auto-detected
            if not provided.
        spatial_size: (H, W) to resize measurement before inference.
            If None, uses the measurement's native resolution.

    Returns:
        Reconstructed RGB image [H, W, 3] in float32, clipped to [0, 1].
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Determine spatial size
    if spatial_size is None:
        spatial_size = (measurement.shape[0], measurement.shape[1])

    # Build model
    model = FlatNet(spatial_size=spatial_size, out_channels=3).to(device)

    # Load weights
    wpath = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    if wpath.exists():
        checkpoint = torch.load(str(wpath), map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Prepare input tensor [1, 1, H, W]
    meas = measurement.astype(np.float32)
    meas_t = torch.from_numpy(meas).unsqueeze(0).unsqueeze(0).to(device)

    # Resize if measurement does not match spatial_size
    if meas_t.shape[2:] != torch.Size(spatial_size):
        meas_t = F.interpolate(
            meas_t, size=spatial_size, mode="bilinear", align_corners=False
        )

    # Inference
    with torch.no_grad():
        recon = model(meas_t)

    # Convert to numpy [H, W, 3]
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Wiener initialization & quick training
# ============================================================================


def flatnet_init_wiener(
    model: "FlatNet",
    psf: np.ndarray,
    reg_param: float = 0.01,
) -> None:
    """Initialize FlatNet InversionLayer from a Wiener filter.

    Computes H = FFT(psf), sets filter = conj(H) / (|H|^2 + lambda).

    Args:
        model: FlatNet model to initialize.
        psf: Point spread function (H, W), float32.
        reg_param: Regularization parameter for Wiener filter.
    """
    _require_torch()

    H, W = model.inversion.spatial_size
    psf_resized = np.zeros((H, W), dtype=np.float32)
    ph, pw = psf.shape[:2]
    psf_resized[:min(ph, H), :min(pw, W)] = psf[:min(ph, H), :min(pw, W)]
    psf_resized /= psf_resized.sum() + 1e-10

    H_freq = np.fft.rfft2(psf_resized)
    wiener = np.conj(H_freq) / (np.abs(H_freq) ** 2 + reg_param)

    with torch.no_grad():
        for c in range(model.inversion.out_channels):
            model.inversion.filter_real.data[c] = torch.from_numpy(wiener.real.astype(np.float32))
            model.inversion.filter_imag.data[c] = torch.from_numpy(wiener.imag.astype(np.float32))


def flatnet_train_quick(
    measurement: np.ndarray,
    clean: np.ndarray,
    psf: Optional[np.ndarray] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> "FlatNet":
    """Quick-train FlatNet on a (measurement, clean) pair.

    If psf is provided, initializes InversionLayer with Wiener filter.

    Args:
        measurement: Lensless measurement (H, W), float32.
        clean: Clean ground truth (H, W), float32 in [0, 1].
        psf: Point spread function (H, W) for Wiener init.
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device string.

    Returns:
        Trained FlatNet model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    H, W = measurement.shape[:2]
    model = FlatNet(spatial_size=(H, W), out_channels=1, base_ch=32).to(device)

    if psf is not None:
        flatnet_init_wiener(model, psf)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    x = torch.from_numpy(measurement.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    y = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        # Take first channel if multi-channel output
        if pred.shape[1] > 1:
            pred = pred[:, :1, :, :]
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if epoch == epochs // 2:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1

    model.eval()
    return model


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_flatnet(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run FlatNet lensless reconstruction (portfolio interface).

    Args:
        y: lensless measurement array [H, W].
        physics: physics operator (should expose ``.psf`` attribute).
        cfg: configuration dict with optional keys:
            - weights_path: path to checkpoint file.
            - device: torch device string.
            - spatial_size: (H, W) tuple for model input resolution.

    Returns:
        Tuple of (reconstructed_image [H, W, 3], info_dict).
    """
    info: Dict[str, Any] = {"solver": "flatnet"}

    psf = None
    if hasattr(physics, "psf"):
        psf = physics.psf
    elif hasattr(physics, "info"):
        op_info = physics.info()
        if "psf" in op_info:
            psf = op_info["psf"]

    try:
        result = flatnet_reconstruct(
            measurement=y,
            psf=psf,
            weights_path=cfg.get("weights_path"),
            device=cfg.get("device"),
            spatial_size=cfg.get("spatial_size"),
        )
        info["status"] = "success"
        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
