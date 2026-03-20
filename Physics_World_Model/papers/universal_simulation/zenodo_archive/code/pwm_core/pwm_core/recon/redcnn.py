"""RED-CNN: Residual Encoder-Decoder CNN for Low-Dose CT.

References:
- Chen, H. et al. (2017). "Low-Dose CT with a Residual Encoder-Decoder
  Convolutional Neural Network", IEEE TMI.

Benchmark (Mayo Low-Dose CT):
- PSNR: ~42 dB (1mm), ~38 dB (3mm)
- Params: 1.5M, VRAM: <0.5GB
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "RED-CNN requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Package root for default weight paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Model
# ===========================================================================


if HAS_TORCH:

    class REDCNN(nn.Module):
        """Residual Encoder-Decoder CNN for low-dose CT denoising.

        Architecture follows Chen et al. (2017):
        - 5-layer encoder with Conv2d(96, k=5) and ReLU
        - 5-layer decoder with ConvTranspose2d mirroring the encoder
        - Skip connections between encoder and decoder at each level
        - Global residual learning: output = input + network(input)

        Input:  (B, 1, H, W)
        Output: (B, 1, H, W)
        Params: ~1.5M
        """

        def __init__(self):
            super().__init__()

            # Encoder
            self.enc1 = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=2)
            self.enc2 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
            self.enc3 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
            self.enc4 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
            self.enc5 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)

            # Decoder (ConvTranspose2d mirrors the encoder)
            self.dec5 = nn.ConvTranspose2d(
                96, 96, kernel_size=5, stride=1, padding=2
            )
            self.dec4 = nn.ConvTranspose2d(
                96, 96, kernel_size=5, stride=1, padding=2
            )
            self.dec3 = nn.ConvTranspose2d(
                96, 96, kernel_size=5, stride=1, padding=2
            )
            self.dec2 = nn.ConvTranspose2d(
                96, 96, kernel_size=5, stride=1, padding=2
            )
            self.dec1 = nn.ConvTranspose2d(
                96, 1, kernel_size=5, stride=1, padding=2
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass with residual learning.

            Args:
                x: Input image tensor (B, 1, H, W).

            Returns:
                Denoised image tensor (B, 1, H, W).
            """
            residual = x

            # Encoder
            e1 = self.relu(self.enc1(x))
            e2 = self.relu(self.enc2(e1))
            e3 = self.relu(self.enc3(e2))
            e4 = self.relu(self.enc4(e3))
            e5 = self.relu(self.enc5(e4))

            # Decoder with skip connections
            d5 = self.relu(self.dec5(e5) + e4)
            d4 = self.relu(self.dec4(d5) + e3)
            d3 = self.relu(self.dec3(d4) + e2)
            d2 = self.relu(self.dec2(d3) + e1)
            d1 = self.dec1(d2)

            # Global residual
            out = d1 + residual
            return out


# ===========================================================================
# High-level denoise function
# ===========================================================================


def redcnn_denoise(
    image: np.ndarray,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Denoise a CT image using RED-CNN.

    Args:
        image: 2D grayscale image (H, W), float32, any range.
        weights_path: Path to pretrained ``redcnn.pth``.  When *None* the
            default location ``{pkg_root}/weights/redcnn/redcnn.pth`` is
            tried.
        device: Torch device string.  Defaults to CUDA if available.

    Returns:
        Denoised image (H, W), same dtype as input.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Resolve weight path
    if weights_path is None:
        weights_path = str(_PKG_ROOT / "weights" / "redcnn" / "redcnn.pth")

    # Build model
    model = REDCNN().to(device)

    # Load weights if they exist
    wp = Path(weights_path)
    if wp.exists():
        state = torch.load(
            str(wp), map_location=device, weights_only=False
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = {
                k.replace("module.", ""): v
                for k, v in state["state_dict"].items()
            }
        model.load_state_dict(state, strict=False)

    model.eval()

    # Prepare input: (H, W) -> (1, 1, H, W)
    x = (
        torch.from_numpy(image.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = model(x)

    return out.squeeze().cpu().numpy().astype(image.dtype)


def redcnn_train_quick(
    noisy: np.ndarray,
    clean: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> "REDCNN":
    """Quick-train RED-CNN on a (noisy_FBP, clean_CT) pair.

    Args:
        noisy: Noisy/FBP reconstruction (H, W), float32.
        clean: Clean ground truth (H, W), float32.
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device string.

    Returns:
        Trained REDCNN model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = REDCNN().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    y = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        # Reduce LR at halfway point
        if epoch == epochs // 2:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1

    model.eval()
    return model


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_redcnn(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for RED-CNN CT denoising.

    Args:
        y: Noisy/low-dose CT image or reconstruction (H, W).
        physics: Physics operator (unused for post-processing denoiser,
            but kept for portfolio interface consistency).
        cfg: Configuration dict with optional keys:
            - weights_path: path to ``redcnn.pth``
            - device: torch device string

    Returns:
        Tuple of (denoised_image, info_dict).
    """
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    info: Dict[str, Any] = {"solver": "redcnn"}

    try:
        result = redcnn_denoise(
            image=y, weights_path=weights_path, device=device
        )
        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
