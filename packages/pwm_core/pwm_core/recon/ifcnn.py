"""IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network.

General-purpose CNN for multi-focus and multi-exposure image fusion.

References:
- Zhang, Y. et al. (2020). "IFCNN: A general image fusion framework based
  on convolutional neural network", Information Fusion.

Benchmark:
- Strong multi-focus fusion quality
- Params: ~0.3M, VRAM: <0.5GB
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


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("IFCNN requires PyTorch. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Default weight path
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "ifcnn" / "ifcnn.pth"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class IFCNN(nn.Module):
        """Image Fusion CNN (Zhang et al. 2020).

        A lightweight shared-encoder architecture that extracts features from
        each source image independently, fuses them via element-wise maximum,
        and reconstructs the fused result through a small decoder.

        Args:
            in_channels: Number of input channels per image (default: 1).
            fuse_mode: ``"max"`` for element-wise maximum (default) or
                ``"attention"`` for a learnable channel-attention gate.

        Shapes:
            - Input : list of tensors, each ``(B, in_channels, H, W)``
            - Output: ``(B, in_channels, H, W)``

        Parameters: ~0.3 M
        """

        def __init__(self, in_channels: int = 1, fuse_mode: str = "max"):
            super().__init__()
            self.fuse_mode = fuse_mode

            # --- shared feature extractor ---
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
            )

            # --- learnable attention (used when fuse_mode == "attention") ---
            if fuse_mode == "attention":
                self.attn = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 64),
                    nn.Sigmoid(),
                )
            else:
                self.attn = None

            # --- reconstruction decoder ---
            self.dec = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, in_channels, kernel_size=1),
            )

        # -- helpers ---------------------------------------------------------

        def _extract(self, x: "torch.Tensor") -> "torch.Tensor":
            """Shared feature extraction for a single source image."""
            return self.enc2(self.enc1(x))

        def _fuse_max(
            self, feats: List["torch.Tensor"]
        ) -> "torch.Tensor":
            """Element-wise maximum fusion."""
            stacked = torch.stack(feats, dim=0)  # (N, B, C, H, W)
            return stacked.max(dim=0).values

        def _fuse_attention(
            self, feats: List["torch.Tensor"]
        ) -> "torch.Tensor":
            """Learnable channel-attention fusion."""
            weights = []
            for f in feats:
                w = self.attn(f)  # (B, 64)
                weights.append(w.unsqueeze(-1).unsqueeze(-1))  # (B, 64, 1, 1)
            # softmax over source dimension
            w_stack = torch.stack(weights, dim=0)  # (N, B, 64, 1, 1)
            w_stack = torch.softmax(w_stack, dim=0)
            f_stack = torch.stack(feats, dim=0)
            return (w_stack * f_stack).sum(dim=0)

        # -- forward ---------------------------------------------------------

        def forward(
            self, images: List["torch.Tensor"]
        ) -> "torch.Tensor":
            """Fuse a list of source images.

            Args:
                images: list of tensors each shaped ``(B, C, H, W)``.

            Returns:
                Fused image ``(B, C, H, W)``.
            """
            feats = [self._extract(img) for img in images]

            if self.fuse_mode == "attention" and self.attn is not None:
                fused = self._fuse_attention(feats)
            else:
                fused = self._fuse_max(feats)

            return self.dec(fused)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def ifcnn_fuse(
    images: List[np.ndarray],
    weights_path: Optional[str] = None,
    fuse_mode: str = "max",
    device: Optional[str] = None,
) -> np.ndarray:
    """Fuse multiple grayscale images using IFCNN.

    Args:
        images: list of 2-D numpy arrays ``(H, W)``, value range [0, 1].
        weights_path: path to ``ifcnn.pth``.  If *None* the default package
            location is tried; if the file does not exist the model runs with
            random (untrained) weights.
        fuse_mode: ``"max"`` or ``"attention"``.
        device: torch device string (auto-selected if *None*).

    Returns:
        Fused image ``(H, W)`` as float32, clipped to [0, 1].
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if len(images) < 2:
        raise ValueError("ifcnn_fuse requires at least 2 input images")

    H, W = images[0].shape[:2]
    for img in images:
        if img.shape[:2] != (H, W):
            raise ValueError(
                f"All images must have the same spatial size; "
                f"got {img.shape[:2]} vs ({H}, {W})"
            )

    # Build model
    model = IFCNN(in_channels=1, fuse_mode=fuse_mode).to(device)

    # Load weights
    wp = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    if wp.exists():
        ckpt = torch.load(str(wp), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = {
                k.replace("module.", ""): v
                for k, v in ckpt["state_dict"].items()
            }
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)

    model.eval()

    # Prepare tensors: (H, W) -> (1, 1, H, W)
    tensors = []
    for img in images:
        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensors.append(t.to(device))

    with torch.no_grad():
        fused = model(tensors)

    out = fused.squeeze(0).squeeze(0).cpu().numpy()
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def ifcnn_train_quick(
    source_images: List[np.ndarray],
    ground_truth: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> "IFCNN":
    """Quick-train IFCNN on (multi-focus inputs, sharp ground truth).

    Args:
        source_images: List of source images, each (H, W) float32 in [0, 1].
        ground_truth: Sharp all-in-focus image (H, W) float32 in [0, 1].
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device string.

    Returns:
        Trained IFCNN model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = IFCNN(in_channels=1, fuse_mode="max").to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Prepare tensors
    tensors = [
        torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        for img in source_images
    ]
    gt_t = torch.from_numpy(ground_truth.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(tensors)
        loss = loss_fn(pred, gt_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------


def run_ifcnn(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for IFCNN image fusion.

    The function inspects *physics* and *cfg* to obtain the list of source
    images to fuse.  If the physics object carries a ``sources`` attribute
    (list of arrays), those are used directly; otherwise *y* is treated as
    the first source and ``cfg["sources"]`` supplies the remaining ones.

    Args:
        y: Measurement / first source image ``(H, W)``.
        physics: Physics operator (may carry ``sources`` attribute).
        cfg: Configuration dict with optional keys:

            - ``sources``: additional source images (list of ndarray).
            - ``weights_path``: path to IFCNN weights.
            - ``fuse_mode``: ``"max"`` or ``"attention"`` (default ``"max"``).
            - ``device``: torch device string.

    Returns:
        Tuple of ``(fused_image, info_dict)``.
    """
    info: Dict[str, Any] = {"solver": "ifcnn"}

    # Collect source images
    if hasattr(physics, "sources") and physics.sources is not None:
        sources: List[np.ndarray] = list(physics.sources)
    else:
        extra = cfg.get("sources", [])
        sources = [y] + list(extra)

    if len(sources) < 2:
        # Fallback: cannot fuse a single image, just return it.
        info["note"] = "fewer than 2 sources; returning input"
        return y.astype(np.float32), info

    fuse_mode = cfg.get("fuse_mode", "max")
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    try:
        fused = ifcnn_fuse(
            images=sources,
            weights_path=weights_path,
            fuse_mode=fuse_mode,
            device=device,
        )
        info["fuse_mode"] = fuse_mode
        info["num_sources"] = len(sources)
        return fused, info
    except Exception as exc:
        info["error"] = str(exc)
        return y.astype(np.float32), info
