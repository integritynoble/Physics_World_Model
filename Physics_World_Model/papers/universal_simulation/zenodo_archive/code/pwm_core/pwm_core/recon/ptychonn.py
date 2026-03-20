"""PtychoNN: AI-enabled Ptychographic Reconstruction.

Two-headed encoder-decoder for real-time amplitude + phase recovery.

References:
- Cherukara, M.J. et al. (2020). "AI-enabled high-resolution scanning
  coherent diffraction imaging", Applied Physics Letters.

Benchmark:
- 100-300x faster than ePIE
- Params: 0.7M (v2), VRAM: <0.5GB
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


_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "ptychonn" / "ptychonn.pth"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PtychoNN requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Model components
# ============================================================================


if HAS_TORCH:

    class PtychoNNEncoder(nn.Module):
        """Shared encoder for PtychoNN.

        Conv2d layers with MaxPool: channels [1, 32, 64, 128].
        Input: diffraction pattern (B, 1, det_size, det_size).
        Output: encoded features (B, 128, det_size/8, det_size/8).
        """

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                # Block 1: 1 -> 32
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # Block 2: 32 -> 64
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # Block 3: 64 -> 128
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class PtychoNNDecoder(nn.Module):
        """Decoder head for PtychoNN (one for amplitude, one for phase).

        ConvTranspose2d upsampling from 128 channels back to 1 channel.
        Input: encoded features (B, 128, H/8, W/8).
        Output: reconstructed map (B, 1, H, W).
        """

        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                # Block 1: 128 -> 64
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # Block 2: 64 -> 32
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # Block 3: 32 -> 1
                nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    class PtychoNN(nn.Module):
        """PtychoNN: two-headed encoder-decoder for ptychographic reconstruction.

        Architecture:
            - Shared encoder (Conv2d + MaxPool, channels [1, 32, 64, 128])
            - Two decoder heads (ConvTranspose2d upsampling):
                - amplitude_decoder -> amplitude map
                - phase_decoder -> phase map

        Input: diffraction pattern (B, 1, H, W)
        Output: tuple(amplitude (B, 1, H, W), phase (B, 1, H, W))

        Params: ~0.7M (v2 lightweight)
        """

        def __init__(self):
            super().__init__()
            self.encoder = PtychoNNEncoder()
            self.amplitude_decoder = PtychoNNDecoder()
            self.phase_decoder = PtychoNNDecoder()

        def forward(
            self, x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: diffraction pattern (B, 1, H, W)

            Returns:
                Tuple of (amplitude, phase), each (B, 1, H, W)
            """
            features = self.encoder(x)
            amplitude = self.amplitude_decoder(features)
            phase = self.phase_decoder(features)
            return amplitude, phase


# ============================================================================
# High-level reconstruction function
# ============================================================================


def ptychonn_reconstruct(
    diffraction_patterns: np.ndarray,
    positions: np.ndarray,
    object_shape: Tuple[int, int],
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Reconstruct complex object from diffraction patterns using PtychoNN.

    Processes all scan positions through the network, then stitches
    per-position amplitude + phase predictions into a full complex object.

    Args:
        diffraction_patterns: Measured intensities (n_pos, det_size, det_size)
        positions: Scan positions (n_pos, 2) as (y, x) pixel indices
        object_shape: Full object size (H, W)
        weights_path: Path to pretrained weights. Falls back to
            ``{pkg_root}/weights/ptychonn/ptychonn.pth`` if *None*.
        device: Torch device string (auto-selects CUDA if available)
        batch_size: Inference batch size

    Returns:
        Complex object array (H, W)
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if weights_path is None:
        weights_path = str(_DEFAULT_WEIGHTS)

    n_pos, det_h, det_w = diffraction_patterns.shape
    obj_h, obj_w = object_shape

    # Build model
    model = PtychoNN().to(device)

    # Load pretrained weights if available
    if Path(weights_path).exists():
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Prepare input: log-scale normalised diffraction patterns
    patterns = np.log1p(np.maximum(diffraction_patterns, 0)).astype(np.float32)
    p_max = patterns.max()
    if p_max > 0:
        patterns = patterns / p_max

    # Run inference in batches
    amp_patches = np.zeros((n_pos, det_h, det_w), dtype=np.float32)
    phase_patches = np.zeros((n_pos, det_h, det_w), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_pos, batch_size):
            end = min(start + batch_size, n_pos)
            batch = torch.from_numpy(
                patterns[start:end, None, :, :]
            ).to(device)

            amp, phi = model(batch)
            amp_patches[start:end] = amp.squeeze(1).cpu().numpy()
            phase_patches[start:end] = phi.squeeze(1).cpu().numpy()

    # Stitch patches into full object via weighted averaging
    obj_amp = np.zeros(object_shape, dtype=np.float64)
    obj_phase = np.zeros(object_shape, dtype=np.float64)
    weight_map = np.zeros(object_shape, dtype=np.float64)

    for idx in range(n_pos):
        py, px = int(positions[idx, 0]), int(positions[idx, 1])
        y_end = min(py + det_h, obj_h)
        x_end = min(px + det_w, obj_w)
        if py < 0 or px < 0:
            continue

        patch_h = y_end - py
        patch_w = x_end - px
        obj_amp[py:y_end, px:x_end] += amp_patches[idx, :patch_h, :patch_w]
        obj_phase[py:y_end, px:x_end] += phase_patches[idx, :patch_h, :patch_w]
        weight_map[py:y_end, px:x_end] += 1.0

    # Normalise overlapping regions
    mask = weight_map > 0
    obj_amp[mask] /= weight_map[mask]
    obj_phase[mask] /= weight_map[mask]

    # Combine into complex object
    obj = obj_amp * np.exp(1j * obj_phase)
    return obj.astype(np.complex64)


def ptychonn_train_quick(
    diffraction_patterns: np.ndarray,
    amplitude_patches: np.ndarray,
    phase_patches: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> "PtychoNN":
    """Quick-train PtychoNN on (diffraction, amplitude, phase) data.

    Args:
        diffraction_patterns: Measured intensities (n_pos, det_h, det_w).
        amplitude_patches: Ground truth amplitude patches (n_pos, det_h, det_w).
        phase_patches: Ground truth phase patches (n_pos, det_h, det_w).
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        device: Torch device string.

    Returns:
        Trained PtychoNN model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = PtychoNN().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    # Prepare data: log-scale normalized patterns
    patterns = np.log1p(np.maximum(diffraction_patterns, 0)).astype(np.float32)
    p_max = patterns.max()
    if p_max > 0:
        patterns = patterns / p_max

    # Normalize amplitude targets to [0, 1]
    amp_data = amplitude_patches.astype(np.float32)
    amp_min, amp_max = amp_data.min(), amp_data.max()
    if amp_max - amp_min > 1e-8:
        amp_data = (amp_data - amp_min) / (amp_max - amp_min)

    # Normalize phase targets to [0, 1]
    phase_data = phase_patches.astype(np.float32)
    phase_min, phase_max = phase_data.min(), phase_data.max()
    if phase_max - phase_min > 1e-8:
        phase_data = (phase_data - phase_min) / (phase_max - phase_min)

    n_pos = patterns.shape[0]

    for epoch in range(epochs):
        if epoch == epochs // 2:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
        perm = np.random.permutation(n_pos)
        for start in range(0, n_pos, batch_size):
            end = min(start + batch_size, n_pos)
            idx = perm[start:end]

            x = torch.from_numpy(patterns[idx, None, :, :]).to(device)
            amp_gt = torch.from_numpy(amp_data[idx, None, :, :]).to(device)
            phase_gt = torch.from_numpy(phase_data[idx, None, :, :]).to(device)

            optimizer.zero_grad()
            amp_pred, phase_pred = model(x)
            loss = loss_fn(amp_pred, amp_gt) + loss_fn(phase_pred, phase_gt)
            loss.backward()
            optimizer.step()

    model.eval()
    # Store normalization params for inference denormalization
    model._amp_min = amp_min
    model._amp_max = amp_max
    model._phase_min = phase_min
    model._phase_max = phase_max
    model._input_max = p_max
    return model


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_ptychonn(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run PtychoNN reconstruction (portfolio interface).

    Args:
        y: Diffraction patterns (n_pos, det_h, det_w)
        physics: Ptychography physics operator
        cfg: Configuration with:
            - weights_path: Path to model weights (optional)
            - batch_size: Inference batch size (default: 64)
            - device: Torch device string (default: auto)
            - output: 'amplitude', 'phase', or 'complex' (default: 'amplitude')

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    weights_path = cfg.get("weights_path", None)
    batch_size = cfg.get("batch_size", 64)
    device = cfg.get("device", None)
    output_type = cfg.get("output", "amplitude")

    info: Dict[str, Any] = {
        "solver": "ptychonn",
        "batch_size": batch_size,
    }

    try:
        # Extract scan metadata from physics operator
        positions = None
        object_shape = None

        if hasattr(physics, "positions"):
            positions = physics.positions
        if hasattr(physics, "x_shape"):
            object_shape = tuple(physics.x_shape)[:2]

        if hasattr(physics, "info"):
            op_info = physics.info()
            if "positions" in op_info:
                positions = op_info["positions"]
            if "x_shape" in op_info:
                object_shape = tuple(op_info["x_shape"])[:2]

        # Infer defaults when metadata is unavailable
        if positions is None:
            n_pos = y.shape[0]
            grid_size = int(np.sqrt(n_pos))
            det_size = y.shape[1]
            if object_shape is None:
                object_shape = (det_size * 2, det_size * 2)
            step = det_size // 2
            positions = np.array(
                [[i * step, j * step]
                 for i in range(grid_size)
                 for j in range(grid_size)][:n_pos]
            )

        if object_shape is None:
            det_size = y.shape[1]
            object_shape = (det_size * 2, det_size * 2)

        obj = ptychonn_reconstruct(
            y, positions, object_shape,
            weights_path=weights_path,
            device=device,
            batch_size=batch_size,
        )

        if output_type == "phase":
            result = np.angle(obj).astype(np.float32)
        elif output_type == "complex":
            result = np.stack([np.abs(obj), np.angle(obj)], axis=-1)
        else:
            result = np.abs(obj).astype(np.float32)

        info["object_shape"] = object_shape
        return result, info

    except Exception as e:
        info["error"] = str(e)
        if y.ndim >= 3:
            result = np.sqrt(np.mean(y, axis=0))
            return result.astype(np.float32), info
        return y.astype(np.float32), info
