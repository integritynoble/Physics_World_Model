"""CASSI Differentiable MST-L Module

Wraps pre-trained MST-L model to match DifferentiableGAPTV interface, enabling
gradient flow for parameter optimization in Algorithm 1 & 2.

Key Design:
- Frozen weights by default (gradients flow only through inputs)
- Matches DifferentiableGAPTV.forward() interface exactly
- Reuses shift operations from mst.py (already PyTorch/autograd-compatible)
- Supports both GPU and CPU inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pwm_core.recon.mst import create_mst, shift_torch, shift_back_meas_torch

logger = logging.getLogger(__name__)


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("DifferentiableMST requires PyTorch. Install with: pip install torch")


class DifferentiableMST(nn.Module):
    """Wrapper for pre-trained MST-L model enabling gradient-based optimization.

    Interface matches DifferentiableGAPTV exactly:
    - forward(y, mask_2d, phi_d_deg) -> [1, L, H, W] reconstructed hyperspectral cube
    - Supports gradient flow through measurement and mask for mismatch calibration

    Design:
    - Model weights frozen by default (frozen_weights=True)
    - Shift operations copied from mst.py are fully differentiable
    - Pre-trained weights loaded from model_zoo or auto-search

    Attributes:
        H: Spatial height
        W: Spatial width
        L: Number of spectral bands
        step: CASSI dispersion step (pixels/band)
        variant: MST model variant ('mst_s', 'mst_m', 'mst_l', 'mst_plus_plus')
        device: torch device
        model: MST model instance (frozen by default)
    """

    def __init__(
        self,
        H: int = 256,
        W: int = 256,
        L: int = 28,
        step: int = 2,
        variant: str = "mst_l",
        frozen_weights: bool = True,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize DifferentiableMST.

        Args:
            H: Spatial height (default 256)
            W: Spatial width (default 256)
            L: Number of spectral bands (default 28)
            step: CASSI dispersion step (default 2)
            variant: MST variant name ('mst_s', 'mst_m', 'mst_l', 'mst_plus_plus')
            frozen_weights: If True, freeze model parameters for gradient flow through inputs only
            weights_path: Optional explicit path to pretrained weights
            device: torch device string or None (auto-select cuda:0 or cpu)
        """
        super().__init__()
        _require_torch()

        self.H = H
        self.W = W
        self.L = L
        self.step = step
        self.variant = variant
        self.frozen_weights = frozen_weights

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            f"Initializing DifferentiableMST: {variant} @ {self.device} "
            f"(frozen_weights={frozen_weights})"
        )

        # Create MST model from variant config
        self.model = create_mst(
            variant=variant,
            in_channels=L,
            out_channels=L,
            base_resolution=H,
            step=step,
        ).to(self.device)

        # Load pretrained weights
        self._load_weights(weights_path)

        # Freeze parameters if requested
        if frozen_weights:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info(f"Froze {sum(1 for _ in self.model.parameters())} parameters")
        else:
            logger.info(f"Model weights are trainable")

        # Set both wrapper and model to eval mode
        self.eval()
        self.model.eval()

    def _load_weights(self, weights_path: Optional[str] = None):
        """Load pretrained weights from checkpoint.

        Searches standard locations:
        1. Explicit weights_path if provided
        2. pkg_root/weights/mst/{variant}.pth
        3. pkg_root/weights/mst/mst_l.pth (fallback)
        4. /home/spiritai/MST-main/model_zoo/mst/{variant}.pth

        Args:
            weights_path: Optional explicit path to weights file
        """
        # Try explicit path first
        if weights_path and Path(weights_path).exists():
            logger.info(f"Loading weights from explicit path: {weights_path}")
            self._load_checkpoint(weights_path)
            return

        # Search in standard locations
        pkg_root = Path(__file__).resolve().parent.parent
        search_paths = [
            pkg_root / f"weights/mst/{self.variant}.pth",
            pkg_root / "weights/mst/mst_l.pth",  # fallback to mst_l
            Path("/home/spiritai/MST-main/model_zoo/mst") / f"{self.variant}.pth",
            Path("/home/spiritai/MST-main/model_zoo/mst/mst_l.pth"),  # fallback
        ]

        for weights_file in search_paths:
            if weights_file.exists():
                logger.info(f"Loading weights from: {weights_file}")
                self._load_checkpoint(str(weights_file))
                return

        logger.warning(f"No weights found for {self.variant}. Model initialized with random weights.")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load state_dict from checkpoint file.

        Handles both direct state_dict and wrapped checkpoints (with 'state_dict' key).
        Removes 'module.' prefix for compatibility with DataParallel.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Extract state_dict if checkpoint is wrapped
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix for compatibility
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded weights from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights from {checkpoint_path}: {e}")

    def forward(
        self,
        y: torch.Tensor,
        mask_2d: torch.Tensor,
        phi_d_deg: float = 0.0,
    ) -> torch.Tensor:
        """Reconstruct hyperspectral cube using MST-L.

        Matches DifferentiableGAPTV interface exactly:
        1. Convert 2D measurement to initial spectral estimate via shift_back
        2. Create weighted input: initial_estimate * mask_2d
        3. Forward through MST model
        4. Clamp output to [0, 1]

        Args:
            y: 2D measurement [B, H, W_ext] where W_ext = W + (L-1)*step
            mask_2d: 2D coded aperture mask [H, W] (single spatial mask applied to all bands)
            phi_d_deg: Spectral response angle in degrees (unused for MST, for interface compatibility)

        Returns:
            Reconstructed hyperspectral cube [B, L, H, W] with values in [0, 1]
        """
        # Ensure inputs are on correct device
        y = y.to(self.device)
        mask_2d = mask_2d.to(self.device)

        # Convert measurement to initial spectral estimate via shift_back
        # shape: [B, L, H, W]
        x_initial = shift_back_meas_torch(y, step=self.step, nC=self.L)

        # Expand mask to all bands and weight initial estimate
        # mask_2d: [H, W] -> [1, 1, H, W] -> [1, L, H, W]
        if mask_2d.ndim == 2:
            mask_expanded = mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            mask_expanded = mask_2d  # assume already expanded

        mask_expanded = mask_expanded.expand_as(x_initial)  # [1, L, H, W]

        # Weight initial estimate by mask
        x_weighted = x_initial * mask_expanded

        # Forward through MST model
        with torch.set_grad_enabled(True):
            x_recon = self.model(x_weighted)

        # Clamp to valid range [0, 1]
        x_recon = torch.clamp(x_recon, min=0.0, max=1.0)

        return x_recon

    def set_frozen(self, frozen: bool):
        """Enable/disable weight freezing.

        Args:
            frozen: If True, freeze all model parameters
        """
        for param in self.model.parameters():
            param.requires_grad = not frozen
        self.frozen_weights = frozen
        logger.info(f"Model weights {'frozen' if frozen else 'unfrozen'}")


__all__ = ["DifferentiableMST"]
