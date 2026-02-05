"""CACTI (Coded Aperture Compressive Temporal Imaging) operator.

Implements Snapshot Compressive Imaging for video.
Multiple video frames are compressed into a single 2D measurement
using time-varying coded aperture masks.

Input: 3D video cube (H, W, T) where T is the number of frames
Output: Single 2D snapshot measurement (H, W)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


class CACTIOperator(BaseOperator):
    """CACTI operator for video snapshot compressive imaging.

    Forward: y = sum_t(mask_t * x_t) where mask_t is shifted mask
    Adjoint: x_t = mask_t * y for each frame
    """

    def __init__(
        self,
        operator_id: str = "cacti",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int, int] = (64, 64, 8),
        mask: Optional[np.ndarray] = None,
        shift_type: str = "vertical",
        seed: int = 42,
    ):
        """Initialize CACTI operator.

        Args:
            operator_id: Operator identifier
            theta: Optional parameters
            x_shape: (H, W, T) video dimensions
            mask: Optional (H, W) base coded aperture mask. If None, random binary mask is generated.
            shift_type: 'vertical', 'horizontal', or 'random' mask shift pattern
            seed: Random seed for mask generation
        """
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.shift_type = shift_type

        H, W, T = x_shape
        self.n_frames = T

        # Generate or use provided base mask
        if mask is not None:
            self.base_mask = mask.astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            self.base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)

        # Pre-compute shifted masks for each frame
        self.masks = self._generate_shifted_masks()

    def _generate_shifted_masks(self) -> np.ndarray:
        """Generate time-varying masks by shifting base mask."""
        H, W, T = self.x_shape
        masks = np.zeros((H, W, T), dtype=np.float32)

        for t in range(T):
            if self.shift_type == "vertical":
                # Vertical shift (common in CACTI)
                shift = t
                masks[:, :, t] = np.roll(self.base_mask, shift, axis=0)
            elif self.shift_type == "horizontal":
                # Horizontal shift
                shift = t
                masks[:, :, t] = np.roll(self.base_mask, shift, axis=1)
            elif self.shift_type == "diagonal":
                # Diagonal shift
                masks[:, :, t] = np.roll(np.roll(self.base_mask, t, axis=0), t, axis=1)
            else:
                # Random independent masks for each frame
                rng = np.random.default_rng(42 + t)
                masks[:, :, t] = (rng.random((H, W)) > 0.5).astype(np.float32)

        return masks

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compress video to single snapshot.

        Args:
            x: Input video cube (H, W, T) or 2D image (H, W)

        Returns:
            y: Compressed 2D measurement (H, W)
        """
        H, W, T = self.x_shape

        # Handle 2D input by creating synthetic video frames
        if x.ndim == 2:
            # Create video by adding motion/variation to 2D image
            x_3d = np.zeros((H, W, T), dtype=np.float32)
            for t in range(T):
                # Simulate temporal variation with slight shifts
                shift = t - T // 2
                x_3d[:, :, t] = np.roll(x, shift, axis=0) * (1.0 - 0.05 * abs(shift))
        else:
            x_3d = x

        # Sum of masked frames
        y = np.zeros((H, W), dtype=np.float32)
        for t in range(T):
            y += x_3d[:, :, t] * self.masks[:, :, t]

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project measurement to video frames.

        Args:
            y: 2D measurement (H, W)

        Returns:
            x: Reconstructed video cube (H, W, T)
        """
        H, W, T = self.x_shape

        if y.ndim != 2:
            raise ValueError(f"Expected 2D input (H, W), got shape {y.shape}")

        # Distribute measurement back using masks
        x = np.zeros((H, W, T), dtype=np.float32)
        for t in range(T):
            x[:, :, t] = y * self.masks[:, :, t]

        return x.astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_frames": self.n_frames,
            "shift_type": self.shift_type,
            "compression_ratio": self.n_frames,
        }
