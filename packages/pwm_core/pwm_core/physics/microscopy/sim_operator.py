"""Structured Illumination Microscopy (SIM) operator.

Implements SIM with multiple illumination patterns.
Output is 3D stack (H, W, n_patterns) from 2D input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class SIMOperator(BaseOperator):
    """SIM operator with pattern modulation.

    Forward: Generate pattern-modulated measurement stack
    Adjoint: Combine patterns to estimate high-res image
    """

    def __init__(
        self,
        operator_id: str = "sim",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        n_angles: int = 3,
        n_phases: int = 3,
        pattern_freq: float = 0.1,
        psf_sigma: float = 1.5,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.n_angles = n_angles
        self.n_phases = n_phases
        self.n_patterns = n_angles * n_phases
        self.pattern_freq = pattern_freq
        self.psf_sigma = psf_sigma

        # Pre-compute patterns
        self.patterns = self._generate_patterns()

    def _generate_patterns(self) -> np.ndarray:
        """Generate SIM illumination patterns."""
        H, W = self.x_shape
        patterns = np.zeros((H, W, self.n_patterns), dtype=np.float32)

        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        idx = 0
        for a in range(self.n_angles):
            angle = a * np.pi / self.n_angles
            for p in range(self.n_phases):
                phase = p * 2 * np.pi / self.n_phases
                pattern = 0.5 + 0.5 * np.cos(
                    2 * np.pi * self.pattern_freq * (
                        xx * np.cos(angle) + yy * np.sin(angle)
                    ) + phase
                )
                patterns[:, :, idx] = pattern
                idx += 1

        return patterns

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply SIM forward model: modulate + blur."""
        H, W = self.x_shape
        y = np.zeros((H, W, self.n_patterns), dtype=np.float32)

        for i in range(self.n_patterns):
            # Modulate with pattern
            modulated = x * self.patterns[:, :, i]
            # Apply PSF blur
            blurred = ndimage.gaussian_filter(modulated, sigma=self.psf_sigma)
            y[:, :, i] = blurred

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: sum pattern-weighted back-blurred images."""
        H, W = self.x_shape
        x_adj = np.zeros((H, W), dtype=np.float32)

        for i in range(self.n_patterns):
            # Apply adjoint of blur (same as blur for Gaussian)
            back_blurred = ndimage.gaussian_filter(y[:, :, i], sigma=self.psf_sigma)
            # Weight by pattern
            x_adj += back_blurred * self.patterns[:, :, i]

        return (x_adj / self.n_patterns).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_patterns": self.n_patterns,
            "pattern_freq": self.pattern_freq,
        }
