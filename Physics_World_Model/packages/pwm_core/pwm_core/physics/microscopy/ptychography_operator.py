"""Ptychography operator.

Implements ptychographic imaging with overlapping probe scans.
Output is 3D stack of diffraction patterns (n_positions, H, W) from 2D input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


class PtychographyOperator(BaseOperator):
    """Ptychography operator with scanning probe.

    Forward: Apply probe at scan positions, compute magnitude of FFT
    Adjoint: Back-propagate through positions
    """

    def __init__(
        self,
        operator_id: str = "ptychography",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        n_positions: int = 16,
        probe_size: int = 32,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.n_positions = n_positions
        self.probe_size = probe_size

        H, W = x_shape

        # Generate scan positions (grid pattern with overlap)
        rng = np.random.default_rng(seed)
        n_side = int(np.sqrt(n_positions))
        step_h = (H - probe_size) // max(n_side - 1, 1)
        step_w = (W - probe_size) // max(n_side - 1, 1)

        self.positions = []
        for i in range(n_side):
            for j in range(n_side):
                pos_h = min(i * step_h, H - probe_size)
                pos_w = min(j * step_w, W - probe_size)
                self.positions.append((pos_h, pos_w))
        self.n_positions = len(self.positions)

        # Generate probe function (Gaussian illumination)
        yy, xx = np.meshgrid(np.arange(probe_size), np.arange(probe_size))
        center = probe_size // 2
        sigma = probe_size / 4
        self.probe = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * sigma**2))
        self.probe = self.probe.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ptychography forward model: probe * exit_wave -> |FFT|^2"""
        y = np.zeros((self.n_positions, self.probe_size, self.probe_size), dtype=np.float32)

        for i, (pos_h, pos_w) in enumerate(self.positions):
            # Extract patch and apply probe
            patch = x[pos_h:pos_h+self.probe_size, pos_w:pos_w+self.probe_size]
            exit_wave = patch * self.probe

            # Compute diffraction pattern (magnitude of FFT)
            diffraction = np.abs(np.fft.fft2(exit_wave))**2
            y[i] = diffraction.astype(np.float32)

        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: accumulate weighted back-projections."""
        H, W = self.x_shape
        x_adj = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for i, (pos_h, pos_w) in enumerate(self.positions):
            # IFFT of sqrt(intensity) as amplitude estimate
            amplitude = np.sqrt(np.maximum(y[i], 0))
            back = np.real(np.fft.ifft2(amplitude))

            # Weight by probe and accumulate
            x_adj[pos_h:pos_h+self.probe_size, pos_w:pos_w+self.probe_size] += back * self.probe
            weight[pos_h:pos_h+self.probe_size, pos_w:pos_w+self.probe_size] += self.probe

        # Normalize by weight
        x_adj = np.divide(x_adj, weight, where=weight > 1e-6, out=x_adj)

        return x_adj.astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_positions": self.n_positions,
            "probe_size": self.probe_size,
        }
