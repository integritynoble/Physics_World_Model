"""Fourier optics IO convention -- standardized discretization for all optics nodes."""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PhotonFieldSpec(BaseModel):
    """Standardized discretization contract for all optics primitives.

    Every optics primitive (FresnelProp, AngularSpectrum, FourierRelay,
    Maxwell stub) must produce output on the same grid described by this spec.
    This ensures tiers are swappable (Tier 1 -> Tier 2 same IO).
    """
    model_config = ConfigDict(extra="forbid")

    wavelength_m: float = 532e-9
    wavelength_bins: Optional[List[float]] = None
    grid_shape: Tuple[int, int] = (64, 64)
    pixel_pitch_m: float = 6.5e-6

    @property
    def freq_pitch(self) -> Tuple[float, float]:
        ny, nx = self.grid_shape
        return (1.0 / (ny * self.pixel_pitch_m), 1.0 / (nx * self.pixel_pitch_m))

    @property
    def max_freq(self) -> float:
        return 1.0 / (2.0 * self.pixel_pitch_m)

    def freq_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.grid_shape
        fy = np.fft.fftfreq(ny, d=self.pixel_pitch_m)
        fx = np.fft.fftfreq(nx, d=self.pixel_pitch_m)
        return np.meshgrid(fy, fx, indexing='ij')

    def real_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.grid_shape
        y = (np.arange(ny) - ny / 2) * self.pixel_pitch_m
        x = (np.arange(nx) - nx / 2) * self.pixel_pitch_m
        return np.meshgrid(y, x, indexing='ij')

    def validate_output_shape(self, output: np.ndarray) -> bool:
        expected = self.grid_shape
        actual = output.shape[-2:]
        return actual == expected
