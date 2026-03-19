"""FLIM (Fluorescence Lifetime Imaging Microscopy) operator.

Forward model: y(t) = IRF * (a * exp(-t / tau))
  Convolution of mono-exponential decay with the instrument response function.

The object x is a "parameter image" of shape (ny, nx, 2) where
  x[:,:,0] = amplitude (a)
  x[:,:,1] = lifetime  (tau, nanoseconds)

The measurement y is a "decay histogram" of shape (ny, nx, n_time_bins).

Because tau appears in the exponent the forward model is nonlinear in x,
so is_linear = False.  The adjoint is a linearised approximation (Jacobian
transpose evaluated at a reference lifetime tau_ref).

References:
- Becker, W. (2012). "Fluorescence lifetime imaging - techniques and
  applications", J. Microscopy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class FLIMOperator(BaseOperator):
    """FLIM fluorescence lifetime imaging operator.

    Forward: x (ny, nx, 2) -> y (ny, nx, n_time_bins)
        For each pixel, y = IRF * (a * exp(-t / tau))

    Adjoint (linearised at tau_ref):
        amplitude channel: sum of (y corr IRF) weighted by exp(-t/tau_ref)
        tau channel:       sum of (y corr IRF) weighted by t*exp(-t/tau_ref)/tau_ref^2
    """

    def __init__(
        self,
        operator_id: str = "flim",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 32,
        nx: int = 32,
        n_time_bins: int = 64,
        time_range_ns: float = 12.5,
        irf_sigma_ns: float = 0.3,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_time_bins = n_time_bins
        self.time_range_ns = time_range_ns
        self.irf_sigma_ns = irf_sigma_ns
        self._x_shape = (ny, nx, 2)
        self._y_shape = (ny, nx, n_time_bins)
        self._is_linear = False
        self._supports_autodiff = False

        # Time axis (nanoseconds)
        self.time_axis = np.linspace(0.0, time_range_ns, n_time_bins,
                                     dtype=np.float64)

        # Instrument response function: normalised Gaussian centred at t=0
        self.irf = np.exp(-0.5 * (self.time_axis / irf_sigma_ns) ** 2)
        self.irf /= self.irf.sum()  # normalise to unit area

        # Pre-compute IRF spectrum for FFT-based convolution
        self._irf_fft = np.fft.rfft(self.irf)

        # Reference lifetime for linearised adjoint (ns)
        self._tau_ref = 3.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute fluorescence decay histograms from parameter image.

        Args:
            x: Parameter image (ny, nx, 2).
               Channel 0 = amplitude, channel 1 = lifetime (ns).

        Returns:
            Decay histograms (ny, nx, n_time_bins).
        """
        x64 = x.astype(np.float64)
        a = x64[..., 0]        # (ny, nx)
        tau = x64[..., 1]      # (ny, nx)

        # Clamp tau to avoid division by zero / negative values
        tau = np.clip(tau, 1e-6, None)

        # Mono-exponential decay per pixel: (ny, nx, n_time_bins)
        # time_axis is broadcast: (1, 1, n_time_bins)
        t = self.time_axis[np.newaxis, np.newaxis, :]   # (1, 1, T)
        decay = a[..., np.newaxis] * np.exp(-t / tau[..., np.newaxis])

        # Convolve each pixel's decay with the IRF via FFT along time axis
        n_fft = self.n_time_bins
        decay_fft = np.fft.rfft(decay, n=n_fft, axis=-1)
        y = np.fft.irfft(decay_fft * self._irf_fft[np.newaxis, np.newaxis, :],
                         n=n_fft, axis=-1)

        return y.astype(np.float32)

    # ------------------------------------------------------------------
    # Adjoint (linearised)
    # ------------------------------------------------------------------

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Linearised adjoint (J^T y) evaluated at reference lifetime.

        The Jacobian of the forward model has two blocks per pixel
        (amplitude and tau). We evaluate J^T at tau_ref to obtain a
        fixed linear map suitable for iterative solvers.

        Args:
            y: Decay histograms (ny, nx, n_time_bins).

        Returns:
            Parameter image (ny, nx, 2).
        """
        y64 = y.astype(np.float64)
        tau_ref = self._tau_ref
        t = self.time_axis  # (T,)

        # Correlate y with IRF (matched filter) via FFT
        # correlation(y, irf) = ifft(fft(y) * conj(fft(irf)))
        y_fft = np.fft.rfft(y64, n=self.n_time_bins, axis=-1)
        irf_conj_fft = np.conj(self._irf_fft)[np.newaxis, np.newaxis, :]
        y_filtered = np.fft.irfft(y_fft * irf_conj_fft,
                                  n=self.n_time_bins, axis=-1)

        # Reference exponential basis functions
        exp_ref = np.exp(-t / tau_ref)                     # (T,)
        d_tau_basis = (t * np.exp(-t / tau_ref)) / (tau_ref ** 2)  # (T,)

        # Amplitude channel: <y_filtered, exp(-t/tau_ref)>
        x_amp = np.sum(y_filtered * exp_ref[np.newaxis, np.newaxis, :],
                       axis=-1)

        # Tau channel: <y_filtered, t*exp(-t/tau_ref)/tau_ref^2>
        x_tau = np.sum(y_filtered * d_tau_basis[np.newaxis, np.newaxis, :],
                       axis=-1)

        x = np.stack([x_amp, x_tau], axis=-1)
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx, 2)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx, self.n_time_bins)

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def supports_autodiff(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Info / Metadata
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "ny": self.ny,
            "nx": self.nx,
            "n_time_bins": self.n_time_bins,
            "time_range_ns": self.time_range_ns,
            "irf_sigma_ns": self.irf_sigma_ns,
            "tau_ref_ns": self._tau_ref,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="flim",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=False,
            supports_autodiff=False,
            axes={
                "x_dim0": "ny",
                "x_dim1": "nx",
                "x_dim2": "param (amp|tau)",
                "y_dim0": "ny",
                "y_dim1": "nx",
                "y_dim2": "time_bins",
            },
            units={
                "time": "nanoseconds",
                "amplitude": "a.u.",
                "lifetime": "nanoseconds",
            },
            sampling_info={
                "time_range_ns": self.time_range_ns,
                "n_time_bins": self.n_time_bins,
                "irf_sigma_ns": self.irf_sigma_ns,
            },
        )
