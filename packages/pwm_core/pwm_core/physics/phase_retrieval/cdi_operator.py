"""CDI (Coherent Diffraction Imaging) phase retrieval operator.

Forward model: y = |F{x}|^2 (squared Fourier magnitude / power spectrum)

The object x is a complex-valued (or real-valued) exit wave of shape (ny, nx).
The measurement y is the far-field diffraction pattern (ny, nx).

Because the forward model involves a squared magnitude the mapping is
nonlinear in x, so is_linear = False.  The adjoint is an approximate
back-projection: x_adj = real(ifft2(sqrt(y))), corresponding to a
magnitude-only inverse with zero-phase assumption.

References:
- Miao, J. et al. (1999). "Extending the methodology of X-ray
  crystallography to allow imaging of micrometre-sized non-crystalline
  specimens", Nature.
- Fienup, J. R. (1982). "Phase retrieval algorithms: a comparison",
  Applied Optics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class CDIOperator(BaseOperator):
    """CDI coherent diffraction imaging (phase retrieval) operator.

    Forward: x (ny, nx) -> y (ny, nx)
        y = |fft2(x)|^2

    Adjoint (approximate back-projection):
        x = real(ifft2(sqrt(y)))
        This is the zero-phase magnitude-only inverse, suitable as an
        initialiser or as a linear step inside iterative algorithms
        (HIO, ER, RAAR, etc.).
    """

    def __init__(
        self,
        operator_id: str = "phase_retrieval",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        support: Optional[np.ndarray] = None,
        oversampling: int = 2,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.oversampling = oversampling
        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = False
        self._supports_autodiff = False

        # Support constraint
        if support is not None:
            self.support = support.astype(np.float64)
        else:
            # Default: central circular support with radius = ny // 3
            yy, xx = np.ogrid[:ny, :nx]
            cy, cx = ny / 2.0, nx / 2.0
            radius = ny // 3
            self.support = ((yy - cy) ** 2 + (xx - cx) ** 2
                            <= radius ** 2).astype(np.float64)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute far-field diffraction pattern (power spectrum).

        Args:
            x: Exit wave (ny, nx), real or complex.

        Returns:
            Diffraction intensity (ny, nx), real non-negative.
        """
        x64 = np.asarray(x, dtype=np.complex128)
        X = np.fft.fft2(x64)
        y = np.abs(X) ** 2
        return y.astype(np.float32)

    # ------------------------------------------------------------------
    # Adjoint (approximate)
    # ------------------------------------------------------------------

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Approximate back-projection from diffraction intensities.

        Uses the magnitude-only inverse with zero-phase assumption:
            x_adj = real(ifft2(sqrt(y)))

        This is NOT the true adjoint of the nonlinear forward model,
        but serves as a useful initialisation for iterative phase
        retrieval algorithms and satisfies the interface contract.

        Args:
            y: Diffraction intensity (ny, nx), real non-negative.

        Returns:
            Estimated exit wave (ny, nx), real-valued.
        """
        y64 = np.asarray(y, dtype=np.float64)

        # Clamp to non-negative before sqrt
        y64 = np.clip(y64, 0.0, None)

        # Magnitude-only inverse: assume zero phase
        magnitudes = np.sqrt(y64)
        x = np.real(np.fft.ifft2(magnitudes))

        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx)

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
            "oversampling": self.oversampling,
            "support_sum": float(self.support.sum()),
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="phase_retrieval",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=False,
            supports_autodiff=False,
            axes={
                "x_dim0": "ny",
                "x_dim1": "nx",
                "y_dim0": "ny",
                "y_dim1": "nx",
            },
            units={
                "x": "complex amplitude (exit wave)",
                "y": "photon counts (intensity)",
            },
            sampling_info={
                "oversampling": self.oversampling,
                "support_pixels": int(self.support.sum()),
            },
        )
