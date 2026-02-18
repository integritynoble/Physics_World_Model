"""Level 4 Example: A custom Tier-1 primitive (Gaussian blur with learnable width).

This demonstrates how to implement a new PrimitiveOp for the PWM
PRIMITIVE_REGISTRY. This is a simplified version of the existing conv2d
primitive, provided as a reference implementation.

To contribute a new primitive:
1. Open RFC issue with physics justification + adjoint proof
2. Implement forward() and adjoint()
3. Pass adjoint correctness tests (see adjoint_test below)
4. Submit PR for steward review
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class GaussianBlurPrimitive:
    """Example Tier-1 primitive: 2D Gaussian blur via FFT.

    Implements the PrimitiveOp protocol:
    - forward(x) -> y
    - adjoint(y) -> x
    - serialize() -> dict
    - Properties: primitive_id, is_linear, _physics_tier
    """

    primitive_id = "example_gaussian_blur"
    _is_linear = True
    _is_stochastic = False
    _is_differentiable = True
    _is_stateful = False
    _physics_tier = "tier1_approx"
    _physics_subrole = "modulation"

    def __init__(self, sigma: float = 2.0, **kwargs):
        self._params = {"sigma": sigma}
        self._sigma = sigma

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def is_stochastic(self) -> bool:
        return self._is_stochastic

    @property
    def is_differentiable(self) -> bool:
        return self._is_differentiable

    @property
    def is_stateful(self) -> bool:
        return self._is_stateful

    def _build_kernel_fft(self, shape):
        """Build Gaussian kernel in frequency domain."""
        sigma = self._params["sigma"]
        rows, cols = shape[-2], shape[-1]
        y_coords = np.fft.fftfreq(rows)
        x_coords = np.fft.fftfreq(cols)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        # Gaussian in frequency domain
        kernel_fft = np.exp(-2 * np.pi**2 * sigma**2 * (xx**2 + yy**2))
        return kernel_fft

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur: y = G * x (convolution via FFT)."""
        kernel_fft = self._build_kernel_fft(x.shape)
        x_fft = np.fft.fft2(x)
        y_fft = x_fft * kernel_fft
        return np.real(np.fft.ifft2(y_fft))

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint of Gaussian blur: x = G^T * y.

        For real symmetric kernels, adjoint = forward (self-adjoint).
        """
        kernel_fft = self._build_kernel_fft(y.shape)
        y_fft = np.fft.fft2(y)
        # Adjoint = conjugate of kernel in freq domain
        x_fft = y_fft * np.conj(kernel_fft)
        return np.real(np.fft.ifft2(x_fft))

    def serialize(self) -> Dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "params": dict(self._params),
            "physics_tier": self._physics_tier,
        }


# ---------------------------------------------------------------------------
# Adjoint correctness test (required for all primitives)
# ---------------------------------------------------------------------------


def test_adjoint_correctness(n_trials: int = 5, tol: float = 1e-6):
    """Verify <Ax, y> == <x, A^T y> for random vectors.

    This is the fundamental test that every primitive must pass.
    """
    prim = GaussianBlurPrimitive(sigma=1.5)
    rng = np.random.default_rng(42)
    shape = (32, 32)

    max_err = 0.0
    for trial in range(n_trials):
        x = rng.standard_normal(shape)
        y = rng.standard_normal(shape)

        Ax = prim.forward(x)
        ATy = prim.adjoint(y)

        inner_Ax_y = float(np.sum(Ax * y))
        inner_x_ATy = float(np.sum(x * ATy))

        denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
        rel_err = abs(inner_Ax_y - inner_x_ATy) / denom
        max_err = max(max_err, rel_err)

    passed = max_err < tol
    status = "PASS" if passed else "FAIL"
    print(f"Adjoint test [{status}]: max_rel_err = {max_err:.2e}, tol = {tol:.2e}")
    return passed


if __name__ == "__main__":
    test_adjoint_correctness()
