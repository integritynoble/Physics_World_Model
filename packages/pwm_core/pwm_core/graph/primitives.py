"""pwm_core.graph.primitives
============================

PrimitiveOp protocol and ~30 primitive operator implementations.

Each primitive provides forward(), adjoint() (where applicable), serialize(),
and an is_linear property.  Primitives are registered by ``primitive_id`` in
PRIMITIVE_REGISTRY and looked up at compile time by the GraphCompiler.

Design rules
------------
* Primitives accept and return numpy arrays.
* Parameters are passed at construction time (``params`` dict).
* Where an existing operator already implements the maths, the primitive
  wraps it; otherwise a standalone implementation is provided.
* Non-linear primitives (noise, magnitude_sq, saturation, ...) set
  ``is_linear = False`` and their adjoint raises ``NotImplementedError``
  unless a meaningful pseudo-adjoint exists.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy import ndimage

from pwm_core.mismatch.subpixel import subpixel_shift_2d


# ---------------------------------------------------------------------------
# PrimitiveOp protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PrimitiveOp(Protocol):
    """Structural interface for every graph primitive."""

    primitive_id: str

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        ...

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        ...

    def serialize(self) -> Dict[str, Any]:
        ...

    @property
    def is_linear(self) -> bool:
        ...


# ---------------------------------------------------------------------------
# Base class for primitives (convenience, not mandatory)
# ---------------------------------------------------------------------------


class BasePrimitive:
    """Convenience base for primitives with sensible defaults."""

    primitive_id: str = "base"
    _is_linear: bool = True
    _is_stochastic: bool = False
    _is_differentiable: bool = True
    _is_stateful: bool = False
    _n_inputs: int = 1
    _physics_subrole: Optional[str] = None
    _params: Dict[str, Any]

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._params = dict(params or {})

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError

    def forward_multi(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass for multi-input nodes. Default delegates to forward()."""
        return self.forward(next(iter(inputs.values())))

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError(
            f"Adjoint not available for primitive '{self.primitive_id}'"
        )

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

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "primitive_id": self.primitive_id,
            "is_linear": self._is_linear,
            "params": {},
            "blobs": [],
        }
        for k, v in self._params.items():
            if isinstance(v, np.ndarray) and v.size > 1000:
                sha = hashlib.sha256(v.tobytes()).hexdigest()
                result["blobs"].append({
                    "name": k,
                    "sha256": sha,
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                })
            elif isinstance(v, np.ndarray):
                result["params"][k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result["params"][k] = v.item()
            else:
                result["params"][k] = v
        return result


# =========================================================================
# Propagation family
# =========================================================================


class FresnelProp(BasePrimitive):
    """Fresnel propagation via single-FFT method."""

    primitive_id = "fresnel_prop"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        wavelength = self._params.get("wavelength", 0.5e-6)
        distance = self._params.get("distance", 1e-3)
        pixel_size = self._params.get("pixel_size", 1e-6)
        H, W = x.shape[-2], x.shape[-1]
        fy = np.fft.fftfreq(H, d=pixel_size).reshape(-1, 1)
        fx = np.fft.fftfreq(W, d=pixel_size).reshape(1, -1)
        kernel = np.exp(
            1j * np.pi * wavelength * distance * (fx**2 + fy**2)
        )
        X = np.fft.fft2(x.astype(np.complex128))
        return np.fft.ifft2(X * kernel).astype(np.complex128)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        wavelength = self._params.get("wavelength", 0.5e-6)
        distance = self._params.get("distance", 1e-3)
        pixel_size = self._params.get("pixel_size", 1e-6)
        H, W = y.shape[-2], y.shape[-1]
        fy = np.fft.fftfreq(H, d=pixel_size).reshape(-1, 1)
        fx = np.fft.fftfreq(W, d=pixel_size).reshape(1, -1)
        kernel_conj = np.exp(
            -1j * np.pi * wavelength * distance * (fx**2 + fy**2)
        )
        Y = np.fft.fft2(y.astype(np.complex128))
        return np.fft.ifft2(Y * kernel_conj).astype(np.complex128)


class AngularSpectrum(BasePrimitive):
    """Angular-spectrum propagation (full transfer function)."""

    primitive_id = "angular_spectrum"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        wavelength = self._params.get("wavelength", 0.5e-6)
        distance = self._params.get("distance", 1e-3)
        pixel_size = self._params.get("pixel_size", 1e-6)
        H, W = x.shape[-2], x.shape[-1]
        fy = np.fft.fftfreq(H, d=pixel_size).reshape(-1, 1)
        fx = np.fft.fftfreq(W, d=pixel_size).reshape(1, -1)
        kz_sq = np.maximum(1.0 / wavelength**2 - fx**2 - fy**2, 0.0)
        kz = np.sqrt(kz_sq)
        H_tf = np.exp(2j * np.pi * distance * kz)
        X = np.fft.fft2(x.astype(np.complex128))
        return np.fft.ifft2(X * H_tf).astype(np.complex128)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        wavelength = self._params.get("wavelength", 0.5e-6)
        distance = self._params.get("distance", 1e-3)
        pixel_size = self._params.get("pixel_size", 1e-6)
        H, W = y.shape[-2], y.shape[-1]
        fy = np.fft.fftfreq(H, d=pixel_size).reshape(-1, 1)
        fx = np.fft.fftfreq(W, d=pixel_size).reshape(1, -1)
        kz_sq = np.maximum(1.0 / wavelength**2 - fx**2 - fy**2, 0.0)
        kz = np.sqrt(kz_sq)
        H_tf_conj = np.exp(-2j * np.pi * distance * kz)
        Y = np.fft.fft2(y.astype(np.complex128))
        return np.fft.ifft2(Y * H_tf_conj).astype(np.complex128)


class RayTrace(BasePrimitive):
    """Simplified ray-tracing as geometric warping (thin-lens model)."""

    primitive_id = "ray_trace"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        focal_length = self._params.get("focal_length", 50.0)
        magnification = self._params.get("magnification", 1.0)
        zoom = 1.0 / magnification if magnification != 0 else 1.0
        return ndimage.zoom(x, zoom, order=1, mode="constant").astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        magnification = self._params.get("magnification", 1.0)
        zoom = magnification
        return ndimage.zoom(y, zoom, order=1, mode="constant").astype(np.float64)


# =========================================================================
# PSF / Convolution family
# =========================================================================


class Conv2d(BasePrimitive):
    """2D convolution with symmetric PSF (self-adjoint when PSF is real symmetric)."""

    primitive_id = "conv2d"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 2.0)
        mode = self._params.get("mode", "reflect")
        return ndimage.gaussian_filter(
            x.astype(np.float64), sigma=sigma, mode=mode
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 2.0)
        mode = self._params.get("mode", "reflect")
        return ndimage.gaussian_filter(
            y.astype(np.float64), sigma=sigma, mode=mode
        )


class Conv3d(BasePrimitive):
    """3D convolution with isotropic Gaussian PSF."""

    primitive_id = "conv3d"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", [2.0, 2.0, 2.0])
        mode = self._params.get("mode", "reflect")
        return ndimage.gaussian_filter(
            x.astype(np.float64), sigma=sigma, mode=mode
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", [2.0, 2.0, 2.0])
        mode = self._params.get("mode", "reflect")
        return ndimage.gaussian_filter(
            y.astype(np.float64), sigma=sigma, mode=mode
        )


class DeconvRL(BasePrimitive):
    """Richardson-Lucy deconvolution (iterative, non-linear, no adjoint)."""

    primitive_id = "deconv_rl"
    _is_linear = False
    _is_differentiable = False
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 2.0)
        n_iter = self._params.get("n_iter", 10)
        mode = self._params.get("mode", "reflect")
        blurred = ndimage.gaussian_filter(x.astype(np.float64), sigma=sigma, mode=mode)
        estimate = np.ones_like(blurred, dtype=np.float64) * blurred.mean()
        eps = 1e-12
        for _ in range(n_iter):
            conv_est = ndimage.gaussian_filter(estimate, sigma=sigma, mode=mode)
            ratio = blurred / (conv_est + eps)
            correction = ndimage.gaussian_filter(ratio, sigma=sigma, mode=mode)
            estimate = estimate * correction
        return estimate


# =========================================================================
# Modulation family
# =========================================================================


class CodedMask(BasePrimitive):
    """Element-wise coded aperture masking (linear).

    Supports 2D input (H, W) and 3D input (H, W, L) -- the 2D mask
    is broadcast along the spectral/temporal axis.
    """

    primitive_id = "coded_mask"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "modulation"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        mask = self._params.get("mask", None)
        seed = self._params.get("seed", 42)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        if mask is not None:
            self._mask = np.asarray(mask, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed)
            self._mask = (rng.random((H, W)) > 0.5).astype(np.float64)

    def _broadcast_mask(self, x: np.ndarray) -> np.ndarray:
        """Broadcast 2D mask to match input dimensions."""
        if x.ndim > self._mask.ndim:
            # Add trailing dimensions for broadcasting (H,W) -> (H,W,1,...)
            mask = self._mask
            for _ in range(x.ndim - self._mask.ndim):
                mask = mask[..., np.newaxis]
            return mask
        return self._mask

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        mask = self._broadcast_mask(x)
        return (x.astype(np.float64) * mask).astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        mask = self._broadcast_mask(y)
        return (y.astype(np.float64) * mask).astype(np.float64)


class DMDPattern(BasePrimitive):
    """DMD (digital micromirror device) pattern modulation."""

    primitive_id = "dmd_pattern"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "modulation"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        pattern = self._params.get("pattern", None)
        seed = self._params.get("seed", 42)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        if pattern is not None:
            self._pattern = np.asarray(pattern, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed)
            self._pattern = (rng.random((H, W)) > 0.5).astype(np.float64) * 2 - 1

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        return (x.astype(np.float64) * self._pattern).astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        return (y.astype(np.float64) * self._pattern).astype(np.float64)


class SIMPattern(BasePrimitive):
    """Structured illumination pattern modulation."""

    primitive_id = "sim_pattern"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "modulation"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        freq = self._params.get("freq", 0.1)
        angle = self._params.get("angle", 0.0)
        phase = self._params.get("phase", 0.0)
        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        self._pattern = (
            0.5
            + 0.5
            * np.cos(
                2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle))
                + phase
            )
        ).astype(np.float64)

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        return (x.astype(np.float64) * self._pattern).astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        return (y.astype(np.float64) * self._pattern).astype(np.float64)


# =========================================================================
# Warp / Dispersion family
# =========================================================================


class SpectralDispersion(BasePrimitive):
    """Spectral dispersion shift per wavelength band."""

    primitive_id = "spectral_dispersion"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "encoding"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        disp_step = self._params.get("disp_step", 1.0)
        L = x.shape[-1] if x.ndim >= 3 else 1
        if x.ndim < 3:
            return x.copy()
        out = np.zeros_like(x, dtype=np.float64)
        for l in range(L):
            shift = disp_step * l
            out[:, :, l] = subpixel_shift_2d(x[:, :, l].astype(np.float64), shift, 0.0)
        return out

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        disp_step = self._params.get("disp_step", 1.0)
        L = y.shape[-1] if y.ndim >= 3 else 1
        if y.ndim < 3:
            return y.copy()
        out = np.zeros_like(y, dtype=np.float64)
        for l in range(L):
            shift = -(disp_step * l)
            out[:, :, l] = subpixel_shift_2d(y[:, :, l].astype(np.float64), shift, 0.0)
        return out


class ChromaticWarp(BasePrimitive):
    """Wavelength-dependent geometric warp (chromatic aberration)."""

    primitive_id = "chromatic_warp"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        warp_coeff = self._params.get("warp_coeff", 0.01)
        L = x.shape[-1] if x.ndim >= 3 else 1
        if x.ndim < 3:
            return x.copy()
        out = np.zeros_like(x, dtype=np.float64)
        for l in range(L):
            zoom_factor = 1.0 + warp_coeff * (l - L / 2)
            band = x[:, :, l].astype(np.float64)
            out[:, :, l] = ndimage.zoom(
                band, zoom_factor, order=1, mode="constant"
            )[: band.shape[0], : band.shape[1]]
        return out

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        warp_coeff = self._params.get("warp_coeff", 0.01)
        L = y.shape[-1] if y.ndim >= 3 else 1
        if y.ndim < 3:
            return y.copy()
        out = np.zeros_like(y, dtype=np.float64)
        for l in range(L):
            zoom_factor = 1.0 / (1.0 + warp_coeff * (l - L / 2))
            band = y[:, :, l].astype(np.float64)
            out[:, :, l] = ndimage.zoom(
                band, zoom_factor, order=1, mode="constant"
            )[: band.shape[0], : band.shape[1]]
        return out


# =========================================================================
# Sampling family
# =========================================================================


class RandomMask(BasePrimitive):
    """Random binary subsampling mask (e.g. SPC measurement matrix rows)."""

    primitive_id = "random_mask"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "sampling"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        seed = self._params.get("seed", 42)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        rate = self._params.get("sampling_rate", 0.25)
        rng = np.random.default_rng(seed)
        N = H * W
        M = max(1, int(N * rate))
        self._A = (rng.random((M, N)) > 0.5).astype(np.float64) * 2 - 1
        self._A /= np.sqrt(N)
        self._H = H
        self._W = W

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        return (self._A @ x.ravel().astype(np.float64)).astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        return (self._A.T @ y.ravel().astype(np.float64)).reshape(
            self._H, self._W
        )


class CTRadon(BasePrimitive):
    """Radon transform (CT sinogram projection).

    Uses an explicit system matrix for exact adjoint consistency.
    The matrix is precomputed at construction time by applying the
    rotation+sum operator to each standard basis vector.
    """

    primitive_id = "ct_radon"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "sampling"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self._n_angles = self._params.get("n_angles", 180)
        self._H = self._params.get("H", 64)
        self._W = self._params.get("W", 64)
        self._angles = np.linspace(0, 180, self._n_angles, endpoint=False)
        self._build_matrix()

    def _build_matrix(self) -> None:
        """Build the explicit (M, N) system matrix for exact adjoint."""
        N = self._H * self._W
        M = self._n_angles * self._W
        # Only build full matrix for small problems (N*M <= 4M elements)
        if N <= 1024:  # up to 32x32
            self._A = np.zeros((M, N), dtype=np.float64)
            for j in range(N):
                e_j = np.zeros(N, dtype=np.float64)
                e_j[j] = 1.0
                img = e_j.reshape(self._H, self._W)
                sino = self._radon_fwd(img)
                self._A[:, j] = sino.ravel()
            self._use_matrix = True
        else:
            self._use_matrix = False

    def _radon_fwd(self, img: np.ndarray) -> np.ndarray:
        """Internal Radon forward using rotation."""
        sinogram = np.zeros((self._n_angles, self._W), dtype=np.float64)
        for i, angle in enumerate(self._angles):
            rotated = ndimage.rotate(
                img, angle, reshape=False, mode="constant", order=1
            )
            sinogram[i, :] = rotated.sum(axis=0)
        return sinogram

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        img = x.astype(np.float64)
        if img.ndim != 2:
            img = img.reshape(self._H, self._W)
        if self._use_matrix:
            return (self._A @ img.ravel()).reshape(self._n_angles, self._W)
        return self._radon_fwd(img)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sino = y.astype(np.float64).ravel()
        if self._use_matrix:
            return (self._A.T @ sino).reshape(self._H, self._W)
        # Fallback: rotation-based backprojection (approximate)
        bp = np.zeros((self._H, self._W), dtype=np.float64)
        y2d = y.reshape(self._n_angles, self._W)
        for i, angle in enumerate(self._angles):
            projection = y2d[i, :]
            smeared = np.tile(projection, (self._H, 1))
            rotated = ndimage.rotate(
                smeared, -angle, reshape=False, mode="constant", order=1
            )
            bp += rotated
        return bp


class MRIKspace(BasePrimitive):
    """MRI k-space undersampling (FFT + mask)."""

    primitive_id = "mri_kspace"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "sampling"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        rate = self._params.get("sampling_rate", 0.25)
        seed = self._params.get("seed", 42)
        rng = np.random.default_rng(seed)
        self._mask = (rng.random((H, W)) < rate).astype(np.float64)
        ch, cw = H // 8, W // 8
        self._mask[H // 2 - ch : H // 2 + ch, W // 2 - cw : W // 2 + cw] = 1.0

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        kspace = np.fft.fftshift(np.fft.fft2(x.astype(np.complex128)))
        return (kspace * self._mask).astype(np.complex128)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        masked = y.astype(np.complex128) * self._mask
        return np.fft.ifft2(np.fft.ifftshift(masked)).real.astype(np.float64)


class TemporalMask(BasePrimitive):
    """Time-varying coded aperture masking (CACTI-style)."""

    primitive_id = "temporal_mask"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "encoding"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        T = self._params.get("T", 8)
        seed = self._params.get("seed", 42)
        rng = np.random.default_rng(seed)
        base_mask = (rng.random((H, W)) > 0.5).astype(np.float64)
        self._masks = np.zeros((H, W, T), dtype=np.float64)
        for t in range(T):
            self._masks[:, :, t] = np.roll(base_mask, t, axis=0)

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        if x.ndim == 2:
            return x.astype(np.float64) * self._masks[:, :, 0]
        return np.sum(
            x.astype(np.float64) * self._masks[: x.shape[0], : x.shape[1], : x.shape[2]],
            axis=2,
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        T = self._masks.shape[2]
        out = np.zeros((*y.shape, T), dtype=np.float64)
        for t in range(T):
            out[:, :, t] = y.astype(np.float64) * self._masks[: y.shape[0], : y.shape[1], t]
        return out


# =========================================================================
# Nonlinearity family
# =========================================================================


class MagnitudeSq(BasePrimitive):
    """|x|^2 (intensity from complex field)."""

    primitive_id = "magnitude_sq"
    _is_linear = False
    _physics_tier = "tier1_approx"
    _physics_subrole = "transduction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        return np.abs(x.astype(np.complex128)) ** 2


class Saturation(BasePrimitive):
    """Soft saturation (clamp to [0, max_val])."""

    primitive_id = "saturation"
    _is_linear = False
    _physics_tier = "tier1_approx"
    _physics_subrole = "transduction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        max_val = self._params.get("max_val", 1.0)
        return np.clip(x.astype(np.float64), 0.0, max_val)


class LogCompress(BasePrimitive):
    """Logarithmic compression: log(1 + alpha * x)."""

    primitive_id = "log_compress"
    _is_linear = False
    _physics_tier = "tier1_approx"
    _physics_subrole = "transduction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        alpha = self._params.get("alpha", 1.0)
        return np.log1p(alpha * np.maximum(x.astype(np.float64), 0.0))


# =========================================================================
# Noise family (forward-only, no adjoint)
# =========================================================================


class PoissonNoise(BasePrimitive):
    """Poisson noise model."""

    primitive_id = "poisson"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = False
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        peak = self._params.get("peak_photons", 1e4)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        scaled = np.maximum(x.astype(np.float64) * peak, 0.0)
        noisy = rng.poisson(scaled).astype(np.float64) / peak
        return noisy


class GaussianNoise(BasePrimitive):
    """Additive Gaussian noise."""

    primitive_id = "gaussian"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = True
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 0.01)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        return x.astype(np.float64) + rng.normal(0, sigma, size=x.shape)


class PoissonGaussianNoise(BasePrimitive):
    """Mixed Poisson-Gaussian noise (shot + read)."""

    primitive_id = "poisson_gaussian"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = False
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        peak = self._params.get("peak_photons", 1e4)
        read_sigma = self._params.get("read_sigma", 0.01)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        scaled = np.maximum(x.astype(np.float64) * peak, 0.0)
        shot = rng.poisson(scaled).astype(np.float64) / peak
        return shot + rng.normal(0, read_sigma, size=x.shape)


class FPN(BasePrimitive):
    """Fixed-pattern noise (multiplicative + additive)."""

    primitive_id = "fpn"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = True
    _physics_tier = "tier1_approx"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        seed = self._params.get("seed", 42)
        rng = np.random.default_rng(seed)
        H = self._params.get("H", 64)
        W = self._params.get("W", 64)
        gain_sigma = self._params.get("gain_sigma", 0.02)
        offset_sigma = self._params.get("offset_sigma", 0.01)
        self._gain = 1.0 + rng.normal(0, gain_sigma, (H, W))
        self._offset = rng.normal(0, offset_sigma, (H, W))

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        g = self._gain[: x.shape[0], : x.shape[1]]
        o = self._offset[: x.shape[0], : x.shape[1]]
        return x.astype(np.float64) * g + o


# =========================================================================
# Temporal family
# =========================================================================


class FrameIntegration(BasePrimitive):
    """Sum along the temporal/spectral axis."""

    primitive_id = "frame_integration"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "encoding"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        axis = self._params.get("axis", -1)
        return np.sum(x.astype(np.float64), axis=axis)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        T = self._params.get("T", 8)
        return np.stack([y.astype(np.float64)] * T, axis=-1)


class MotionWarp(BasePrimitive):
    """Shift image by (dx, dy) to simulate inter-frame motion."""

    primitive_id = "motion_warp"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        dx = self._params.get("dx", 0.0)
        dy = self._params.get("dy", 0.0)
        return ndimage.shift(x.astype(np.float64), [dy, dx], order=1, mode="constant")

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        dx = self._params.get("dx", 0.0)
        dy = self._params.get("dy", 0.0)
        return ndimage.shift(y.astype(np.float64), [-dy, -dx], order=1, mode="constant")


# =========================================================================
# Readout family
# =========================================================================


class Quantize(BasePrimitive):
    """ADC quantization to fixed bit depth."""

    primitive_id = "quantize"
    _is_linear = False
    _is_differentiable = False
    _physics_tier = "tier0_geometry"
    _physics_subrole = "transduction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        bit_depth = self._params.get("bit_depth", 16)
        max_val = 2**bit_depth - 1
        return np.round(
            np.clip(x.astype(np.float64), 0.0, 1.0) * max_val
        ) / max_val


class ADCClip(BasePrimitive):
    """Clip to ADC range [0, full_well]."""

    primitive_id = "adc_clip"
    _is_linear = False
    _physics_tier = "tier0_geometry"
    _physics_subrole = "transduction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        full_well = self._params.get("full_well", 1.0)
        return np.clip(x.astype(np.float64), 0.0, full_well)


# =========================================================================
# Identity (pass-through)
# =========================================================================


class Identity(BasePrimitive):
    """Identity / pass-through primitive."""

    primitive_id = "identity"
    _is_linear = True
    _physics_tier = "tier0_geometry"
    _physics_subrole = "relay"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        return x.copy().astype(np.float64)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        return y.copy().astype(np.float64)


# =========================================================================
# Summation (reduce axis)
# =========================================================================


class SumAxis(BasePrimitive):
    """Sum (reduce) along a specified axis."""

    primitive_id = "sum_axis"
    _is_linear = True
    _physics_tier = "tier0_geometry"
    _physics_subrole = "encoding"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        axis = self._params.get("axis", -1)
        return np.sum(x.astype(np.float64), axis=axis)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        axis = self._params.get("axis", -1)
        n = self._params.get("n", 8)
        return np.stack([y.astype(np.float64)] * n, axis=axis)


# =========================================================================
# Source family (role=source, is_linear=True)
# =========================================================================


class PhotonSource(BasePrimitive):
    """Photon illumination source: scales input by strength, applies spatial profile."""

    primitive_id = "photon_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (x.astype(np.float64) * strength)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (y.astype(np.float64) * strength)


class XRaySource(BasePrimitive):
    """X-ray source: photon source with keV spectrum hint."""

    primitive_id = "xray_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (x.astype(np.float64) * strength)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (y.astype(np.float64) * strength)


class AcousticSource(BasePrimitive):
    """Acoustic carrier emission source."""

    primitive_id = "acoustic_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (x.astype(np.float64) * strength)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (y.astype(np.float64) * strength)


class SpinSource(BasePrimitive):
    """RF excitation source for MRI/NMR."""

    primitive_id = "spin_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (x.astype(np.float64) * strength)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (y.astype(np.float64) * strength)


class GenericSource(BasePrimitive):
    """Identity-like fallback source for Matrix/NeRF/3DGS modalities."""

    primitive_id = "generic_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (x.astype(np.float64) * strength)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        strength = self._params.get("strength", 1.0)
        return (y.astype(np.float64) * strength)


# =========================================================================
# Sensor family (role=sensor, is_linear=True)
# =========================================================================


class PhotonSensor(BasePrimitive):
    """Photon sensor: QE * gain + dark_current -> expected electron count.

    Supports multi-channel output via n_channels parameter.
    """

    primitive_id = "photon_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        qe = self._params.get("quantum_efficiency", 0.9)
        gain = self._params.get("gain", 1.0)
        dark_current = self._params.get("dark_current", 0.0)
        n_channels = int(self._params.get("n_channels", 1))

        base = x.astype(np.float64) * qe * gain + dark_current

        if n_channels > 1:
            channel_responses = self._params.get("channel_responses", None)
            if channel_responses is not None:
                # Apply per-channel response
                result = np.stack([base * r for r in channel_responses[:n_channels]], axis=0)
            else:
                # Replicate across channels
                result = np.stack([base] * n_channels, axis=0)
            return result
        return base

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        qe = self._params.get("quantum_efficiency", 0.9)
        gain = self._params.get("gain", 1.0)
        n_channels = int(self._params.get("n_channels", 1))

        if n_channels > 1:
            channel_responses = self._params.get("channel_responses", None)
            if channel_responses is not None:
                result = sum(y[c] * channel_responses[c] for c in range(min(n_channels, y.shape[0])))
            else:
                result = np.sum(y, axis=0)
            return result.astype(np.float64) * qe * gain
        return (y.astype(np.float64) * qe * gain)


class CoilSensor(BasePrimitive):
    """MRI coil sensitivity: complex multiply by coil map.

    Supports multi-coil output via n_coils parameter.
    """

    primitive_id = "coil_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sensitivity = self._params.get("sensitivity", 1.0)
        n_coils = int(self._params.get("n_coils", 1))

        x_c = x.astype(np.complex128)

        if n_coils > 1:
            sensitivity_maps = self._params.get("sensitivity_maps", None)
            if sensitivity_maps is not None:
                # sensitivity_maps: (n_coils, *spatial_shape) complex
                maps = np.asarray(sensitivity_maps, dtype=np.complex128)
                return maps[:n_coils] * x_c
            else:
                # Default: generate random-phase sensitivities
                rng = np.random.RandomState(42)
                phases = np.exp(2j * np.pi * rng.rand(n_coils, 1, 1))
                return np.stack([x_c * sensitivity * phases[c, 0, 0] for c in range(n_coils)], axis=0)
        return x_c * sensitivity

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sensitivity = self._params.get("sensitivity", 1.0)
        n_coils = int(self._params.get("n_coils", 1))

        if n_coils > 1:
            sensitivity_maps = self._params.get("sensitivity_maps", None)
            if sensitivity_maps is not None:
                maps = np.asarray(sensitivity_maps, dtype=np.complex128)
                return np.sum(np.conj(maps[:n_coils]) * y[:n_coils], axis=0)
            else:
                rng = np.random.RandomState(42)
                phases = np.exp(2j * np.pi * rng.rand(n_coils, 1, 1))
                conj_phases = np.conj(phases)
                return np.sum(
                    np.stack([y[c] * np.conj(sensitivity) * conj_phases[c, 0, 0]
                             for c in range(min(n_coils, y.shape[0]))], axis=0),
                    axis=0
                )
        return (y.astype(np.complex128) * np.conj(sensitivity))


class TransducerSensor(BasePrimitive):
    """Acoustic-to-voltage conversion for ultrasound/photoacoustic."""

    primitive_id = "transducer_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sensitivity = self._params.get("sensitivity", 1.0)
        return (x.astype(np.float64) * sensitivity)

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sensitivity = self._params.get("sensitivity", 1.0)
        return (y.astype(np.float64) * sensitivity)


class GenericSensor(BasePrimitive):
    """Identity sensor with gain for Matrix/NeRF/3DGS modalities.

    Supports multi-channel output via n_channels parameter.
    """

    primitive_id = "generic_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        gain = self._params.get("gain", 1.0)
        n_channels = int(self._params.get("n_channels", 1))
        base = x.astype(np.float64) * gain
        if n_channels > 1:
            return np.stack([base] * n_channels, axis=0)
        return base

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        gain = self._params.get("gain", 1.0)
        n_channels = int(self._params.get("n_channels", 1))
        if n_channels > 1:
            return np.sum(y, axis=0).astype(np.float64) * gain
        return (y.astype(np.float64) * gain)


# =========================================================================
# Sensor noise family (role=noise, is_linear=False, is_stochastic=True)
# =========================================================================


class PoissonGaussianSensorNoise(BasePrimitive):
    """Poisson shot + Gaussian read noise with likelihood method.

    Forward: shot(peak * x_clean) / peak + N(0, read_sigma)
    """

    primitive_id = "poisson_gaussian_sensor"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = False
    _node_role = "noise"
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        peak = self._params.get("peak_photons", 1e4)
        read_sigma = self._params.get("read_sigma", 0.01)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        scaled = np.maximum(x.astype(np.float64) * peak, 0.0)
        shot = rng.poisson(scaled).astype(np.float64) / peak
        return shot + rng.normal(0, read_sigma, size=x.shape)

    def likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        """Mixed Poisson-Gaussian NLL."""
        peak = self._params.get("peak_photons", 1e4)
        read_sigma = self._params.get("read_sigma", 0.01)
        eps = 1e-10
        lam = np.maximum(y_clean.ravel().astype(np.float64) * peak, eps)
        y_flat = y.ravel().astype(np.float64) * peak
        # Poisson contribution + Gaussian read noise
        poisson_nll = float(np.sum(lam - y_flat * np.log(lam)))
        gauss_nll = float(0.5 * np.sum(
            (y.ravel() - y_clean.ravel()) ** 2 / (read_sigma ** 2)
        ))
        return poisson_nll + gauss_nll


class ComplexGaussianSensorNoise(BasePrimitive):
    """Complex Gaussian noise for MRI k-space data."""

    primitive_id = "complex_gaussian_sensor"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = True
    _node_role = "noise"
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 0.01)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, x.shape) + 1j * rng.normal(0, sigma, x.shape)
        return x.astype(np.complex128) + noise

    def likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        """Complex Gaussian NLL."""
        sigma = self._params.get("sigma", 0.01)
        r = (y.ravel() - y_clean.ravel()).astype(np.complex128)
        return float(np.sum(np.abs(r) ** 2 / (sigma ** 2)).real)


class PoissonOnlySensorNoise(BasePrimitive):
    """Poisson-only noise for photon-counting detectors (CT)."""

    primitive_id = "poisson_only_sensor"
    _is_linear = False
    _is_stochastic = True
    _is_differentiable = False
    _node_role = "noise"
    _physics_tier = "tier1_approx"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        peak = self._params.get("peak_photons", 1e5)
        seed = self._params.get("seed", 0)
        rng = np.random.default_rng(seed)
        scaled = np.maximum(x.astype(np.float64) * peak, 0.0)
        return rng.poisson(scaled).astype(np.float64) / peak

    def likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        """Poisson NLL."""
        peak = self._params.get("peak_photons", 1e5)
        eps = 1e-10
        lam = np.maximum(y_clean.ravel().astype(np.float64) * peak, eps)
        y_flat = y.ravel().astype(np.float64) * peak
        return float(np.sum(lam - y_flat * np.log(lam)))


# =========================================================================
# Multi-input family
# =========================================================================


class Interference(BasePrimitive):
    """Two-beam interference: signal + reference."""
    primitive_id = "interference"
    _is_linear = True
    _n_inputs = 2
    _physics_tier = "tier1_approx"
    _physics_subrole = "interaction"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        # Single-input fallback: identity
        return x

    def forward_multi(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        signal = inputs.get("signal", next(iter(inputs.values())))
        ref = inputs.get("reference", np.zeros_like(signal))
        return np.abs(signal + ref) ** 2

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError("Adjoint not available for non-linear interference")


class FourierRelay(BasePrimitive):
    """Fourier-domain relay: FFT -> transfer function -> IFFT.

    Tier-1 approximation for free-space propagation, low-pass filtering,
    or band-pass filtering in the frequency domain.
    """
    primitive_id = "fourier_relay"
    _is_linear = True
    _physics_tier = "tier1_approx"
    _physics_subrole = "relay"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        transfer_function = self._params.get("transfer_function", "free_space")
        wavelength = self._params.get("wavelength_m", 0.5e-6)
        distance = self._params.get("propagation_distance_m", 1e-3)
        pixel_size = self._params.get("pixel_size_m", 1e-6)

        H, W = x.shape[-2], x.shape[-1]
        fy = np.fft.fftfreq(H, d=pixel_size).reshape(-1, 1)
        fx = np.fft.fftfreq(W, d=pixel_size).reshape(1, -1)

        if transfer_function == "free_space":
            kz_sq = np.maximum(1.0 / wavelength**2 - fx**2 - fy**2, 0.0)
            H_tf = np.exp(2j * np.pi * distance * np.sqrt(kz_sq))
        elif transfer_function == "low_pass":
            cutoff = self._params.get("cutoff_freq", 0.5 / pixel_size)
            H_tf = ((fx**2 + fy**2) <= cutoff**2).astype(np.complex128)
        elif transfer_function == "band_pass":
            low = self._params.get("low_freq", 0.1 / pixel_size)
            high = self._params.get("high_freq", 0.4 / pixel_size)
            r2 = fx**2 + fy**2
            H_tf = ((r2 >= low**2) & (r2 <= high**2)).astype(np.complex128)
        else:
            H_tf = np.ones((H, W), dtype=np.complex128)

        self._H_tf = H_tf  # Cache for adjoint
        X = np.fft.fft2(x.astype(np.complex128))
        result = np.fft.ifft2(X * H_tf)
        return np.real(result) if np.isrealobj(x) else result

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        if not hasattr(self, '_H_tf'):
            # Build transfer function (in case adjoint called without forward first)
            dummy = np.zeros(y.shape)
            self.forward(dummy)
        H_tf_conj = np.conj(self._H_tf)
        Y = np.fft.fft2(y.astype(np.complex128))
        result = np.fft.ifft2(Y * H_tf_conj)
        return np.real(result) if np.isrealobj(y) else result


class MaxwellInterface(BasePrimitive):
    """Tier-2 Maxwell solver interface stub (FDTD/BPM).

    Defines the correct input/output interface for full-wave electromagnetic
    simulation but raises NotImplementedError until a backend is integrated.
    """
    primitive_id = "maxwell_interface"
    _is_linear = True
    _physics_tier = "tier2_full"
    _physics_subrole = "propagation"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        backend = self._params.get("backend", "none")
        raise NotImplementedError(
            f"Maxwell solver (FDTD/BPM) not yet integrated. "
            f"Set up via MaxwellInterface.configure(backend='meep'|'tidy3d'|'custom'). "
            f"Current backend='{backend}'"
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError(
            "Maxwell adjoint requires full-wave solver backend."
        )


# =========================================================================
# Correction family (role=correction)
# =========================================================================


class AffineCorrectionNode(BasePrimitive):
    """Per-element affine correction: y_corrected = gain * y + offset."""

    primitive_id = "affine_correction"
    _is_linear = True
    _node_role = "correction"
    _physics_tier = "tier0_geometry"

    def forward(self, x, **params):
        gain = self._params.get("gain", 1.0)
        offset = self._params.get("offset", 0.0)
        if isinstance(gain, (list, np.ndarray)):
            gain = np.asarray(gain, dtype=np.float64)
        if isinstance(offset, (list, np.ndarray)):
            offset = np.asarray(offset, dtype=np.float64)
        return x.astype(np.float64) * gain + offset

    def adjoint(self, y, **params):
        gain = self._params.get("gain", 1.0)
        if isinstance(gain, (list, np.ndarray)):
            gain = np.asarray(gain, dtype=np.float64)
        return y.astype(np.float64) * gain  # adjoint of affine: gain^T * y


class ResidualCorrectionNode(BasePrimitive):
    """Additive residual correction: y_corrected = y + residual."""

    primitive_id = "residual_correction"
    _is_linear = True
    _node_role = "correction"
    _physics_tier = "tier0_geometry"

    def forward(self, x, **params):
        residual = self._params.get("residual", 0.0)
        if isinstance(residual, (list, np.ndarray)):
            residual = np.asarray(residual, dtype=np.float64)
        return x.astype(np.float64) + residual

    def adjoint(self, y, **params):
        return y.astype(np.float64)  # adjoint of identity + constant = identity


class FieldMapCorrectionNode(BasePrimitive):
    """Multiplicative field map correction (e.g., B0 inhomogeneity)."""

    primitive_id = "field_map_correction"
    _is_linear = True
    _node_role = "correction"
    _physics_tier = "tier1_approx"

    def forward(self, x, **params):
        field_map = self._params.get("field_map", 1.0)
        if isinstance(field_map, (list, np.ndarray)):
            field_map = np.asarray(field_map, dtype=np.float64)
        return x.astype(np.float64) * field_map

    def adjoint(self, y, **params):
        field_map = self._params.get("field_map", 1.0)
        if isinstance(field_map, (list, np.ndarray)):
            field_map = np.asarray(field_map, dtype=np.float64)
        return y.astype(np.float64) * field_map  # self-adjoint for real field map


# =========================================================================
# R6: Physically-correct primitives for CT, Photoacoustic, NeRF, 3DGS
# =========================================================================


class BeerLambert(BasePrimitive):
    """Beer-Lambert transmission: I = I_0 * exp(-sinogram).

    Models photon transmission through an attenuating medium.
    Input is attenuation line-integral (sinogram), output is transmitted
    photon intensity.  Non-linear (exponential).
    """

    primitive_id = "beer_lambert"
    _is_linear = False
    _node_role = "transport"
    _physics_tier = "tier1_approx"
    _physics_subrole = "transduction"

    def forward(self, x, **params):
        I_0 = self._params.get("I_0", 10000.0)
        x = np.asarray(x, dtype=np.float64)
        self._cached_sinogram = x
        return I_0 * np.exp(-x)

    def adjoint(self, y, **params):
        I_0 = self._params.get("I_0", 10000.0)
        sinogram = getattr(self, "_cached_sinogram", np.zeros_like(y))
        return -I_0 * np.exp(-sinogram) * y


class OpticalAbsorption(BasePrimitive):
    """Photoacoustic optical absorption: p0 = grueneisen * mu_a * fluence.

    Converts absorbed optical energy to initial acoustic pressure via
    the Grüneisen parameter.  Linear in fluence.
    Carrier transition: photon → acoustic.
    """

    primitive_id = "optical_absorption"
    _is_linear = True
    _node_role = "interaction"
    _physics_tier = "tier1_approx"
    _physics_subrole = "interaction"
    _carrier_type = "acoustic"

    def forward(self, x, **params):
        grueneisen = self._params.get("grueneisen", 0.8)
        mu_a = self._params.get("mu_a", 1.0)
        return grueneisen * mu_a * np.asarray(x, dtype=np.float64)

    def adjoint(self, y, **params):
        grueneisen = self._params.get("grueneisen", 0.8)
        mu_a = self._params.get("mu_a", 1.0)
        return grueneisen * mu_a * np.asarray(y, dtype=np.float64)


class AcousticPropagation(BasePrimitive):
    """Simplified acoustic wave propagation (circular mean projection).

    Initial implementation: simplified projection operator modeling
    acoustic wave detection at transducer elements.
    """

    primitive_id = "acoustic_propagation"
    _is_linear = True
    _node_role = "transport"
    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"
    _carrier_type = "acoustic"

    def forward(self, x, **params):
        x = np.asarray(x, dtype=np.float64)
        speed = self._params.get("speed_of_sound", 1500.0)
        n_sensors = self._params.get("n_sensors", x.shape[0])
        # Simplified: project initial pressure onto sensor lines
        # Use Radon-like projection as proxy for time-of-flight integration
        n_angles = n_sensors
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        H, W = x.shape[-2], x.shape[-1]
        result = np.zeros((n_angles, max(H, W)), dtype=np.float64)
        for i, angle in enumerate(angles):
            rotated = ndimage.rotate(x, angle, reshape=False, order=1, mode="constant")
            result[i, :H] = rotated.sum(axis=1)[:max(H, W)]
        return result

    def adjoint(self, y, **params):
        # Back-projection (time-reversal approximation)
        x_shape = self._params.get("x_shape", (64, 64))
        H, W = x_shape[-2], x_shape[-1]
        n_angles = y.shape[0]
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        result = np.zeros((H, W), dtype=np.float64)
        for i, angle in enumerate(angles):
            proj = y[i, :H]
            bp = np.tile(proj[:, None], (1, W))
            result += ndimage.rotate(bp, -angle, reshape=False, order=1, mode="constant")
        return result / max(n_angles, 1)


class VolumeRenderingStub(BasePrimitive):
    """Volume rendering stub (tier3_learned).

    Initial implementation: maximum intensity projection (MIP).
    Full differentiable volume rendering requires PyTorch/JAX backend.
    """

    primitive_id = "volume_rendering_stub"
    _is_linear = False
    _node_role = "transport"
    _physics_tier = "tier3_learned"
    _physics_subrole = "propagation"

    def forward(self, x, **params):
        x = np.asarray(x, dtype=np.float64)
        render_mode = self._params.get("render_mode", "mip")
        if render_mode == "mip":
            if x.ndim == 3:
                return x.max(axis=0)
            return x
        elif render_mode == "quadrature":
            raise NotImplementedError(
                "Full volume rendering requires PyTorch/JAX backend"
            )
        return x

    def adjoint(self, y, **params):
        raise NotImplementedError(
            "Volume rendering adjoint requires differentiable backend"
        )


class GaussianSplattingStub(BasePrimitive):
    """3D Gaussian splatting projection stub (tier3_learned).

    Initial implementation: weighted sum of 2D Gaussians.
    Full differentiable splatting requires CUDA/PyTorch backend.
    """

    primitive_id = "gaussian_splatting_stub"
    _is_linear = False
    _node_role = "transport"
    _physics_tier = "tier3_learned"
    _physics_subrole = "propagation"

    def forward(self, x, **params):
        x = np.asarray(x, dtype=np.float64)
        image_size = self._params.get("image_size", [64, 64])
        H, W = image_size[0], image_size[1]
        # Stub: render as blurred version of input
        if x.ndim == 3:
            rendered = x.sum(axis=0)
        else:
            rendered = x.copy()
        # Resize to target
        if rendered.shape != (H, W):
            zoom_h = H / max(rendered.shape[0], 1)
            zoom_w = W / max(rendered.shape[1], 1)
            rendered = ndimage.zoom(rendered, (zoom_h, zoom_w), order=1)
        return rendered

    def adjoint(self, y, **params):
        raise NotImplementedError(
            "Gaussian splatting adjoint requires differentiable backend"
        )


class GaussianSensorNoise(BasePrimitive):
    """Real-valued Gaussian sensor noise (thermal + electronic).

    Unlike ComplexGaussianSensorNoise, this operates in the real domain only.
    Suitable for acoustic/electronic imaging modalities (PA, ultrasound).
    """

    primitive_id = "gaussian_sensor_noise"
    _is_linear = False
    _is_stochastic = True
    _node_role = "noise"
    _physics_tier = "tier1_approx"
    _physics_subrole = None

    def forward(self, x, **params):
        x = np.asarray(x, dtype=np.float64)
        sigma = self._params.get("sigma", 0.01)
        seed = self._params.get("seed", 0)
        rng = np.random.RandomState(seed)
        return x + sigma * rng.randn(*x.shape)

    def adjoint(self, y, **params):
        return np.asarray(y, dtype=np.float64)

    def likelihood(self, y, y_clean):
        """Gaussian negative log-likelihood."""
        sigma = self._params.get("sigma", 0.01)
        diff = np.asarray(y, dtype=np.float64) - np.asarray(y_clean, dtype=np.float64)
        return 0.5 * np.sum(diff**2) / (sigma**2)


# =========================================================================
# Registry
# =========================================================================

_ALL_PRIMITIVES: List[type] = [
    # Propagation
    FresnelProp,
    AngularSpectrum,
    RayTrace,
    # PSF / Convolution
    Conv2d,
    Conv3d,
    DeconvRL,
    # Modulation
    CodedMask,
    DMDPattern,
    SIMPattern,
    # Warp / Dispersion
    SpectralDispersion,
    ChromaticWarp,
    # Sampling
    RandomMask,
    CTRadon,
    MRIKspace,
    TemporalMask,
    # Nonlinearity
    MagnitudeSq,
    Saturation,
    LogCompress,
    # Noise (legacy)
    PoissonNoise,
    GaussianNoise,
    PoissonGaussianNoise,
    FPN,
    # Temporal
    FrameIntegration,
    MotionWarp,
    # Readout
    Quantize,
    ADCClip,
    # Identity
    Identity,
    SumAxis,
    # Source family
    PhotonSource,
    XRaySource,
    AcousticSource,
    SpinSource,
    GenericSource,
    # Sensor family
    PhotonSensor,
    CoilSensor,
    TransducerSensor,
    GenericSensor,
    # Sensor noise family
    PoissonGaussianSensorNoise,
    ComplexGaussianSensorNoise,
    PoissonOnlySensorNoise,
    # Multi-input family
    Interference,
    # Tier-based primitives
    FourierRelay,
    MaxwellInterface,
    # Correction family
    AffineCorrectionNode,
    ResidualCorrectionNode,
    FieldMapCorrectionNode,
    # R6: Physically-correct primitives
    BeerLambert,
    OpticalAbsorption,
    AcousticPropagation,
    VolumeRenderingStub,
    GaussianSplattingStub,
    GaussianSensorNoise,
]

PRIMITIVE_REGISTRY: Dict[str, type] = {
    cls.primitive_id: cls for cls in _ALL_PRIMITIVES
}


def get_primitive(primitive_id: str, params: Optional[Dict[str, Any]] = None) -> BasePrimitive:
    """Look up a primitive by ID and instantiate with the given params.

    Raises KeyError if the primitive_id is not registered.
    """
    if primitive_id not in PRIMITIVE_REGISTRY:
        raise KeyError(
            f"Unknown primitive_id '{primitive_id}'. "
            f"Available: {sorted(PRIMITIVE_REGISTRY.keys())}"
        )
    cls = PRIMITIVE_REGISTRY[primitive_id]
    return cls(params=params)
