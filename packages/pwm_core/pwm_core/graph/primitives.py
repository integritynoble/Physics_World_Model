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
# Phase 1: Extended node families (v4)
# =========================================================================


# P1.2: ExplicitLinearOperator
class ExplicitLinearOperator(BasePrimitive):
    """Accepts a provided A as dense/sparse/callback for operator correction."""
    primitive_id = "explicit_linear_operator"
    _is_linear = True
    _node_role = "transport"
    _physics_tier = "tier0_geometry"
    _physics_subrole = "relay"

    def __init__(self, params=None):
        super().__init__(params)
        self._matrix = self._params.get("matrix", None)
        self._forward_fn = self._params.get("forward_fn", None)
        self._adjoint_fn = self._params.get("adjoint_fn", None)
        if self._matrix is not None:
            self._matrix = np.asarray(self._matrix)

    def forward(self, x, **params):
        if self._matrix is not None:
            return (self._matrix @ x.ravel()).reshape(self._params.get("y_shape", (-1,)))
        elif self._forward_fn is not None:
            return self._forward_fn(x)
        return x

    def adjoint(self, y, **params):
        if self._matrix is not None:
            return (self._matrix.T @ y.ravel()).reshape(self._params.get("x_shape", (-1,)))
        elif self._adjoint_fn is not None:
            return self._adjoint_fn(y)
        return y

    def compute_hash(self):
        if self._matrix is not None:
            return hashlib.sha256(self._matrix.tobytes()).hexdigest()[:16]
        return "callback"


# P1.3: Electron source + sensor (D2-compliant)
class ElectronBeamSource(BasePrimitive):
    """Electron beam source -- does NOT consume x (Pattern B).
    Outputs incident beam state scaled by beam_current.
    """
    primitive_id = "electron_beam_source"
    _is_linear = True
    _node_role = "source"
    _physics_tier = "tier0_geometry"
    _physics_subrole = None
    _carrier_type = "electron"

    def forward(self, x, **params):
        current = self._params.get("beam_current_na", 1.0)
        return current * np.ones_like(x)

    def adjoint(self, y, **params):
        current = self._params.get("beam_current_na", 1.0)
        return current * y


class ElectronDetectorSensor(BasePrimitive):
    """Electron detector sensor (SE/BSE/BF/ADF/HAADF)."""
    primitive_id = "electron_detector_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"
    _carrier_type = "electron"

    def forward(self, x, **params):
        ce = self._params.get("collection_efficiency", 0.5)
        gain = self._params.get("gain", 1.0)
        return ce * gain * np.asarray(x, dtype=np.float64)

    def adjoint(self, y, **params):
        ce = self._params.get("collection_efficiency", 0.5)
        gain = self._params.get("gain", 1.0)
        return ce * gain * np.asarray(y, dtype=np.float64)


# P1.4: Specialized sensors
class SinglePixelSensor(BasePrimitive):
    """Single-pixel detector for SPC modality."""
    primitive_id = "single_pixel_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x, **params):
        n_patterns = self._params.get("n_patterns", 64)
        rng = np.random.RandomState(self._params.get("seed", 0))
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        N = len(x_flat)
        patterns = rng.randn(n_patterns, N)
        return patterns @ x_flat

    def adjoint(self, y, **params):
        n_patterns = self._params.get("n_patterns", 64)
        rng = np.random.RandomState(self._params.get("seed", 0))
        x_shape = self._params.get("x_shape", (64, 64))
        N = int(np.prod(x_shape))
        patterns = rng.randn(n_patterns, N)
        return (patterns.T @ np.asarray(y).ravel()).reshape(x_shape)


class XRayDetectorSensor(BasePrimitive):
    """X-ray flat panel detector with scintillator."""
    primitive_id = "xray_detector_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x, **params):
        eff = self._params.get("scintillator_efficiency", 0.8)
        gain = self._params.get("gain", 1.0)
        offset = self._params.get("offset", 0.0)
        return gain * eff * np.asarray(x, dtype=np.float64) + offset

    def adjoint(self, y, **params):
        eff = self._params.get("scintillator_efficiency", 0.8)
        gain = self._params.get("gain", 1.0)
        return gain * eff * np.asarray(y, dtype=np.float64)


class AcousticReceiveSensor(BasePrimitive):
    """Ultrasound receive array sensor."""
    primitive_id = "acoustic_receive_sensor"
    _is_linear = True
    _node_role = "sensor"
    _physics_tier = "tier0_geometry"

    def forward(self, x, **params):
        sensitivity = self._params.get("sensitivity", 1.0)
        return sensitivity * np.asarray(x, dtype=np.float64)

    def adjoint(self, y, **params):
        sensitivity = self._params.get("sensitivity", 1.0)
        return sensitivity * np.asarray(y, dtype=np.float64)


# P1.5: Correlated noise (simulation-only)
class CorrelatedNoiseSensor(BasePrimitive):
    """Correlated noise -- simulation-only, blocks Mode C until whitening."""
    primitive_id = "correlated_noise_sensor"
    _is_linear = False
    _is_stochastic = True
    _node_role = "noise"
    _physics_tier = "tier1_approx"
    _simulation_only = True

    def forward(self, x, **params):
        sigma = self._params.get("base_sigma", 0.1)
        corr_type = self._params.get("correlation_type", "none")
        seed = self._params.get("seed", 0)
        rng = np.random.RandomState(seed)
        noise = rng.randn(*np.asarray(x).shape).astype(np.float64) * sigma

        if corr_type == "spatial":
            length = self._params.get("correlation_length", 2.0)
            noise = ndimage.gaussian_filter(noise, sigma=length)
            noise = noise * sigma / max(np.std(noise), 1e-12)
        elif corr_type == "1_over_f":
            f_noise = np.fft.fftn(noise)
            shape = noise.shape
            freqs = [np.fft.fftfreq(s) for s in shape]
            grid = np.meshgrid(*freqs, indexing='ij')
            f_mag = np.sqrt(sum(g**2 for g in grid))
            f_mag[f_mag == 0] = 1.0
            f_noise = f_noise / f_mag
            noise = np.real(np.fft.ifftn(f_noise)) * sigma

        return np.asarray(x, dtype=np.float64) + noise

    def adjoint(self, y, **params):
        raise NotImplementedError("Correlated noise has no adjoint")

    def likelihood(self, y, y_clean):
        raise NotImplementedError(
            "Correlated noise likelihood requires whitening. "
            "Use simulation-only mode."
        )


# P1.6: Element primitives for new modalities

class ThinObjectPhase(BasePrimitive):
    """TEM thin-object phase approximation -- multi-input (incident, x)."""
    primitive_id = "thin_object_phase"
    _is_linear = False
    _node_role = "interaction"
    _physics_subrole = "interaction"
    _physics_tier = "tier1_approx"
    _carrier_type = "electron"
    _n_inputs = 2
    _physics_validity_regime = "thin_sample, weak_phase_approx"

    def forward(self, x, **params):
        sigma = self._params.get("sigma", 0.01)
        return np.asarray(x, dtype=np.float64) * np.exp(-sigma * np.abs(x))

    def forward_multi(self, inputs):
        incident = inputs.get("incident", inputs.get("0", np.ones((1,))))
        x = inputs.get("x", inputs.get("1", np.zeros_like(incident)))
        sigma = self._params.get("sigma", 0.01)
        transmission = np.exp(-sigma * np.abs(x))
        return np.asarray(incident, dtype=np.float64) * transmission

    def adjoint(self, y, **params):
        return np.asarray(y, dtype=np.float64)


class CTFTransfer(BasePrimitive):
    """TEM Contrast Transfer Function in Fourier domain."""
    primitive_id = "ctf_transfer"
    _is_linear = True
    _node_role = "transport"
    _physics_subrole = "propagation"
    _physics_tier = "tier1_approx"
    _carrier_type = "electron"
    _physics_validity_regime = "isoplanatic, no_spatial_incoherence"

    def forward(self, x, **params):
        defocus = self._params.get("defocus_nm", 100.0)
        Cs = self._params.get("Cs_mm", 1.0)
        wl = self._params.get("wavelength_pm", 2.51)  # 200kV
        x = np.asarray(x, dtype=np.float64)
        fx = np.fft.fft2(x)
        ny, nx = x.shape[-2:]
        fy = np.fft.fftfreq(ny)
        fxx = np.fft.fftfreq(nx)
        FY, FX = np.meshgrid(fy, fxx, indexing='ij')
        q2 = FY**2 + FX**2
        wl_nm = wl * 1e-3
        chi = np.pi * wl_nm * defocus * q2 - 0.5 * np.pi * Cs * 1e6 * wl_nm**3 * q2**2
        ctf = np.sin(chi)
        return np.real(np.fft.ifft2(fx * ctf))

    def adjoint(self, y, **params):
        return self.forward(y, **params)  # CTF is self-adjoint for real CTF


class YieldModel(BasePrimitive):
    """SEM yield model -- multi-input (incident, x)."""
    primitive_id = "yield_model"
    _is_linear = True
    _node_role = "interaction"
    _physics_subrole = "interaction"
    _physics_tier = "tier0_geometry"
    _carrier_type = "electron"
    _n_inputs = 2
    _physics_validity_regime = "homogeneous_material, normal_incidence"

    def forward(self, x, **params):
        yc = self._params.get("yield_coeff", 0.3)
        return yc * np.asarray(x, dtype=np.float64)

    def forward_multi(self, inputs):
        incident = inputs.get("incident", inputs.get("0", np.ones((1,))))
        x = inputs.get("x", inputs.get("1", np.zeros_like(incident)))
        yc = self._params.get("yield_coeff", 0.3)
        return yc * np.asarray(incident, dtype=np.float64) * np.asarray(x, dtype=np.float64)

    def adjoint(self, y, **params):
        yc = self._params.get("yield_coeff", 0.3)
        return yc * np.asarray(y, dtype=np.float64)


class BeamformDelay(BasePrimitive):
    """Ultrasound delay-and-sum beamforming."""
    primitive_id = "beamform_delay"
    _is_linear = True
    _node_role = "transport"
    _physics_subrole = "propagation"
    _physics_tier = "tier1_approx"

    def forward(self, x, **params):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim >= 2:
            return np.mean(x, axis=0)
        return x

    def adjoint(self, y, **params):
        n_elements = self._params.get("n_elements", 32)
        y = np.asarray(y, dtype=np.float64)
        return np.tile(y, (n_elements, 1)) if y.ndim >= 1 else y


class EmissionProjection(BasePrimitive):
    """PET/SPECT emission projection (system matrix)."""
    primitive_id = "emission_projection"
    _is_linear = True
    _node_role = "transport"
    _physics_subrole = "sampling"
    _physics_tier = "tier1_approx"

    def forward(self, x, **params):
        n_angles = self._params.get("n_angles", 32)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim >= 2:
            sinogram = np.zeros((n_angles, x.shape[-1]))
            angles = np.linspace(0, 180, n_angles, endpoint=False)
            for i, angle in enumerate(angles):
                rotated = ndimage.rotate(x, angle, reshape=False, order=1)
                sinogram[i] = np.sum(rotated, axis=0)
            return sinogram
        return x

    def adjoint(self, y, **params):
        n_angles = self._params.get("n_angles", 32)
        x_shape = self._params.get("x_shape", (64, 64))
        y = np.asarray(y, dtype=np.float64)
        if y.ndim >= 2:
            result = np.zeros(x_shape, dtype=np.float64)
            angles = np.linspace(0, 180, n_angles, endpoint=False)
            for i, angle in enumerate(angles):
                back = np.tile(y[i], (x_shape[0], 1))
                result += ndimage.rotate(back, -angle, reshape=False, order=1)
            return result
        return y


class ScatterModel(BasePrimitive):
    """Additive scatter estimation for X-ray/CT/PET."""
    primitive_id = "scatter_model"
    _is_linear = True
    _node_role = "transport"
    _physics_subrole = "interaction"
    _physics_tier = "tier1_approx"

    def forward(self, x, **params):
        fraction = self._params.get("scatter_fraction", 0.1)
        sigma = self._params.get("kernel_sigma", 3.0)
        x = np.asarray(x, dtype=np.float64)
        scatter = ndimage.gaussian_filter(x, sigma=sigma)
        return x + fraction * scatter

    def adjoint(self, y, **params):
        fraction = self._params.get("scatter_fraction", 0.1)
        sigma = self._params.get("kernel_sigma", 3.0)
        y = np.asarray(y, dtype=np.float64)
        scatter_adj = ndimage.gaussian_filter(y, sigma=sigma)
        return y + fraction * scatter_adj


# =========================================================================
# Phase v4+: Extended modality primitives — Element family
# =========================================================================


class ScanTrajectory(BasePrimitive):
    """Raster/spiral scan pattern sampling.

    Subsamples the input according to a scan pattern (raster, spiral, etc.).
    """

    primitive_id = "scan_trajectory"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        scan_type = self._params.get("scan_type", "raster")
        n_points = self._params.get("n_points", 64)
        dwell_time = self._params.get("dwell_time", 1.0)
        x = np.asarray(x, dtype=np.float64)
        total = int(np.prod(x.shape))
        step = max(1, total // n_points)
        flat = x.ravel()
        indices = np.arange(0, total, step)[:n_points]
        return flat[indices] * dwell_time

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        n_points = self._params.get("n_points", 64)
        dwell_time = self._params.get("dwell_time", 1.0)
        x_shape = self._params.get("x_shape", (64, 64))
        total = int(np.prod(x_shape))
        step = max(1, total // n_points)
        indices = np.arange(0, total, step)[:n_points]
        out = np.zeros(total, dtype=np.float64)
        out[indices] = np.asarray(y, dtype=np.float64).ravel()[:len(indices)] * dwell_time
        return out.reshape(x_shape)


class TimeOfFlightGate(BasePrimitive):
    """Time-of-flight binning with optional timing jitter.

    Bins input signal along first axis; applies Gaussian blur for jitter.
    """

    primitive_id = "tof_gate"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_bins = self._params.get("n_bins", 64)
        bin_width_ns = self._params.get("bin_width_ns", 1.0)
        timing_jitter_ns = self._params.get("timing_jitter_ns", 0.0)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim >= 2:
            # Bin along first axis
            H = x.shape[0]
            bin_size = max(1, H // n_bins)
            n_out = min(n_bins, H)
            out = np.zeros((n_out,) + x.shape[1:], dtype=np.float64)
            for i in range(n_out):
                start = i * bin_size
                end = min(start + bin_size, H)
                out[i] = np.sum(x[start:end], axis=0) * bin_width_ns
            result = out
        else:
            result = x * bin_width_ns
        if timing_jitter_ns > 0:
            result = ndimage.gaussian_filter1d(result, sigma=timing_jitter_ns, axis=0)
        return result

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        bin_width_ns = self._params.get("bin_width_ns", 1.0)
        return np.asarray(y, dtype=np.float64) * bin_width_ns


class CollimatorModel(BasePrimitive):
    """Acceptance cone filtering (collimator geometry).

    Applies Gaussian blur with sigma proportional to hole_diameter / collimator_length.
    """

    primitive_id = "collimator_model"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        hole_diameter = self._params.get("hole_diameter", 1.5)
        septal_thickness = self._params.get("septal_thickness", 0.2)
        collimator_length = self._params.get("collimator_length", 25.0)
        sigma = hole_diameter / max(collimator_length, 1e-12)
        return ndimage.gaussian_filter(
            np.asarray(x, dtype=np.float64), sigma=sigma, mode="reflect"
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        hole_diameter = self._params.get("hole_diameter", 1.5)
        collimator_length = self._params.get("collimator_length", 25.0)
        sigma = hole_diameter / max(collimator_length, 1e-12)
        return ndimage.gaussian_filter(
            np.asarray(y, dtype=np.float64), sigma=sigma, mode="reflect"
        )


class FluoroTemporalIntegrator(BasePrimitive):
    """Motion blur + frame integration for fluoroscopy.

    If 3D (T,H,W), sums along T axis; if 2D, applies Gaussian blur
    with motion_sigma.
    """

    primitive_id = "fluoro_temporal_integrator"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_frames = self._params.get("n_frames", 8)
        motion_sigma = self._params.get("motion_sigma", 0.0)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            return np.sum(x, axis=0)
        if motion_sigma > 0:
            return ndimage.gaussian_filter(x, sigma=motion_sigma, mode="reflect")
        return x.copy()

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        n_frames = self._params.get("n_frames", 8)
        y = np.asarray(y, dtype=np.float64)
        return np.stack([y] * n_frames, axis=0)


class FluorescenceKinetics(BasePrimitive):
    """Fluorescence lifetime/blinking model.

    Forward: x * exp(-t/lifetime). If blinking_rate > 0, multiplies by
    random on/off mask. Non-linear due to exponential and stochastic blinking.
    """

    primitive_id = "fluorescence_kinetics"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        lifetime_ns = self._params.get("lifetime_ns", 3.0)
        blinking_rate = self._params.get("blinking_rate", 0.0)
        saturation_intensity = self._params.get("saturation_intensity", float("inf"))
        x = np.asarray(x, dtype=np.float64)
        t = 1.0  # simplified single time point
        decay = np.exp(-t / max(lifetime_ns, 1e-12))
        result = x * decay
        if blinking_rate > 0:
            seed = self._params.get("seed", 0)
            rng = np.random.default_rng(seed)
            mask = (rng.random(x.shape) > blinking_rate).astype(np.float64)
            result = result * mask
        if np.isfinite(saturation_intensity):
            result = result * saturation_intensity / (np.abs(result) + saturation_intensity)
        return result


class NonlinearExcitation(BasePrimitive):
    """Multi-photon excitation: intensity^n.

    Forward: |x|^n_photons. Non-linear.
    """

    primitive_id = "nonlinear_excitation"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_photons = self._params.get("n_photons", 2)
        return np.power(np.abs(np.asarray(x, dtype=np.float64)), n_photons)


class SaturationDepletion(BasePrimitive):
    """STED-style depletion with donut PSF approximation.

    Forward: x * (1 - depletion_factor * (1 - exp(-r^2 / (2*sigma^2)))).
    Non-linear.
    """

    primitive_id = "saturation_depletion"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        depletion_factor = self._params.get("depletion_factor", 0.5)
        psf_sigma = self._params.get("psf_sigma", 1.0)
        x = np.asarray(x, dtype=np.float64)
        H, W = x.shape[-2], x.shape[-1]
        cy, cx = H / 2.0, W / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        donut = 1.0 - np.exp(-r2 / (2.0 * psf_sigma ** 2))
        if x.ndim > 2:
            donut = donut.reshape((1,) * (x.ndim - 2) + (H, W))
        return x * (1.0 - depletion_factor * donut)


class BlinkingEmitterModel(BasePrimitive):
    """PALM/STORM single-molecule blinking emitter model.

    Generates sparse point emitters from input (threshold above mean).
    Stochastic and non-linear.
    """

    primitive_id = "blinking_emitter"
    _is_linear = False
    _is_stochastic = True
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        density = self._params.get("density", 0.1)
        photons_per_emitter = self._params.get("photons_per_emitter", 1000)
        seed = self._params.get("seed", 0)
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(seed)
        threshold = np.mean(x)
        emitter_mask = (x > threshold).astype(np.float64)
        active = (rng.random(x.shape) < density).astype(np.float64)
        return emitter_mask * active * photons_per_emitter


class EvanescentFieldDecay(BasePrimitive):
    """TIRF exponential evanescent field decay.

    If 3D (Z,H,W), weights each z-slice by exp(-z / penetration_depth).
    If 2D, passthrough.
    """

    primitive_id = "evanescent_decay"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        penetration_depth = self._params.get("penetration_depth", 100.0)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            Z = x.shape[0]
            weights = np.exp(-np.arange(Z, dtype=np.float64) / max(penetration_depth, 1e-12))
            return x * weights[:, np.newaxis, np.newaxis]
        return x.copy()

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        penetration_depth = self._params.get("penetration_depth", 100.0)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 3:
            Z = y.shape[0]
            weights = np.exp(-np.arange(Z, dtype=np.float64) / max(penetration_depth, 1e-12))
            return y * weights[:, np.newaxis, np.newaxis]
        return y.copy()


class DopplerEstimator(BasePrimitive):
    """Velocity estimation from slow-time Doppler samples.

    If 3D (ensemble,H,W), computes mean phase difference between consecutive
    frames. If 2D, returns zeros. Non-linear.
    """

    primitive_id = "doppler_estimator"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        prf = self._params.get("prf", 1000.0)
        n_ensembles = self._params.get("n_ensembles", 8)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3 and x.shape[0] >= 2:
            # Phase difference between consecutive frames
            diffs = np.diff(x, axis=0)
            return np.mean(diffs, axis=0)
        return np.zeros(x.shape[-2:] if x.ndim >= 2 else x.shape, dtype=np.float64)


class ElasticWaveModel(BasePrimitive):
    """Shear wave propagation for elastography.

    Scales input by wave_speed factor (simplified real-valued model).
    """

    primitive_id = "elastic_wave_model"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        wave_speed = self._params.get("wave_speed", 1.5)
        frequency = self._params.get("frequency", 100.0)
        x = np.asarray(x, dtype=np.float64)
        return x * wave_speed

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        wave_speed = self._params.get("wave_speed", 1.5)
        return np.asarray(y, dtype=np.float64) * wave_speed


class DualEnergyBeerLambert(BasePrimitive):
    """Dual-energy Beer-Lambert attenuation for DEXA.

    Forward: stacks [I_0_low * exp(-mu_low * x), I_0_high * exp(-mu_high * x)]
    along new first axis. Non-linear.
    """

    primitive_id = "dual_energy_beer_lambert"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        I_0_low = self._params.get("I_0_low", 1000.0)
        I_0_high = self._params.get("I_0_high", 1000.0)
        mu_low = self._params.get("mu_low", 0.2)
        mu_high = self._params.get("mu_high", 0.1)
        x = np.asarray(x, dtype=np.float64)
        low = I_0_low * np.exp(-mu_low * x)
        high = I_0_high * np.exp(-mu_high * x)
        return np.stack([low, high], axis=0)


class DepthOptics(BasePrimitive):
    """Thin lens + distortion model for depth-dependent blur.

    Simplified: Gaussian blur with sigma proportional to aperture.
    Non-linear due to depth-dependent CoC.
    """

    primitive_id = "depth_optics"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        focal_length = self._params.get("focal_length", 50.0)
        aperture = self._params.get("aperture", 2.8)
        x = np.asarray(x, dtype=np.float64)
        sigma = aperture / max(focal_length, 1e-12)
        return ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")


class DiffractionCamera(BasePrimitive):
    """Far-field diffraction: |FFT(x)|^2 + optional detector PSF.

    Non-linear due to squared magnitude.
    """

    primitive_id = "diffraction_camera"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        detector_psf_sigma = self._params.get("detector_psf_sigma", 0.0)
        x = np.asarray(x, dtype=np.complex128)
        y = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
        if detector_psf_sigma > 0:
            y = ndimage.gaussian_filter(y.astype(np.float64), sigma=detector_psf_sigma)
        return y.astype(np.float64)


class SARBackprojection(BasePrimitive):
    """SAR time-domain backprojection (Radon-like projection).

    Forward: Radon-like projection. Adjoint: back-projection.
    """

    primitive_id = "sar_backprojection"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_pulses = self._params.get("n_pulses", 64)
        swath_width = self._params.get("swath_width", 100.0)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim < 2:
            return x.copy()
        H, W = x.shape[-2], x.shape[-1]
        angles = np.linspace(0, 180, n_pulses, endpoint=False)
        sino = np.zeros((n_pulses, W), dtype=np.float64)
        for i, angle in enumerate(angles):
            rotated = ndimage.rotate(x, angle, reshape=False, mode="constant", order=1)
            sino[i, :] = rotated.sum(axis=0)
        return sino

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        n_pulses = self._params.get("n_pulses", 64)
        x_shape = self._params.get("x_shape", (64, 64))
        H, W = x_shape[-2], x_shape[-1]
        angles = np.linspace(0, 180, n_pulses, endpoint=False)
        y2d = np.asarray(y, dtype=np.float64).reshape(n_pulses, -1)
        bp = np.zeros((H, W), dtype=np.float64)
        for i, angle in enumerate(angles):
            proj = y2d[i, :W]
            smeared = np.tile(proj, (H, 1))
            bp += ndimage.rotate(smeared, -angle, reshape=False, mode="constant", order=1)
        return bp


class ParticleAttenuation(BasePrimitive):
    """Beer-Lambert attenuation for particles.

    Forward: I_0 * exp(-cross_section * x). Non-linear.
    """

    primitive_id = "particle_attenuation"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        I_0 = self._params.get("I_0", 1000.0)
        cross_section = self._params.get("cross_section", 0.1)
        x = np.asarray(x, dtype=np.float64)
        return I_0 * np.exp(-cross_section * x)


class MultipleScatteringKernel(BasePrimitive):
    """Multiple scattering blur kernel.

    Forward: Gaussian blur with given sigma. Self-adjoint.
    """

    primitive_id = "multiple_scattering"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 2.0)
        return ndimage.gaussian_filter(
            np.asarray(x, dtype=np.float64), sigma=sigma, mode="reflect"
        )

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        sigma = self._params.get("sigma", 2.0)
        return ndimage.gaussian_filter(
            np.asarray(y, dtype=np.float64), sigma=sigma, mode="reflect"
        )


class VesselFlowContrast(BasePrimitive):
    """OCTA-style vessel flow decorrelation contrast.

    If 3D (T,H,W), computes variance along T axis. If 2D, passthrough.
    Non-linear.
    """

    primitive_id = "vessel_flow_contrast"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_frames = self._params.get("n_frames", 4)
        threshold = self._params.get("threshold", 0.1)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            return np.var(x, axis=0)
        return x.copy()


class SpecularReflectionModel(BasePrimitive):
    """Endoscopy specular highlights (simplified Phong model).

    Forward: x + specular_strength * |x|^(1/roughness). Non-linear.
    """

    primitive_id = "specular_reflection"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        specular_strength = self._params.get("specular_strength", 0.1)
        roughness = self._params.get("roughness", 0.5)
        x = np.asarray(x, dtype=np.float64)
        exponent = 1.0 / max(roughness, 1e-12)
        return x + specular_strength * np.power(np.abs(x), exponent)


class StructuredLightProjector(BasePrimitive):
    """Pattern projection for structured-light depth sensing.

    Multiplies input by sinusoidal fringe pattern.
    """

    primitive_id = "structured_light_projector"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_patterns = self._params.get("n_patterns", 4)
        fringe_freq = self._params.get("fringe_freq", 8.0)
        x = np.asarray(x, dtype=np.float64)
        H, W = x.shape[-2], x.shape[-1]
        xx = np.arange(W, dtype=np.float64)
        pattern = 0.5 + 0.5 * np.cos(2.0 * np.pi * fringe_freq * xx / W)
        if x.ndim == 2:
            return x * pattern[np.newaxis, :]
        return x * pattern.reshape((1,) * (x.ndim - 1) + (W,))

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        fringe_freq = self._params.get("fringe_freq", 8.0)
        y = np.asarray(y, dtype=np.float64)
        W = y.shape[-1]
        xx = np.arange(W, dtype=np.float64)
        pattern = 0.5 + 0.5 * np.cos(2.0 * np.pi * fringe_freq * xx / W)
        if y.ndim == 2:
            return y * pattern[np.newaxis, :]
        return y * pattern.reshape((1,) * (y.ndim - 1) + (W,))


class SequenceBlock(BasePrimitive):
    """MRI sequence parameterization (TE/TR weighting).

    Simplified: scales by exp(-TE/T2star) where T2star is derived from TR.
    """

    primitive_id = "sequence_block"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sequence_type = self._params.get("sequence_type", "epi")
        TE_ms = self._params.get("TE_ms", 30.0)
        TR_ms = self._params.get("TR_ms", 2000.0)
        x = np.asarray(x, dtype=np.float64)
        # Simplified T2* weighting: T2star ~ TR / 50
        T2star = max(TR_ms / 50.0, 1e-6)
        weight = np.exp(-TE_ms / T2star)
        return x * weight

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        TE_ms = self._params.get("TE_ms", 30.0)
        TR_ms = self._params.get("TR_ms", 2000.0)
        T2star = max(TR_ms / 50.0, 1e-6)
        weight = np.exp(-TE_ms / T2star)
        return np.asarray(y, dtype=np.float64) * weight


class PhysiologyDrift(BasePrimitive):
    """Low-rank temporal drift for fMRI / long scans.

    If 3D (T,H,W), adds sinusoidal drift along T axis. If 2D, passthrough.
    """

    primitive_id = "physiology_drift"
    _is_linear = True
    _node_role = "element"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        drift_amplitude = self._params.get("drift_amplitude", 0.01)
        n_components = self._params.get("n_components", 3)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            T = x.shape[0]
            t = np.arange(T, dtype=np.float64)
            drift = np.zeros(T, dtype=np.float64)
            for k in range(1, n_components + 1):
                drift += np.sin(2.0 * np.pi * k * t / max(T, 1)) / k
            drift = drift_amplitude * drift / max(np.max(np.abs(drift)), 1e-12)
            return x + drift[:, np.newaxis, np.newaxis]
        return x.copy()

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        # Drift is additive constant; adjoint is identity
        return np.asarray(y, dtype=np.float64).copy()


class ReciprocalSpaceGeometry(BasePrimitive):
    """EBSD/diffraction reciprocal-space geometric distortion.

    Applies tilt-dependent scaling via affine_transform. Non-linear.
    """

    primitive_id = "reciprocal_space_geometry"
    _is_linear = False
    _node_role = "element"
    _physics_tier = "tier1_wave"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        sample_tilt_deg = self._params.get("sample_tilt_deg", 70.0)
        detector_distance = self._params.get("detector_distance", 15.0)
        x = np.asarray(x, dtype=np.float64)
        tilt_rad = np.deg2rad(sample_tilt_deg)
        scale_y = np.cos(tilt_rad)
        # Apply affine scaling (stretch along y)
        if x.ndim >= 2:
            matrix = np.array([[1.0 / max(scale_y, 0.01), 0], [0, 1.0]])
            offset = np.array([x.shape[-2] * (1 - 1.0 / max(scale_y, 0.01)) / 2.0, 0])
            return ndimage.affine_transform(
                x, matrix, offset=offset, order=1, mode="constant"
            )
        return x.copy()


# =========================================================================
# Phase v4+: Extended modality primitives — Sensor family
# =========================================================================


class SPADToFSensor(BasePrimitive):
    """SPAD time-of-flight histogram sensor.

    Forward: x * qe (simplified time-binned detection).
    """

    primitive_id = "spad_tof_sensor"
    _is_linear = True
    _node_role = "sensor"
    _carrier_type = "photon"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_bins = self._params.get("n_bins", 64)
        dead_time_ns = self._params.get("dead_time_ns", 50.0)
        qe = self._params.get("qe", 0.3)
        return np.asarray(x, dtype=np.float64) * qe

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        qe = self._params.get("qe", 0.3)
        return np.asarray(y, dtype=np.float64) * qe


class EnergyResolvingDetector(BasePrimitive):
    """Energy-resolving detector for EELS.

    Forward: x * qe.
    """

    primitive_id = "energy_resolving_detector"
    _is_linear = True
    _node_role = "sensor"
    _carrier_type = "electron"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_channels = self._params.get("n_channels", 256)
        energy_range_ev = self._params.get("energy_range_ev", [0, 2000])
        qe = self._params.get("qe", 0.8)
        return np.asarray(x, dtype=np.float64) * qe

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        qe = self._params.get("qe", 0.8)
        return np.asarray(y, dtype=np.float64) * qe


class FiberBundleSensor(BasePrimitive):
    """Endoscope fiber bundle sensor.

    Subsamples input at random core positions and multiplies by
    coupling_efficiency. Non-linear due to random subsampling.
    """

    primitive_id = "fiber_bundle_sensor"
    _is_linear = False
    _node_role = "sensor"
    _carrier_type = "photon"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_cores = self._params.get("n_cores", 10000)
        core_pitch = self._params.get("core_pitch", 10.0)
        coupling_efficiency = self._params.get("coupling_efficiency", 0.3)
        seed = self._params.get("seed", 0)
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(seed)
        H, W = x.shape[-2], x.shape[-1]
        n_sample = min(n_cores, H * W)
        indices = rng.choice(H * W, size=n_sample, replace=False)
        flat = x.ravel()[:H * W]
        sampled = flat[indices] * coupling_efficiency
        # Return as 1D sampled vector
        return sampled


class TrackDetectorSensor(BasePrimitive):
    """Muon/particle track detector.

    Forward: x * efficiency.
    """

    primitive_id = "track_detector_sensor"
    _is_linear = True
    _node_role = "sensor"
    _carrier_type = "abstract"
    _physics_tier = "tier0_geometry"

    def forward(self, x: np.ndarray, **params: Any) -> np.ndarray:
        n_layers = self._params.get("n_layers", 8)
        efficiency = self._params.get("efficiency", 0.95)
        return np.asarray(x, dtype=np.float64) * efficiency

    def adjoint(self, y: np.ndarray, **params: Any) -> np.ndarray:
        efficiency = self._params.get("efficiency", 0.95)
        return np.asarray(y, dtype=np.float64) * efficiency


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
    # Phase 1 v4: Extended node families
    ExplicitLinearOperator,
    ElectronBeamSource,
    ElectronDetectorSensor,
    SinglePixelSensor,
    XRayDetectorSensor,
    AcousticReceiveSensor,
    CorrelatedNoiseSensor,
    ThinObjectPhase,
    CTFTransfer,
    YieldModel,
    BeamformDelay,
    EmissionProjection,
    ScatterModel,
    # Phase v4+: Extended modality primitives
    ScanTrajectory,
    TimeOfFlightGate,
    CollimatorModel,
    FluoroTemporalIntegrator,
    FluorescenceKinetics,
    NonlinearExcitation,
    SaturationDepletion,
    BlinkingEmitterModel,
    EvanescentFieldDecay,
    DopplerEstimator,
    ElasticWaveModel,
    DualEnergyBeerLambert,
    DepthOptics,
    DiffractionCamera,
    SARBackprojection,
    ParticleAttenuation,
    MultipleScatteringKernel,
    VesselFlowContrast,
    SpecularReflectionModel,
    StructuredLightProjector,
    SequenceBlock,
    PhysiologyDrift,
    ReciprocalSpaceGeometry,
    # Phase v4+: Extended sensors
    SPADToFSensor,
    EnergyResolvingDetector,
    FiberBundleSensor,
    TrackDetectorSensor,
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
