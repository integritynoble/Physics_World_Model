"""Differentiable primitive operators for the forward-model compiler.

Each primitive composes into a CompiledOperator. Linear primitives provide an
exact adjoint (so the composed operator gets BaseOperator.check_adjoint and
torch autograd for free). Nonlinear / stochastic primitives set is_linear=False
and adjoint=None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.spectral.dispersion_models import dispersion_shift


@dataclass
class Primitive:
    name: str
    forward: Callable[..., np.ndarray]
    out_shape: Callable[..., Tuple[int, ...]]
    adjoint: Optional[Callable[..., np.ndarray]] = None
    is_linear: bool = True


PRIMITIVES: Dict[str, Primitive] = {}


def register_primitive(prim: Primitive) -> Primitive:
    PRIMITIVES[prim.name] = prim
    return prim


def get_primitive(name: str) -> Primitive:
    if name not in PRIMITIVES:
        raise KeyError(f"unknown primitive {name!r}; known: {sorted(PRIMITIVES)}")
    return PRIMITIVES[name]


# --- scale: y = c * x -------------------------------------------------------
register_primitive(Primitive(
    name="scale",
    forward=lambda x, c=1.0: (x * float(c)),
    adjoint=lambda y, c=1.0: (y * float(c)),
    out_shape=lambda in_shape, c=1.0: tuple(in_shape),
    is_linear=True,
))


# --- mask_multiply: y = x * mask (mask broadcast over trailing band axis) ----
def _mask_fwd(x, mask=None):
    m = np.asarray(mask)
    if x.ndim == m.ndim + 1:        # (H,W,L) * (H,W) -> broadcast over bands
        m = m[..., None]
    return x * m


def _mask_shape(in_shape, mask=None):
    return tuple(in_shape)


register_primitive(Primitive(
    name="mask_multiply",
    forward=_mask_fwd,
    adjoint=_mask_fwd,              # multiplication by a real mask is self-adjoint
    out_shape=_mask_shape,
    is_linear=True,
))


# --- band_sum: (H,W,L) -> (H,W) ---------------------------------------------
def _band_sum_fwd(x):
    return np.sum(x, axis=-1)


def _band_sum_shape(in_shape):
    if len(in_shape) < 3:
        raise ValueError(f"band_sum expects (...,L) with ndim>=3, got {in_shape}")
    return tuple(in_shape[:-1])


def _band_sum_adjoint(y, n_bands=None):
    if n_bands is None:
        raise ValueError("band_sum adjoint requires n_bands param")
    return np.repeat(y[..., None], int(n_bands), axis=-1)


register_primitive(Primitive(
    name="band_sum",
    forward=lambda x, n_bands=None: _band_sum_fwd(x),
    adjoint=_band_sum_adjoint,
    out_shape=lambda in_shape, n_bands=None: _band_sum_shape(in_shape),
    is_linear=True,
))


# --- band_shift: shift each spectral band by dispersion (H,W,L)->(H,W,L) -----
# Implemented as an EXACT linear bilinear shift so the adjoint is exact at any
# (including sub-pixel) shift. A zero-fill integer shift S_k has transpose
# S_{-k}; the fractional blend (1-a)*S_f + a*S_{f+1} transposes term-by-term.

def _roll_zero_1d(arr, k, axis):
    """Shift `arr` along `axis` by integer k (k>0 => toward higher index),
    filling vacated entries with zero. Its transpose is the same op with -k."""
    if k == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    if abs(k) >= n:
        return out
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    if k > 0:
        dst[axis] = slice(k, None)
        src[axis] = slice(0, n - k)
    else:
        dst[axis] = slice(0, n + k)
        src[axis] = slice(-k, None)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _shift_axis(img, s, axis):
    f = int(np.floor(s))
    a = float(s - f)
    return (1.0 - a) * _roll_zero_1d(img, f, axis) + a * _roll_zero_1d(img, f + 1, axis)


def _shift_axis_adj(img, s, axis):
    f = int(np.floor(s))
    a = float(s - f)
    return (1.0 - a) * _roll_zero_1d(img, -f, axis) + a * _roll_zero_1d(img, -(f + 1), axis)


def _band_shift_fwd(x, dispersion=None):
    disp = dispersion if dispersion is not None else {}
    xf = np.asarray(x, dtype=np.float64)
    L = xf.shape[-1]
    out = np.zeros_like(xf)
    for l in range(L):
        dx, dy = dispersion_shift(disp, band=l)
        # dx shifts columns (axis=1, horizontal); dy shifts rows (axis=0, vertical)
        out[..., l] = _shift_axis(_shift_axis(xf[..., l], dx, axis=1), dy, axis=0)
    return out


def _band_shift_adj(y, dispersion=None):
    disp = dispersion if dispersion is not None else {}
    yf = np.asarray(y, dtype=np.float64)
    L = yf.shape[-1]
    out = np.zeros_like(yf)
    for l in range(L):
        dx, dy = dispersion_shift(disp, band=l)
        # (Sy Sx)^T = Sx^T Sy^T  -> apply Sy^T (axis 0) then Sx^T (axis 1)
        out[..., l] = _shift_axis_adj(_shift_axis_adj(yf[..., l], dy, axis=0), dx, axis=1)
    return out


register_primitive(Primitive(
    name="band_shift",
    forward=_band_shift_fwd,
    adjoint=_band_shift_adj,
    out_shape=lambda in_shape, dispersion=None: tuple(in_shape),
    is_linear=True,
))


# --- square_magnitude: y = x**2 (intensity / phase-retrieval forward) --------
register_primitive(Primitive(
    name="square_magnitude",
    forward=lambda x: np.abs(x) ** 2,
    adjoint=None,
    out_shape=lambda in_shape: tuple(in_shape),
    is_linear=False,
))


# --- gaussian_noise: y = x + N(0, sigma^2), seeded for reproducibility -------
def _gaussian_noise_fwd(x, sigma=0.0, seed=0):
    rng = np.random.default_rng(int(seed))
    return x + rng.standard_normal(x.shape) * float(sigma)


register_primitive(Primitive(
    name="gaussian_noise",
    forward=_gaussian_noise_fwd,
    adjoint=None,
    out_shape=lambda in_shape, sigma=0.0, seed=0: tuple(in_shape),
    is_linear=False,
))
