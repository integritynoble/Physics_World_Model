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

from pwm_core.mismatch.subpixel import subpixel_shift_2d
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
def _band_shift_fwd(x, dispersion=None, sign=1.0):
    disp = dispersion or {}
    L = x.shape[-1]
    out = np.zeros_like(x)
    for l in range(L):
        dx, dy = dispersion_shift(disp, band=l)
        out[..., l] = subpixel_shift_2d(x[..., l], sign * dx, sign * dy)
    return out


def _band_shift_adj(y, dispersion=None, sign=1.0):
    return _band_shift_fwd(y, dispersion=dispersion, sign=-sign)


register_primitive(Primitive(
    name="band_shift",
    forward=_band_shift_fwd,
    adjoint=_band_shift_adj,
    out_shape=lambda in_shape, dispersion=None, sign=1.0: tuple(in_shape),
    is_linear=True,
))
