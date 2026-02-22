"""GPU utility functions for iterative solvers.

Provides device resolution, NumPy/Torch conversion, and GPU-accelerated
primitives (soft thresholding, TV proximal operators) used across all
classical solver files.

All public functions handle the case where PyTorch is unavailable.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np


def resolve_device(device=None):
    """Resolve compute device for GPU-accelerated solvers.

    Args:
        device: None (auto-detect), 'cpu', 'cuda', 'cuda:0', etc.
            When None, uses CUDA if available, else returns (None, False).

    Returns:
        (torch_device_or_None, use_gpu: bool)
    """
    try:
        import torch
    except ImportError:
        return None, False

    if device is not None:
        dev = torch.device(device)
        return dev, (dev.type == 'cuda')

    if torch.cuda.is_available():
        return torch.device('cuda'), True

    return torch.device('cpu'), False


def to_torch(x, device, dtype=None):
    """Convert NumPy array to torch tensor on *device*.

    Args:
        x: numpy ndarray
        device: torch.device
        dtype: optional torch dtype (default: inferred from x)

    Returns:
        torch.Tensor on *device*
    """
    import torch

    if dtype is None:
        # Map numpy dtypes to torch dtypes
        _map = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
        }
        dtype = _map.get(x.dtype.type, torch.float32)

    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


def to_numpy(t):
    """Convert torch tensor to NumPy array (always CPU, contiguous)."""
    return t.detach().cpu().numpy()


def wrap_operator(fn, device):
    """Wrap a numpy-in/numpy-out callable to accept/return torch tensors.

    The returned callable: torch tensor in -> numpy -> fn -> numpy -> torch tensor out.
    This is the simplest bridge for forward/adjoint operators that have no
    native torch implementation.

    Args:
        fn: callable that takes and returns numpy arrays
        device: torch.device for input/output tensors

    Returns:
        Wrapped callable operating on torch tensors.
    """
    def wrapped(t):
        import torch
        x_np = to_numpy(t)
        y_np = fn(x_np)
        return to_torch(y_np, device)
    return wrapped


# -------------------------------------------------------------------------
# GPU primitives
# -------------------------------------------------------------------------

def soft_threshold_torch(x, tau):
    """GPU soft thresholding: sign(x) * max(|x| - tau, 0).

    Args:
        x: torch tensor
        tau: threshold (float or tensor)

    Returns:
        Soft-thresholded tensor (same device)
    """
    import torch
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)


def soft_threshold_complex_torch(x, tau):
    """GPU soft thresholding for complex tensors.

    Args:
        x: complex torch tensor
        tau: threshold (float or tensor)

    Returns:
        Soft-thresholded complex tensor (same device)
    """
    import torch
    mag = torch.abs(x)
    return x * torch.clamp(mag - tau, min=0) / (mag + 1e-10)


def tv_prox_2d_torch(x, lam, iterations=20):
    """GPU Chambolle TV proximal operator for 2D images.

    Dual (Chambolle 2004) algorithm on GPU via PyTorch.

    Args:
        x: 2D torch tensor (H, W)
        lam: regularization strength
        iterations: number of dual iterations

    Returns:
        TV-denoised 2D tensor (same device)
    """
    import torch

    h, w = x.shape
    p = torch.zeros(h, w, 2, device=x.device, dtype=x.dtype)
    tau = 0.25

    for _ in range(iterations):
        # Divergence of p
        div_p = torch.zeros_like(x)
        div_p[:, :-1] += p[:, :-1, 0]
        div_p[:, 1:] -= p[:, :-1, 0]
        div_p[:-1, :] += p[:-1, :, 1]
        div_p[1:, :] -= p[:-1, :, 1]

        # Gradient of (x - lam * div_p)
        u = x - lam * div_p
        grad_u = torch.zeros(h, w, 2, device=x.device, dtype=x.dtype)
        grad_u[:, :-1, 0] = u[:, 1:] - u[:, :-1]
        grad_u[:-1, :, 1] = u[1:, :] - u[:-1, :]

        # Update dual
        p_new = p + tau * grad_u

        # Project onto unit ball
        norm = torch.sqrt(p_new[:, :, 0] ** 2 + p_new[:, :, 1] ** 2 + 1e-10)
        norm = torch.clamp(norm, min=1)
        p = p_new / norm.unsqueeze(-1)

    # Final result
    div_p = torch.zeros_like(x)
    div_p[:, :-1] += p[:, :-1, 0]
    div_p[:, 1:] -= p[:, :-1, 0]
    div_p[:-1, :] += p[:-1, :, 1]
    div_p[1:, :] -= p[:-1, :, 1]

    return x - lam * div_p


def tv_denoiser_3d_torch(x, lam, iterations=10, axis_weights=(1.0, 1.0, 0.5)):
    """GPU 3D anisotropic TV denoising (Chambolle dual).

    Args:
        x: 3D torch tensor (H, W, C)
        lam: regularization strength
        iterations: number of dual iterations
        axis_weights: (wy, wx, wc) relative weights

    Returns:
        TV-denoised 3D tensor (same device)
    """
    import torch

    h, w, c = x.shape
    p = torch.zeros(h, w, c, 3, device=x.device, dtype=x.dtype)
    tau = 0.125
    wy, wx, wc = axis_weights

    for _ in range(iterations):
        # Divergence
        div = torch.zeros_like(x)

        # y-component
        div[:-1, :, :] += p[:-1, :, :, 0]
        div[1:, :, :] -= p[:-1, :, :, 0]
        div *= wy

        # x-component
        divx = torch.zeros_like(x)
        divx[:, :-1, :] += p[:, :-1, :, 1]
        divx[:, 1:, :] -= p[:, :-1, :, 1]
        div += wx * divx

        # c-component
        divc = torch.zeros_like(x)
        divc[:, :, :-1] += p[:, :, :-1, 2]
        divc[:, :, 1:] -= p[:, :, :-1, 2]
        div += wc * divc

        # Gradient of (x - lam * div)
        u = x - lam * div
        grad = torch.zeros(h, w, c, 3, device=x.device, dtype=x.dtype)
        grad[:-1, :, :, 0] = wy * (u[1:, :, :] - u[:-1, :, :])
        grad[:, :-1, :, 1] = wx * (u[:, 1:, :] - u[:, :-1, :])
        grad[:, :, :-1, 2] = wc * (u[:, :, 1:] - u[:, :, :-1])

        # Update dual
        p_new = p + tau * grad

        # Project to unit ball
        norm = torch.sqrt(torch.sum(p_new ** 2, dim=3, keepdim=True) + 1e-10)
        p = p_new / torch.clamp(norm, min=1)

    # Final result
    div = torch.zeros_like(x)
    div[:-1, :, :] += p[:-1, :, :, 0]
    div[1:, :, :] -= p[:-1, :, :, 0]
    div *= wy

    divx = torch.zeros_like(x)
    divx[:, :-1, :] += p[:, :-1, :, 1]
    divx[:, 1:, :] -= p[:, :-1, :, 1]
    div += wx * divx

    divc = torch.zeros_like(x)
    divc[:, :, :-1] += p[:, :, :-1, 2]
    divc[:, :, 1:] -= p[:, :, :-1, 2]
    div += wc * divc

    return x - lam * div
