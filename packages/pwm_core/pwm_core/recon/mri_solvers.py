"""MRI Reconstruction Solvers: ESPIRiT and Compressed Sensing.

Standard algorithms for accelerated MRI reconstruction.

References:
- Uecker, M. et al. (2014). "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI"
- Lustig, M. et al. (2007). "Sparse MRI: The application of compressed sensing"

Benchmark: fastMRI knee dataset (4x, 8x acceleration)
Expected PSNR:
- Zero-filled: 28.5 dB
- L1-ESPIRiT: 34.2 dB (4x)
- VarNet: 38.1 dB (4x) [future DL method]
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def estimate_sensitivity_maps(
    kspace: np.ndarray,
    acs_size: int = 24,
    n_maps: int = 1,
    device=None,
) -> np.ndarray:
    """Estimate coil sensitivity maps using ESPIRiT.

    Simplified ESPIRiT implementation.

    Args:
        kspace: Multi-coil k-space (n_coils, H, W)
        acs_size: Size of auto-calibration region
        n_maps: Number of sensitivity maps (typically 1)
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Sensitivity maps (n_coils, H, W) complex
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _estimate_sensitivity_maps_torch(kspace, acs_size, dev)

    n_coils, h, w = kspace.shape

    # Extract ACS (Auto-Calibration Signal) from center
    center_y, center_x = h // 2, w // 2
    acs_y = slice(center_y - acs_size // 2, center_y + acs_size // 2)
    acs_x = slice(center_x - acs_size // 2, center_x + acs_size // 2)

    acs = kspace[:, acs_y, acs_x]

    # Low-resolution images from ACS
    low_res = np.zeros((n_coils, h, w), dtype=np.complex64)
    low_res[:, acs_y, acs_x] = acs

    # Transform to image domain
    coil_images = np.zeros((n_coils, h, w), dtype=np.complex64)
    for c in range(n_coils):
        coil_images[c] = ifft2(ifftshift(low_res[c]))

    # Compute sensitivity maps using SoS normalization
    sos = np.sqrt(np.sum(np.abs(coil_images)**2, axis=0) + 1e-10)

    sensitivity_maps = coil_images / sos[np.newaxis, :, :]

    return sensitivity_maps.astype(np.complex64)


def _estimate_sensitivity_maps_torch(kspace, acs_size, dev):
    """GPU implementation of estimate_sensitivity_maps."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy

    kspace_t = to_torch(kspace, dev, torch.complex64)
    n_coils, h, w = kspace_t.shape

    center_y, center_x = h // 2, w // 2
    acs_y = slice(center_y - acs_size // 2, center_y + acs_size // 2)
    acs_x = slice(center_x - acs_size // 2, center_x + acs_size // 2)

    low_res = torch.zeros(n_coils, h, w, device=dev, dtype=torch.complex64)
    low_res[:, acs_y, acs_x] = kspace_t[:, acs_y, acs_x]

    coil_images = torch.fft.ifft2(torch.fft.ifftshift(low_res, dim=(-2, -1)))
    sos = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0) + 1e-10)
    sensitivity_maps = coil_images / sos.unsqueeze(0)

    return to_numpy(sensitivity_maps).astype(np.complex64)


def espirit_maps(
    kspace: np.ndarray,
    kernel_size: int = 6,
    acs_size: int = 24,
    threshold: float = 0.02,
) -> np.ndarray:
    """Full ESPIRiT sensitivity map estimation.

    Args:
        kspace: Multi-coil k-space (n_coils, H, W)
        kernel_size: GRAPPA kernel size
        acs_size: ACS region size
        threshold: Eigenvalue threshold

    Returns:
        Sensitivity maps (n_coils, H, W)
    """
    n_coils, h, w = kspace.shape

    # For simplicity, use the basic method
    # Full ESPIRiT would construct calibration matrix and do SVD
    return estimate_sensitivity_maps(kspace, acs_size)


def sense_reconstruction(
    kspace: np.ndarray,
    sensitivity_maps: np.ndarray,
    mask: np.ndarray,
    regularization: float = 0.001,
    iterations: int = 30,
    device=None,
) -> np.ndarray:
    """SENSE reconstruction for parallel MRI.

    Args:
        kspace: Under-sampled multi-coil k-space (n_coils, H, W)
        sensitivity_maps: Coil sensitivities (n_coils, H, W)
        mask: Sampling mask (H, W)
        regularization: Regularization parameter
        iterations: CG iterations
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed image (H, W)
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _sense_reconstruction_torch(kspace, sensitivity_maps, mask,
                                           regularization, iterations, dev)

    n_coils, h, w = kspace.shape
    kspace = kspace.astype(np.complex64)
    sens = sensitivity_maps.astype(np.complex64)
    mask = mask.astype(np.float32)

    # Adjoint operation: sum over coils of S^H * F^H * y
    def adjoint(y):
        result = np.zeros((h, w), dtype=np.complex64)
        for c in range(n_coils):
            img = ifft2(ifftshift(y[c]))
            result += np.conj(sens[c]) * img
        return result

    # Forward operation: F * S * x for each coil
    def forward(x):
        result = np.zeros((n_coils, h, w), dtype=np.complex64)
        for c in range(n_coils):
            coil_img = sens[c] * x
            result[c] = fftshift(fft2(coil_img)) * mask
        return result

    # Normal equations: (A^H A + lambda I) x = A^H y
    # where A = M * F * S

    # Initialize with adjoint
    x = adjoint(kspace)

    # Right-hand side
    b = adjoint(kspace)

    # CG solver
    r = b - adjoint(forward(x)) - regularization * x
    p = r.copy()
    rsold = np.sum(np.abs(r)**2)

    for i in range(iterations):
        Ap = adjoint(forward(p)) + regularization * p
        pAp = np.sum(np.conj(p) * Ap)

        if np.abs(pAp) < 1e-12:
            break

        alpha = rsold / (pAp + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.sum(np.abs(r)**2)
        if rsnew < 1e-10:
            break

        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return x.astype(np.complex64)


def _sense_reconstruction_torch(kspace, sensitivity_maps, mask,
                                regularization, iterations, dev):
    """GPU implementation of sense_reconstruction."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy

    kspace_t = to_torch(kspace, dev, torch.complex64)
    sens_t = to_torch(sensitivity_maps, dev, torch.complex64)
    mask_t = to_torch(mask, dev, torch.float32)
    n_coils, h, w = kspace_t.shape

    def adjoint_t(y):
        imgs = torch.fft.ifft2(torch.fft.ifftshift(y, dim=(-2, -1)))
        return torch.sum(torch.conj(sens_t) * imgs, dim=0)

    def forward_t(x):
        coil_imgs = sens_t * x.unsqueeze(0)
        return torch.fft.fftshift(torch.fft.fft2(coil_imgs), dim=(-2, -1)) * mask_t

    x = adjoint_t(kspace_t)
    b = x.clone()

    r = b - adjoint_t(forward_t(x)) - regularization * x
    p = r.clone()
    rsold = torch.sum(torch.abs(r) ** 2)

    for i in range(iterations):
        Ap = adjoint_t(forward_t(p)) + regularization * p
        pAp = torch.sum(torch.conj(p) * Ap)

        if torch.abs(pAp) < 1e-12:
            break

        alpha = rsold / (pAp + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = torch.sum(torch.abs(r) ** 2)
        if rsnew < 1e-10:
            break

        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return to_numpy(x).astype(np.complex64)


def cs_mri_wavelet(
    kspace: np.ndarray,
    mask: np.ndarray,
    lam: float = 0.01,
    iterations: int = 50,
    sensitivity_maps: Optional[np.ndarray] = None,
    device=None,
) -> np.ndarray:
    """Compressed Sensing MRI with wavelet sparsity.

    Solves: min_x ||MFx - y||^2 + lam * ||Wx||_1

    Args:
        kspace: (Undersampled) k-space data
        mask: Sampling mask
        lam: Sparsity weight
        iterations: FISTA iterations
        sensitivity_maps: For multi-coil (optional)
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed image
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    # Handle multi-coil
    if kspace.ndim == 3:
        if sensitivity_maps is None:
            sensitivity_maps = estimate_sensitivity_maps(kspace, device=device)
        return sense_reconstruction(kspace, sensitivity_maps, mask, device=device)

    if use_gpu:
        return _cs_mri_wavelet_torch(kspace, mask, lam, iterations, dev)

    try:
        import pywt
        has_wavelet = True
    except ImportError:
        has_wavelet = False

    h, w = kspace.shape
    kspace = kspace.astype(np.complex64)
    mask = mask.astype(np.float32)

    # Forward: masked FFT
    def forward(x):
        return fftshift(fft2(x)) * mask

    # Adjoint: masked inverse FFT
    def adjoint(y):
        return ifft2(ifftshift(y * mask))

    # Soft thresholding for complex data
    def soft_thresh_complex(x, t):
        mag = np.abs(x)
        return x * np.maximum(mag - t, 0) / (mag + 1e-10)

    # Initialize
    x = adjoint(kspace)
    z = x.copy()
    t = 1.0
    step = 0.5

    for i in range(iterations):
        # Gradient step
        residual = forward(z) - kspace
        grad = adjoint(residual)
        v = z - step * grad

        # Soft thresholding (simplified - on image directly)
        x_new = soft_thresh_complex(v, lam * step)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return x.astype(np.complex64)


def _cs_mri_wavelet_torch(kspace, mask, lam, iterations, dev):
    """GPU implementation of cs_mri_wavelet (FFT/gradient parts on GPU)."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy, soft_threshold_complex_torch

    kspace_t = to_torch(kspace, dev, torch.complex64)
    mask_t = to_torch(mask, dev, torch.float32)

    def forward_t(x):
        return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)) * mask_t

    def adjoint_t(y):
        return torch.fft.ifft2(torch.fft.ifftshift(y * mask_t, dim=(-2, -1)))

    x = adjoint_t(kspace_t)
    z = x.clone()
    t = 1.0
    step = 0.5

    for i in range(iterations):
        residual = forward_t(z) - kspace_t
        grad = adjoint_t(residual)
        v = z - step * grad

        x_new = soft_threshold_complex_torch(v, lam * step)

        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return to_numpy(x).astype(np.complex64)


def zero_filled_reconstruction(
    kspace: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device=None,
) -> np.ndarray:
    """Simple zero-filled reconstruction (baseline).

    Args:
        kspace: K-space data (single or multi-coil)
        mask: Sampling mask (optional)
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed image
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _zero_filled_reconstruction_torch(kspace, dev)

    if kspace.ndim == 3:
        # Multi-coil: root sum of squares
        n_coils = kspace.shape[0]
        imgs = np.array([ifft2(ifftshift(kspace[c])) for c in range(n_coils)])
        return np.sqrt(np.sum(np.abs(imgs)**2, axis=0)).astype(np.float32)
    else:
        return np.abs(ifft2(ifftshift(kspace))).astype(np.float32)


def _zero_filled_reconstruction_torch(kspace, dev):
    """GPU implementation of zero_filled_reconstruction."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy

    kspace_t = to_torch(kspace, dev, torch.complex64)

    if kspace_t.ndim == 3:
        imgs = torch.fft.ifft2(torch.fft.ifftshift(kspace_t, dim=(-2, -1)))
        result = torch.sqrt(torch.sum(torch.abs(imgs) ** 2, dim=0))
        return to_numpy(result).astype(np.float32)
    else:
        result = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_t, dim=(-2, -1))))
        return to_numpy(result).astype(np.float32)


def run_sense(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run SENSE MRI reconstruction (harness-compatible wrapper).

    Args:
        y: K-space measurements (n_coils, H, W) or (H, W).
        physics: MRI physics operator (must expose mask and optionally
            sensitivity_maps via attributes or info()).
        cfg: Configuration with:
            - iters: CG iterations (default: 30)
            - regularization: Tikhonov parameter (default: 0.001)

    Returns:
        Tuple of (reconstructed image, info_dict)
    """
    iters = cfg.get("iters", 30)
    regularization = cfg.get("regularization", 0.001)
    device = cfg.get("device", None)

    info: Dict[str, Any] = {"solver": "sense", "iters": iters}

    try:
        mask = None
        sensitivity_maps = None

        if hasattr(physics, 'mask'):
            mask = physics.mask
        if hasattr(physics, 'sensitivity_maps'):
            sensitivity_maps = physics.sensitivity_maps

        if hasattr(physics, 'info'):
            op_info = physics.info()
            if 'mask' in op_info:
                mask = op_info['mask']
            if 'sensitivity_maps' in op_info:
                sensitivity_maps = op_info['sensitivity_maps']

        # Default mask: all ones
        if mask is None:
            if y.ndim == 3:
                mask = np.ones(y.shape[1:], dtype=np.float32)
            else:
                mask = np.ones(y.shape, dtype=np.float32)

        if y.ndim == 3:
            if sensitivity_maps is None:
                sensitivity_maps = estimate_sensitivity_maps(y, device=device)
            result = sense_reconstruction(
                y, sensitivity_maps, mask, regularization, iters,
                device=device,
            )
            result = np.abs(result).astype(np.float32)
        else:
            # Single-coil fallback: CS-MRI
            result = cs_mri_wavelet(y, mask, 0.01, iters, device=device)
            result = np.abs(result).astype(np.float32)

        return result, info

    except Exception as e:
        info["error"] = str(e)
        result = zero_filled_reconstruction(y)
        return result, info


def run_espirit_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run ESPIRiT-based MRI reconstruction.

    Args:
        y: K-space measurements
        physics: MRI physics operator
        cfg: Configuration with:
            - method: 'sense', 'cs', or 'zerofill' (default: 'sense')
            - iters: Iterations (default: 30)
            - lam: CS regularization (default: 0.01)

    Returns:
        Tuple of (reconstructed image, info_dict)
    """
    method = cfg.get("method", "sense")
    iters = cfg.get("iters", 30)
    lam = cfg.get("lam", 0.01)
    device = cfg.get("device", None)

    info = {
        "solver": f"espirit_{method}",
        "iters": iters,
    }

    try:
        # Get mask from physics
        mask = None
        sensitivity_maps = None

        if hasattr(physics, 'mask'):
            mask = physics.mask
        if hasattr(physics, 'sensitivity_maps'):
            sensitivity_maps = physics.sensitivity_maps

        if hasattr(physics, 'info'):
            op_info = physics.info()
            if 'mask' in op_info:
                mask = op_info['mask']

        # Default mask: all ones (fully sampled)
        if mask is None:
            if y.ndim == 3:
                mask = np.ones(y.shape[1:], dtype=np.float32)
            else:
                mask = np.ones(y.shape, dtype=np.float32)

        if method == "zerofill":
            result = zero_filled_reconstruction(y, mask, device=device)
        elif method == "cs":
            result = cs_mri_wavelet(y, mask, lam, iters, sensitivity_maps,
                                    device=device)
            result = np.abs(result).astype(np.float32)
        else:  # sense
            if y.ndim == 3:
                if sensitivity_maps is None:
                    sensitivity_maps = estimate_sensitivity_maps(y, device=device)
                result = sense_reconstruction(y, sensitivity_maps, mask, 0.001,
                                              iters, device=device)
                result = np.abs(result).astype(np.float32)
            else:
                # Single-coil: use CS
                result = cs_mri_wavelet(y, mask, lam, iters, device=device)
                result = np.abs(result).astype(np.float32)

        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Fall back to zero-filled
        result = zero_filled_reconstruction(y)
        return result, info


def run_cs_mri(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run CS-MRI reconstruction.

    Alias for run_espirit_recon with method='cs'.
    """
    cfg = dict(cfg)
    cfg['method'] = 'cs'
    return run_espirit_recon(y, physics, cfg)
