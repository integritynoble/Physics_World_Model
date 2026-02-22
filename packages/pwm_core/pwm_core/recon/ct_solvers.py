"""CT Reconstruction Solvers: FBP and SART.

Classical algorithms for Computed Tomography reconstruction.

References:
- Feldkamp, L.A. et al. (1984). "Practical cone-beam algorithm"
- Andersen, A.H. & Kak, A.C. (1984). "Simultaneous Algebraic Reconstruction Technique (SART)"

Benchmark: LoDoPaB-CT dataset
Expected PSNR:
- FBP: 35.2 dB (1000 views), 28.1 dB (128 views), 24.3 dB (64 views)
- SART: 36.1 dB (1000 views), 30.5 dB (128 views), 27.2 dB (64 views)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def create_ramlak_filter(size: int, pixel_size: float = 1.0) -> np.ndarray:
    """Create Ram-Lak (ramp) filter for FBP.

    Args:
        size: Filter size (should match projection width)
        pixel_size: Detector pixel size

    Returns:
        Ram-Lak filter in frequency domain
    """
    freqs = np.fft.fftfreq(size, d=pixel_size)
    ramp = np.abs(freqs)
    return ramp.astype(np.float32)


def create_shepp_logan_filter(size: int, pixel_size: float = 1.0) -> np.ndarray:
    """Create Shepp-Logan filter for FBP (smoother than Ram-Lak).

    Args:
        size: Filter size
        pixel_size: Detector pixel size

    Returns:
        Shepp-Logan filter in frequency domain
    """
    freqs = np.fft.fftfreq(size, d=pixel_size)
    ramp = np.abs(freqs)
    # Sinc windowing
    sinc = np.sinc(2 * freqs * pixel_size)
    return (ramp * np.abs(sinc)).astype(np.float32)


def create_cosine_filter(size: int, pixel_size: float = 1.0) -> np.ndarray:
    """Create Cosine filter for FBP.

    Args:
        size: Filter size
        pixel_size: Detector pixel size

    Returns:
        Cosine filter in frequency domain
    """
    freqs = np.fft.fftfreq(size, d=pixel_size)
    ramp = np.abs(freqs)
    # Cosine windowing
    nyquist = 0.5 / pixel_size
    cosine = np.cos(np.pi * freqs / (2 * nyquist + 1e-10))
    cosine = np.clip(cosine, 0, 1)
    return (ramp * cosine).astype(np.float32)


def fbp_2d(
    sinogram: np.ndarray,
    angles: np.ndarray,
    filter_type: str = "ramlak",
    output_size: Optional[int] = None,
    device=None,
) -> np.ndarray:
    """Filtered Back-Projection for parallel-beam CT.

    Args:
        sinogram: Sinogram data (n_angles, n_detectors)
        angles: Projection angles in radians
        filter_type: 'ramlak', 'shepp_logan', 'cosine', or 'none'
        output_size: Size of output image (default: n_detectors)
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed image (output_size, output_size)
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _fbp_2d_torch(sinogram, angles, filter_type, output_size, dev)

    from scipy.ndimage import map_coordinates

    n_angles, n_detectors = sinogram.shape
    sinogram = sinogram.astype(np.float32)

    if output_size is None:
        output_size = n_detectors

    # Create filter
    if filter_type == "ramlak":
        filt = create_ramlak_filter(n_detectors)
    elif filter_type == "shepp_logan":
        filt = create_shepp_logan_filter(n_detectors)
    elif filter_type == "cosine":
        filt = create_cosine_filter(n_detectors)
    else:
        filt = np.ones(n_detectors, dtype=np.float32)

    # Apply filter in frequency domain
    filtered_sinogram = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj_fft = np.fft.fft(sinogram[i])
        filtered_fft = proj_fft * filt
        filtered_sinogram[i] = np.real(np.fft.ifft(filtered_fft))

    # Back-projection
    reconstruction = np.zeros((output_size, output_size), dtype=np.float32)

    # Create coordinate grid
    center = output_size / 2
    x = np.arange(output_size) - center
    y = np.arange(output_size) - center
    X, Y = np.meshgrid(x, y)

    for i, angle in enumerate(angles):
        # Compute projection coordinates
        t = X * np.cos(angle) + Y * np.sin(angle)

        # Map to detector indices
        detector_coords = t + n_detectors / 2

        # Interpolate from filtered projection
        proj_interp = np.interp(
            detector_coords.flatten(),
            np.arange(n_detectors),
            filtered_sinogram[i],
            left=0, right=0
        ).reshape(output_size, output_size)

        reconstruction += proj_interp

    # Normalize
    reconstruction *= np.pi / n_angles

    return reconstruction.astype(np.float32)


def _fbp_2d_torch(sinogram, angles, filter_type, output_size, dev):
    """GPU implementation of fbp_2d.

    Batch FFT filtering + grid_sample-based back-projection.
    """
    import torch
    import torch.nn.functional as F
    from pwm_core.recon.gpu_utils import to_torch, to_numpy

    sinogram_t = to_torch(sinogram, dev)
    n_angles, n_detectors = sinogram_t.shape

    if output_size is None:
        output_size = n_detectors

    # Create filter on device
    freqs = torch.fft.fftfreq(n_detectors, device=dev)
    if filter_type == "ramlak":
        filt = torch.abs(freqs)
    elif filter_type == "shepp_logan":
        ramp = torch.abs(freqs)
        sinc = torch.sinc(2 * freqs)
        filt = ramp * torch.abs(sinc)
    elif filter_type == "cosine":
        ramp = torch.abs(freqs)
        nyquist = 0.5
        cosine = torch.cos(torch.pi * freqs / (2 * nyquist + 1e-10))
        cosine = torch.clamp(cosine, 0, 1)
        filt = ramp * cosine
    else:
        filt = torch.ones(n_detectors, device=dev)

    # Batch filter all projections at once
    sino_fft = torch.fft.fft(sinogram_t, dim=1)
    filtered = torch.real(torch.fft.ifft(sino_fft * filt.unsqueeze(0), dim=1))

    # Back-projection using grid_sample
    reconstruction = torch.zeros(output_size, output_size, device=dev,
                                 dtype=torch.float32)

    center = output_size / 2.0
    x = torch.arange(output_size, device=dev, dtype=torch.float32) - center
    y = torch.arange(output_size, device=dev, dtype=torch.float32) - center
    Y, X = torch.meshgrid(y, x, indexing='ij')

    angles_t = to_torch(np.asarray(angles, dtype=np.float32), dev)

    for i in range(n_angles):
        angle = angles_t[i]
        t = X * torch.cos(angle) + Y * torch.sin(angle)
        detector_coords = t + n_detectors / 2.0

        # Normalize to [-1, 1] for grid_sample
        grid_x = (detector_coords / (n_detectors - 1)) * 2 - 1
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        proj = filtered[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # grid_sample expects (N,C,H_in,W_in), grid (N,H_out,W_out,2)
        proj_2d = proj.expand(-1, -1, 1, -1)
        interp = F.grid_sample(proj_2d, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
        reconstruction += interp.squeeze()

    reconstruction *= np.pi / n_angles

    return to_numpy(reconstruction).astype(np.float32)


def sart_2d(
    sinogram: np.ndarray,
    angles: np.ndarray,
    output_size: Optional[int] = None,
    iterations: int = 10,
    relaxation: float = 1.0,
    initial_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simultaneous Algebraic Reconstruction Technique.

    SART iteratively updates the reconstruction by back-projecting
    the residual between measured and computed projections.

    Note: This function uses explicit pixel loops and stays CPU-only.
    Use sart_operator for GPU-accelerated operator-based SART.

    Args:
        sinogram: Sinogram data (n_angles, n_detectors)
        angles: Projection angles in radians
        output_size: Size of output image
        iterations: Number of SART iterations
        relaxation: Relaxation parameter (0 < lambda <= 1)
        initial_guess: Initial reconstruction estimate

    Returns:
        Reconstructed image
    """
    n_angles, n_detectors = sinogram.shape
    sinogram = sinogram.astype(np.float32)

    if output_size is None:
        output_size = n_detectors

    # Initialize reconstruction
    if initial_guess is not None:
        recon = initial_guess.astype(np.float32)
    else:
        recon = np.zeros((output_size, output_size), dtype=np.float32)

    # Create coordinate grid
    center = output_size / 2
    x = np.arange(output_size) - center
    y = np.arange(output_size) - center
    X, Y = np.meshgrid(x, y)

    for iteration in range(iterations):
        for i, angle in enumerate(angles):
            # Forward project current estimate
            t = X * np.cos(angle) + Y * np.sin(angle)
            detector_coords = t + n_detectors / 2

            # Compute projection
            detector_indices = detector_coords.astype(int)
            valid_mask = (detector_indices >= 0) & (detector_indices < n_detectors)

            projection = np.zeros(n_detectors, dtype=np.float32)
            counts = np.zeros(n_detectors, dtype=np.float32)

            for dy in range(output_size):
                for dx in range(output_size):
                    idx = int(detector_coords[dy, dx])
                    if 0 <= idx < n_detectors:
                        projection[idx] += recon[dy, dx]
                        counts[idx] += 1

            # Normalize projection (simple sum along rays)
            counts = np.maximum(counts, 1)

            # Compute residual
            residual = (sinogram[i] - projection) / counts

            # Back-project residual
            for dy in range(output_size):
                for dx in range(output_size):
                    idx = int(detector_coords[dy, dx])
                    if 0 <= idx < n_detectors:
                        recon[dy, dx] += relaxation * residual[idx]

            # Enforce non-negativity
            recon = np.maximum(recon, 0)

    return recon.astype(np.float32)


def sart_operator(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    iterations: int = 10,
    relaxation: float = 0.25,
    device=None,
) -> np.ndarray:
    """SART using forward/adjoint operators.

    More general version using arbitrary projection operators.

    Args:
        y: Measurements (sinogram)
        forward: Forward projection operator
        adjoint: Back-projection operator
        x_shape: Shape of reconstruction
        iterations: Number of iterations
        relaxation: Relaxation parameter
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed image
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _sart_operator_torch(y, forward, adjoint, x_shape, iterations,
                                    relaxation, dev)

    y = y.astype(np.float32)

    # Initialize with back-projection
    x = adjoint(y).reshape(x_shape).astype(np.float32)

    # Compute normalization (for proper SART scaling)
    ones_y = np.ones_like(y)
    norm = adjoint(ones_y).reshape(x_shape)
    norm = np.maximum(norm, 1e-10)

    for i in range(iterations):
        # Forward project
        Ax = forward(x)

        # Residual
        residual = y - Ax

        # Back-project residual
        bp_residual = adjoint(residual).reshape(x_shape)

        # Update with normalization
        x = x + relaxation * bp_residual / norm

        # Non-negativity
        x = np.maximum(x, 0)

    return x.astype(np.float32)


def _sart_operator_torch(y, forward, adjoint, x_shape, iterations, relaxation,
                         dev):
    """GPU implementation of sart_operator."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy, wrap_operator

    y_t = to_torch(y, dev)
    fwd = wrap_operator(forward, dev)
    adj = wrap_operator(adjoint, dev)

    x = adj(y_t).reshape(x_shape)

    ones_y = torch.ones_like(y_t)
    norm = adj(ones_y).reshape(x_shape)
    norm = torch.clamp(norm, min=1e-10)

    for i in range(iterations):
        Ax = fwd(x)
        residual = y_t - Ax
        bp_residual = adj(residual).reshape(x_shape)
        x = x + relaxation * bp_residual / norm
        x = torch.clamp(x, min=0)

    return to_numpy(x).astype(np.float32)


def run_fbp(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run FBP reconstruction.

    Args:
        y: Sinogram measurements
        physics: CT physics operator
        cfg: Configuration with:
            - filter: 'ramlak', 'shepp_logan', 'cosine', 'none'
            - output_size: Output image size

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    filter_type = cfg.get("filter", "ramlak")
    output_size = cfg.get("output_size", None)
    device = cfg.get("device", None)

    info = {
        "solver": "fbp",
        "filter": filter_type,
    }

    try:
        # Get angles from physics
        angles = None
        n_angles = y.shape[0]

        if hasattr(physics, 'angles'):
            angles = physics.angles
        elif hasattr(physics, 'info'):
            op_info = physics.info()
            if 'angles' in op_info:
                angles = op_info['angles']
            elif 'n_angles' in op_info:
                n_angles = op_info['n_angles']

        if angles is None:
            # Default: uniform angles from 0 to pi
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        # Reshape sinogram if needed
        if y.ndim == 1:
            # Flat sinogram, need to reshape
            n_detectors = int(np.sqrt(len(y) / n_angles))
            y = y.reshape(n_angles, n_detectors)

        result = fbp_2d(y, angles, filter_type, output_size, device=device)

        info["n_angles"] = len(angles)
        info["output_size"] = result.shape[0]

        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Fall back to adjoint
        if hasattr(physics, 'adjoint'):
            result = physics.adjoint(y)
            if hasattr(physics, 'x_shape'):
                result = result.reshape(physics.x_shape)
            return result.astype(np.float32), info
        return y.astype(np.float32), info


def run_sart(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run SART reconstruction.

    Args:
        y: Sinogram measurements
        physics: CT physics operator
        cfg: Configuration with:
            - iters: Number of iterations (default: 10)
            - relaxation: Relaxation parameter (default: 0.25)

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    iters = cfg.get("iters", 10)
    relaxation = cfg.get("relaxation", 0.25)
    device = cfg.get("device", None)

    info = {
        "solver": "sart",
        "iters": iters,
        "relaxation": relaxation,
    }

    try:
        if hasattr(physics, 'forward') and hasattr(physics, 'adjoint'):
            x_shape = y.shape
            if hasattr(physics, 'x_shape'):
                x_shape = tuple(physics.x_shape)
            elif hasattr(physics, 'info'):
                op_info = physics.info()
                if 'x_shape' in op_info:
                    x_shape = tuple(op_info['x_shape'])

            result = sart_operator(
                y, physics.forward, physics.adjoint,
                x_shape, iters, relaxation, device=device
            )
            return result, info

        # Fall back to angle-based SART (CPU-only)
        angles = None
        n_angles = y.shape[0] if y.ndim >= 2 else int(np.sqrt(len(y)))

        if hasattr(physics, 'angles'):
            angles = physics.angles

        if angles is None:
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        if y.ndim == 1:
            n_detectors = int(np.sqrt(len(y) / n_angles))
            y = y.reshape(n_angles, n_detectors)

        result = sart_2d(y, angles, None, iters, relaxation)
        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, 'adjoint'):
            result = physics.adjoint(y)
            return result.astype(np.float32), info
        return y.astype(np.float32), info
