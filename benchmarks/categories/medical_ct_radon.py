"""Medical CT / Radon category module – shared physics for ~15 CT modalities.

Handles modalities based on Radon / line-integral projection:
  CT, CBCT, industrial_ct, mammography, fluoroscopy, dexa,
  electron_tomography, muon_tomo, proton_radiography, etc.

DAG patterns: Pi --> D, Pi --> N --> D, Pi --> I --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Radon transform (simplified, no external dependency required)
# ---------------------------------------------------------------------------

def radon_transform(
    image: np.ndarray,
    angles: Optional[np.ndarray] = None,
    n_angles: int = 180,
) -> np.ndarray:
    """Compute the Radon transform (sinogram) of a 2-D image.

    Uses rotation-based projection. For benchmark purposes, this is
    a simplified but correct implementation.

    Args:
        image: 2-D input image (H, W).
        angles: Projection angles in degrees.
        n_angles: Number of uniformly spaced angles if *angles* is None.

    Returns:
        Sinogram of shape ``(n_angles, n_detectors)`` where
        ``n_detectors = max(H, W)``.
    """
    if angles is None:
        angles = np.linspace(0, 180, n_angles, endpoint=False)

    from scipy.ndimage import rotate

    H, W = image.shape
    n_det = max(H, W)
    # Pad to square
    pad = abs(H - W) // 2
    if H < W:
        padded = np.pad(image, ((pad, pad + (W - H) % 2), (0, 0)), mode="constant")
    elif W < H:
        padded = np.pad(image, ((0, 0), (pad, pad + (H - W) % 2)), mode="constant")
    else:
        padded = image.copy()

    N = padded.shape[0]
    sinogram = np.zeros((len(angles), N), dtype=np.float32)

    for i, angle in enumerate(angles):
        rotated = rotate(padded, angle, reshape=False, order=1)
        sinogram[i] = rotated.sum(axis=0)

    return sinogram


def filtered_back_projection(
    sinogram: np.ndarray,
    angles: Optional[np.ndarray] = None,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """Filtered Back Projection reconstruction.

    Args:
        sinogram: Shape ``(n_angles, n_detectors)``.
        angles: Projection angles in degrees.
        output_size: Side length of output image.

    Returns:
        Reconstructed image.
    """
    n_angles, n_det = sinogram.shape
    if angles is None:
        angles = np.linspace(0, 180, n_angles, endpoint=False)
    if output_size is None:
        output_size = n_det

    from scipy.ndimage import rotate

    # Ram-Lak filter
    freq = np.fft.fftfreq(n_det)
    ramp = np.abs(freq).astype(np.float32)

    # Filter each projection
    filtered = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj_fft = np.fft.fft(sinogram[i])
        filtered[i] = np.real(np.fft.ifft(proj_fft * ramp))

    # Back projection
    recon = np.zeros((output_size, output_size), dtype=np.float32)
    for i, angle in enumerate(angles):
        # Expand projection to 2-D and rotate back
        proj_2d = np.tile(filtered[i], (output_size, 1))
        # Crop/pad to output_size
        if proj_2d.shape[1] > output_size:
            start = (proj_2d.shape[1] - output_size) // 2
            proj_2d = proj_2d[:, start:start + output_size]
        elif proj_2d.shape[1] < output_size:
            pad = output_size - proj_2d.shape[1]
            proj_2d = np.pad(proj_2d, ((0, 0), (pad // 2, pad - pad // 2)), mode="constant")
        rotated = rotate(proj_2d, -angle, reshape=False, order=1)
        recon += rotated

    recon *= np.pi / (2 * n_angles)
    return np.clip(recon, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """Radon projection forward model."""
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: direct Radon
    if x.ndim == 2:
        return radon_transform(x)
    # For 3-D (volume), project slice by slice
    if x.ndim == 3:
        sinograms = []
        for s in range(x.shape[-1]):
            sinograms.append(radon_transform(x[..., s]))
        return np.stack(sinograms, axis=-1)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """FBP adjoint / reconstruction."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass

    if y.ndim == 2:
        return filtered_back_projection(y)
    if y.ndim == 3:
        slices = []
        for s in range(y.shape[-1]):
            slices.append(filtered_back_projection(y[..., s]))
        return np.stack(slices, axis=-1)
    return y


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a Shepp-Logan phantom for CT benchmarks."""
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H

    # Shepp-Logan ellipse parameters: (cx, cy, rx, ry, angle_deg, intensity)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),
        (0.06, -0.605, 0.023, 0.046, 0, 0.01),
    ]

    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    x = np.zeros((H, W), dtype=np.float64)

    for cx, cy, rx, ry, angle, intensity in ellipses:
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Xr = cos_t * (X - cx) + sin_t * (Y - cy)
        Yr = -sin_t * (X - cx) + cos_t * (Y - cy)
        mask = (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1.0
        x[mask] += intensity

    x = ((x - x.min()) / (x.max() - x.min() + 1e-12)).astype(np.float32)

    if len(dims) >= 3:
        C = dims[2]
        return np.stack([x] * C, axis=-1)
    return x
