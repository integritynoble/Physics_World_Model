"""Nuclear emission category module – shared physics for ~10 nuclear/emission imaging modalities.

Handles modalities based on emission tomography / attenuated Radon:
  PET, SPECT, PET-CT, SPECT-CT, DOT, bioluminescence_tomo,
  photoacoustic, ultrasound, doppler_ultrasound, etc.

DAG patterns: A --> Pi --> D, A --> D, E --> Pi --> D, P --> D, P --> R --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Attenuated Radon Transform
# ---------------------------------------------------------------------------

def attenuated_radon(
    activity: np.ndarray,
    attenuation: Optional[np.ndarray] = None,
    angles: Optional[np.ndarray] = None,
    n_angles: int = 180,
) -> np.ndarray:
    """Attenuated Radon transform for emission tomography.

    For PET/SPECT, photons are attenuated along the path from emission
    point to detector.

    Args:
        activity: 2-D activity distribution (H, W).
        attenuation: 2-D attenuation map (same shape). If None, no attenuation.
        angles: Projection angles in degrees.
        n_angles: Number of angles if *angles* is None.

    Returns:
        Sinogram (n_angles, n_detectors).
    """
    from scipy.ndimage import rotate

    if angles is None:
        angles = np.linspace(0, 180, n_angles, endpoint=False)

    H, W = activity.shape
    N = max(H, W)
    # Pad to square
    if H != W:
        pad_h = (N - H) // 2
        pad_w = (N - W) // 2
        act_pad = np.pad(activity, ((pad_h, N - H - pad_h), (pad_w, N - W - pad_w)))
        if attenuation is not None:
            att_pad = np.pad(attenuation, ((pad_h, N - H - pad_h), (pad_w, N - W - pad_w)))
        else:
            att_pad = None
    else:
        act_pad = activity.copy()
        att_pad = attenuation.copy() if attenuation is not None else None

    sinogram = np.zeros((len(angles), N), dtype=np.float32)

    for i, angle in enumerate(angles):
        rot_act = rotate(act_pad, angle, reshape=False, order=1)
        if att_pad is not None:
            rot_att = rotate(att_pad, angle, reshape=False, order=1)
            # Compute cumulative attenuation along columns
            cum_att = np.cumsum(rot_att, axis=0)
            atten_factor = np.exp(-cum_att * 0.01)  # scale factor
            sinogram[i] = (rot_act * atten_factor).sum(axis=0)
        else:
            sinogram[i] = rot_act.sum(axis=0)

    return sinogram


def osem_reconstruct(
    sinogram: np.ndarray,
    angles: Optional[np.ndarray] = None,
    n_subsets: int = 8,
    n_iterations: int = 4,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """Simplified OSEM (Ordered Subsets Expectation Maximisation) for PET/SPECT.

    Uses MLEM as a simplified stand-in (1 subset = MLEM).
    """
    from scipy.ndimage import rotate

    n_angles, n_det = sinogram.shape
    if angles is None:
        angles = np.linspace(0, 180, n_angles, endpoint=False)
    if output_size is None:
        output_size = n_det

    # Initialize with uniform
    recon = np.ones((output_size, output_size), dtype=np.float32)
    sinogram_pos = np.clip(sinogram, 1e-6, None)

    for it in range(n_iterations):
        # Forward project
        fwd_sino = np.zeros_like(sinogram)
        for i, angle in enumerate(angles):
            rot = rotate(recon, angle, reshape=False, order=1)
            fwd_sino[i] = rot.sum(axis=0)

        # Ratio
        ratio_sino = sinogram_pos / (fwd_sino + 1e-10)

        # Back project ratio
        correction = np.zeros_like(recon)
        sensitivity = np.zeros_like(recon)
        for i, angle in enumerate(angles):
            proj = np.tile(ratio_sino[i], (output_size, 1))
            if proj.shape[1] != output_size:
                from scipy.ndimage import zoom
                proj = zoom(proj, (1, output_size / proj.shape[1]), order=1)
            correction += rotate(proj, -angle, reshape=False, order=1)
            ones_proj = np.ones((output_size, output_size), dtype=np.float32)
            sensitivity += rotate(
                rotate(ones_proj, angle, reshape=False, order=1),
                -angle, reshape=False, order=1,
            )

        recon *= correction / (sensitivity + 1e-10)
        recon = np.clip(recon, 0, None)

    return recon.astype(np.float32)


# ---------------------------------------------------------------------------
# Ultrasound model (simplified)
# ---------------------------------------------------------------------------

def ultrasound_bmode(
    tissue: np.ndarray,
    n_scanlines: int = 128,
    axial_samples: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Simplified B-mode ultrasound simulation.

    Models tissue as reflectivity map with speckle noise.
    """
    rng = np.random.RandomState(seed)
    from scipy.ndimage import zoom

    H, W = tissue.shape
    # Resample to scanline geometry
    bmode = zoom(tissue, (axial_samples / H, n_scanlines / W), order=1)

    # Add speckle noise (Rayleigh distributed)
    speckle = rng.rayleigh(1.0, bmode.shape).astype(np.float32)
    bmode = bmode * speckle

    # Log compression
    bmode = np.log1p(np.clip(bmode, 0, None))
    bmode = bmode / (bmode.max() + 1e-12)

    return bmode.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """Emission / nuclear imaging forward model."""
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: attenuated Radon
    if x.ndim == 2:
        return attenuated_radon(x)
    if x.ndim == 3:
        return np.stack([attenuated_radon(x[..., c]) for c in range(x.shape[-1])], axis=-1)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """Emission adjoint: OSEM reconstruction."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass

    if y.ndim == 2:
        return osem_reconstruct(y, n_iterations=2)
    if y.ndim == 3:
        return np.stack([osem_reconstruct(y[..., c], n_iterations=2) for c in range(y.shape[-1])], axis=-1)
    return y


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate an emission / activity phantom for nuclear imaging benchmarks."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)

    # Body outline (ellipse)
    body = (X / 0.8) ** 2 + (Y / 0.95) ** 2 <= 1.0
    x[body] = 0.1  # background activity

    # Organs with different uptake
    organs = [
        (0.0, 0.0, 0.25, 0.3, 0.4),     # heart
        (-0.3, 0.2, 0.15, 0.2, 0.6),     # left kidney
        (0.3, 0.2, 0.15, 0.2, 0.6),      # right kidney
        (0.0, -0.3, 0.2, 0.15, 0.3),     # liver
    ]
    for cx, cy, rx, ry, intensity in organs:
        mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
        x[mask] = intensity

    # Hot spots (tumours)
    for _ in range(3):
        hx = rng.uniform(-0.5, 0.5)
        hy = rng.uniform(-0.5, 0.5)
        hr = rng.uniform(0.03, 0.08)
        hot = ((X - hx) / hr) ** 2 + ((Y - hy) / hr) ** 2 <= 1.0
        x[hot & body] = rng.uniform(0.8, 1.0)

    if len(dims) >= 3:
        return np.stack([x] * dims[2], axis=-1)
    return np.clip(x, 0, 1).astype(np.float32)
