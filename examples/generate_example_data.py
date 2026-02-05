"""Generate synthetic example data for all PWM modalities.

This script creates synthetic measurements and ground truth data for each
of the 16 imaging modalities supported by PWM.

Usage:
    python examples/generate_example_data.py

Output:
    examples/data/<modality>/measurements.npy
    examples/data/<modality>/ground_truth.npy
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import ndimage


def get_examples_dir() -> Path:
    """Get the examples directory path."""
    return Path(__file__).parent


def create_phantom_2d(size: int = 64, seed: int = 42) -> np.ndarray:
    """Create a 2D test phantom with geometric shapes.

    Args:
        size: Image size (square).
        seed: Random seed for reproducibility.

    Returns:
        2D phantom array normalized to [0, 1].
    """
    np.random.seed(seed)
    phantom = np.zeros((size, size), dtype=np.float32)

    # Add circles
    y, x = np.ogrid[:size, :size]
    center = size // 2

    # Large circle
    r1 = size // 4
    mask1 = (x - center) ** 2 + (y - center) ** 2 <= r1 ** 2
    phantom[mask1] = 0.8

    # Small circles
    for _ in range(5):
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        r = np.random.randint(3, 8)
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        phantom[mask] = np.random.uniform(0.5, 1.0)

    # Add rectangle
    phantom[size//3:2*size//3, size//4:size//3] = 0.6

    # Normalize
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-8)
    return phantom.astype(np.float32)


def create_phantom_3d(size: int = 64, depth: int = 32, seed: int = 42) -> np.ndarray:
    """Create a 3D test phantom.

    Args:
        size: XY size.
        depth: Z depth.
        seed: Random seed.

    Returns:
        3D phantom array (H, W, D).
    """
    np.random.seed(seed)
    phantom = np.zeros((size, size, depth), dtype=np.float32)

    # Add spheres
    z, y, x = np.ogrid[:depth, :size, :size]
    z = z.transpose(1, 2, 0)
    y = y.transpose(1, 2, 0)
    x = x.transpose(1, 2, 0)

    # Central sphere
    cx, cy, cz = size // 2, size // 2, depth // 2
    r = min(size, depth) // 4
    mask = (x - cx) ** 2 + (y - cy) ** 2 + ((z - cz) * (size / depth)) ** 2 <= r ** 2
    phantom[mask] = 0.8

    # Smaller spheres
    for _ in range(3):
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        cz = np.random.randint(depth // 4, 3 * depth // 4)
        r = np.random.randint(3, 6)
        mask = (x - cx) ** 2 + (y - cy) ** 2 + ((z - cz) * (size / depth)) ** 2 <= r ** 2
        phantom[mask] = np.random.uniform(0.5, 1.0)

    return phantom.astype(np.float32)


def create_spectral_phantom(size: int = 64, bands: int = 16, seed: int = 42) -> np.ndarray:
    """Create a spectral data cube.

    Args:
        size: Spatial size.
        bands: Number of spectral bands.
        seed: Random seed.

    Returns:
        Spectral cube (H, W, L).
    """
    np.random.seed(seed)
    # Create spatial pattern
    spatial = create_phantom_2d(size, seed)

    # Create spectral signatures
    cube = np.zeros((size, size, bands), dtype=np.float32)
    for b in range(bands):
        # Vary intensity across spectrum
        weight = np.exp(-((b - bands // 2) ** 2) / (bands / 2) ** 2)
        cube[:, :, b] = spatial * (0.5 + 0.5 * weight)

    return cube.astype(np.float32)


def add_noise(y: np.ndarray, photon_level: float = 1000, read_noise: float = 5.0) -> np.ndarray:
    """Add Poisson-Gaussian noise to measurements.

    Args:
        y: Clean measurements.
        photon_level: Max photon count for scaling.
        read_noise: Gaussian read noise std.

    Returns:
        Noisy measurements.
    """
    # Scale to photon counts
    y_scaled = np.clip(y, 0, None) * photon_level

    # Poisson noise
    y_poisson = np.random.poisson(y_scaled.astype(np.float64)).astype(np.float32)

    # Gaussian read noise
    y_noisy = y_poisson + np.random.normal(0, read_noise, y.shape).astype(np.float32)

    # Normalize back
    y_noisy = y_noisy / photon_level

    return y_noisy.astype(np.float32)


# ============================================================================
# Forward models for each modality
# ============================================================================

def forward_widefield(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Widefield: Gaussian PSF blur."""
    return ndimage.gaussian_filter(x, sigma=sigma).astype(np.float32)


def forward_confocal(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Confocal: Sharper PSF with some blur."""
    return ndimage.gaussian_filter(x, sigma=sigma).astype(np.float32)


def forward_sim(x: np.ndarray, n_patterns: int = 9) -> np.ndarray:
    """SIM: Generate pattern-modulated measurements.

    Returns stack of shape (H, W, n_patterns).
    """
    H, W = x.shape[:2]
    measurements = np.zeros((H, W, n_patterns), dtype=np.float32)

    for i in range(n_patterns):
        # Create sinusoidal pattern
        angle = (i % 3) * np.pi / 3
        phase = (i // 3) * 2 * np.pi / 3
        freq = 0.1

        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        pattern = 0.5 + 0.5 * np.cos(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)) + phase)

        # Modulate and blur
        modulated = x * pattern
        measurements[:, :, i] = ndimage.gaussian_filter(modulated, sigma=1.5)

    return measurements.astype(np.float32)


def forward_cassi(x_spectral: np.ndarray) -> np.ndarray:
    """CASSI: Coded aperture spectral imaging.

    Args:
        x_spectral: Spectral cube (H, W, L).

    Returns:
        2D coded snapshot (H, W).
    """
    H, W, L = x_spectral.shape

    # Random binary mask
    np.random.seed(123)
    mask = (np.random.rand(H, W) > 0.5).astype(np.float32)

    # Simulate dispersion: shift each band
    y = np.zeros((H, W), dtype=np.float32)
    for l in range(L):
        shift = l - L // 2  # Dispersion shift
        shifted = np.roll(x_spectral[:, :, l], shift, axis=1)
        y += mask * shifted

    return y.astype(np.float32)


def forward_spc(x: np.ndarray, n_measurements: int = 614) -> np.ndarray:
    """Single-pixel camera: Hadamard-like measurements.

    Returns 1D measurement vector.
    """
    H, W = x.shape
    N = H * W
    x_flat = x.flatten()

    # Random measurement matrix (simplified Hadamard-like)
    np.random.seed(456)
    A = (np.random.rand(n_measurements, N) > 0.5).astype(np.float32) * 2 - 1
    y = A @ x_flat

    return y.astype(np.float32)


def forward_lensless(x: np.ndarray, psf_size: int = 128) -> np.ndarray:
    """Lensless: Diffuser PSF convolution.

    Returns larger measurement due to PSF spread.
    """
    # Create random diffuser PSF
    np.random.seed(789)
    psf = np.random.rand(psf_size, psf_size).astype(np.float32)
    psf = ndimage.gaussian_filter(psf, sigma=5)
    psf = psf / psf.sum()

    # Convolve (output is larger)
    y = ndimage.convolve(x, psf, mode='constant')

    return y.astype(np.float32)


def forward_lightsheet(x: np.ndarray) -> np.ndarray:
    """Light-sheet: Add stripe artifacts.

    Operates on 3D volume.
    """
    y = x.copy()

    # Add horizontal stripes (light-sheet artifacts)
    H, W, D = y.shape
    for z in range(D):
        stripe = 0.9 + 0.2 * np.sin(2 * np.pi * np.arange(H) / 10)[:, None]
        y[:, :, z] *= stripe

    # Add attenuation with depth
    for z in range(D):
        attenuation = np.exp(-z * 0.02)
        y[:, :, z] *= attenuation

    return y.astype(np.float32)


def forward_ct(x: np.ndarray, n_angles: int = 180) -> np.ndarray:
    """CT: Radon transform (sinogram).

    Simplified projection.
    """
    H, W = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, W), dtype=np.float32)

    for i, angle in enumerate(angles):
        rotated = ndimage.rotate(x, angle, reshape=False, mode='constant')
        sinogram[i, :] = rotated.sum(axis=0)

    return sinogram.astype(np.float32)


def forward_mri(x: np.ndarray, sampling_rate: float = 0.25) -> np.ndarray:
    """MRI: FFT with undersampling.

    Returns complex k-space.
    """
    # Full k-space
    kspace = np.fft.fft2(x)
    kspace = np.fft.fftshift(kspace)

    # Random undersampling mask
    np.random.seed(321)
    mask = np.random.rand(*x.shape) < sampling_rate

    # Keep center lines (calibration region)
    H, W = x.shape
    center_h, center_w = H // 8, W // 8
    mask[H//2-center_h:H//2+center_h, W//2-center_w:W//2+center_w] = True

    # Apply mask
    kspace_under = kspace * mask

    return kspace_under.astype(np.complex64)


def forward_ptychography(x: np.ndarray, n_positions: int = 16) -> np.ndarray:
    """Ptychography: Overlapping diffraction patterns.

    Args:
        x: Complex object (or real, converted to complex).

    Returns:
        Stack of diffraction patterns (n_positions, H, W).
    """
    if not np.iscomplexobj(x):
        x = x.astype(np.complex64)

    H, W = x.shape
    probe_size = H // 2
    patterns = np.zeros((n_positions, H, W), dtype=np.float32)

    # Create probe
    yy, xx = np.meshgrid(np.arange(probe_size), np.arange(probe_size))
    probe = np.exp(-((xx - probe_size//2)**2 + (yy - probe_size//2)**2) / (probe_size/4)**2)
    probe = probe.astype(np.complex64)

    # Scan positions
    positions = []
    n_side = int(np.sqrt(n_positions))
    step = (H - probe_size) // (n_side - 1) if n_side > 1 else 0
    for i in range(n_side):
        for j in range(n_side):
            positions.append((i * step, j * step))

    for idx, (py, px) in enumerate(positions[:n_positions]):
        # Extract region
        region = x[py:py+probe_size, px:px+probe_size]
        if region.shape != (probe_size, probe_size):
            continue

        # Exit wave
        exit_wave = probe * region

        # Diffraction pattern (intensity)
        diff = np.fft.fftshift(np.fft.fft2(exit_wave))
        pattern = np.abs(diff) ** 2

        # Pad to full size
        patterns[idx, py:py+probe_size, px:px+probe_size] = pattern[:H-py, :W-px][:probe_size, :probe_size]

    return patterns.astype(np.float32)


def forward_holography(x: np.ndarray) -> np.ndarray:
    """Holography: Interference pattern.

    Args:
        x: Complex object field.

    Returns:
        Hologram intensity.
    """
    if not np.iscomplexobj(x):
        # Create complex field with phase
        phase = ndimage.gaussian_filter(np.random.rand(*x.shape), sigma=5) * 2 * np.pi
        x = x * np.exp(1j * phase)

    H, W = x.shape

    # Reference wave (off-axis)
    yy, xx = np.meshgrid(np.arange(W), np.arange(H))
    freq = 0.1
    reference = np.exp(1j * 2 * np.pi * freq * xx)

    # Hologram = |object + reference|^2
    hologram = np.abs(x + reference) ** 2

    return hologram.astype(np.float32)


def forward_nerf(x_volume: np.ndarray, n_views: int = 10) -> np.ndarray:
    """NeRF: Multi-view projections.

    Args:
        x_volume: 3D volume (H, W, D).

    Returns:
        Multi-view images (n_views, H, W, 3).
    """
    H, W, D = x_volume.shape
    views = np.zeros((n_views, H, W, 3), dtype=np.float32)

    for v in range(n_views):
        angle = v * 360 / n_views
        # Simple max projection along rotated axis
        rotated = ndimage.rotate(x_volume, angle, axes=(0, 2), reshape=False, mode='constant')
        projection = rotated.max(axis=2)
        # Convert to RGB
        views[v, :, :, 0] = projection
        views[v, :, :, 1] = projection
        views[v, :, :, 2] = projection

    return views.astype(np.float32)


def forward_gaussian_splatting(x_volume: np.ndarray, n_views: int = 10) -> np.ndarray:
    """Gaussian splatting: Similar to NeRF, multi-view rendering."""
    return forward_nerf(x_volume, n_views)


def forward_matrix(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Generic matrix: y = Ax."""
    return (A @ x.flatten()).astype(np.float32)


# ============================================================================
# Generate data for each modality
# ============================================================================

def generate_all_data():
    """Generate example data for all modalities."""
    data_dir = get_examples_dir() / "data"

    print("Generating example data for all modalities...")

    # 1. Widefield Basic
    print("  1/16: widefield_basic")
    x = create_phantom_2d(64)
    y = forward_widefield(x, sigma=2.0)
    y = add_noise(y, photon_level=1000, read_noise=5.0)
    np.save(data_dir / "widefield_basic" / "ground_truth.npy", x)
    np.save(data_dir / "widefield_basic" / "measurements.npy", y)

    # 2. Widefield Low Dose
    print("  2/16: widefield_lowdose")
    x = create_phantom_2d(64, seed=43)
    y = forward_widefield(x, sigma=2.0)
    y = add_noise(y, photon_level=100, read_noise=10.0)  # Low dose
    np.save(data_dir / "widefield_lowdose" / "ground_truth.npy", x)
    np.save(data_dir / "widefield_lowdose" / "measurements.npy", y)

    # 3. Confocal Live-cell
    print("  3/16: confocal_livecell")
    x = create_phantom_2d(64, seed=44)
    y = forward_confocal(x, sigma=1.5)
    # Add drift
    y = ndimage.shift(y, [1.5, 0.8])
    y = add_noise(y, photon_level=500, read_noise=3.0)
    np.save(data_dir / "confocal_livecell" / "ground_truth.npy", x)
    np.save(data_dir / "confocal_livecell" / "measurements.npy", y)

    # 4. Confocal 3D
    print("  4/16: confocal_3d")
    x = create_phantom_3d(64, 32, seed=45)
    y = ndimage.gaussian_filter(x, sigma=[1.5, 1.5, 1.0])  # Anisotropic PSF
    # Add attenuation
    for z in range(y.shape[2]):
        y[:, :, z] *= np.exp(-z * 0.03)
    y = add_noise(y, photon_level=800, read_noise=5.0)
    np.save(data_dir / "confocal_3d" / "ground_truth.npy", x)
    np.save(data_dir / "confocal_3d" / "measurements.npy", y)

    # 5. SIM
    print("  5/16: sim")
    x = create_phantom_2d(64, seed=46)
    y = forward_sim(x, n_patterns=9)
    y = add_noise(y, photon_level=600, read_noise=3.0)
    np.save(data_dir / "sim" / "ground_truth.npy", x)
    np.save(data_dir / "sim" / "measurements.npy", y)

    # 6. CASSI
    print("  6/16: cassi")
    x = create_spectral_phantom(64, 16, seed=47)
    y = forward_cassi(x)
    y = add_noise(y, photon_level=800, read_noise=5.0)
    np.save(data_dir / "cassi" / "ground_truth.npy", x)
    np.save(data_dir / "cassi" / "measurements.npy", y)

    # 7. Single-Pixel Camera
    print("  7/16: spc")
    x = create_phantom_2d(64, seed=48)
    y = forward_spc(x, n_measurements=614)
    y = add_noise(y, photon_level=1000, read_noise=10.0)
    np.save(data_dir / "spc" / "ground_truth.npy", x)
    np.save(data_dir / "spc" / "measurements.npy", y)

    # 8. Lensless
    print("  8/16: lensless")
    x = create_phantom_2d(64, seed=49)
    y = forward_lensless(x, psf_size=64)
    y = add_noise(y, photon_level=500, read_noise=8.0)
    np.save(data_dir / "lensless" / "ground_truth.npy", x)
    np.save(data_dir / "lensless" / "measurements.npy", y)

    # 9. Light-sheet
    print("  9/16: lightsheet")
    x = create_phantom_3d(64, 32, seed=50)
    y = forward_lightsheet(x)
    y = add_noise(y, photon_level=700, read_noise=4.0)
    np.save(data_dir / "lightsheet" / "ground_truth.npy", x)
    np.save(data_dir / "lightsheet" / "measurements.npy", y)

    # 10. CT
    print("  10/16: ct")
    x = create_phantom_2d(64, seed=51)
    y = forward_ct(x, n_angles=180)
    y = add_noise(y, photon_level=500, read_noise=5.0)
    np.save(data_dir / "ct" / "ground_truth.npy", x)
    np.save(data_dir / "ct" / "measurements.npy", y)

    # 11. MRI
    print("  11/16: mri")
    x = create_phantom_2d(64, seed=52)
    y = forward_mri(x, sampling_rate=0.25)
    # Add complex noise
    noise = (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)) * 0.01
    y = y + noise.astype(np.complex64)
    np.save(data_dir / "mri" / "ground_truth.npy", x)
    np.save(data_dir / "mri" / "measurements.npy", y)

    # 12. Ptychography
    print("  12/16: ptychography")
    x = create_phantom_2d(64, seed=53)
    # Add phase
    phase = ndimage.gaussian_filter(np.random.rand(64, 64), sigma=5) * np.pi
    x_complex = x * np.exp(1j * phase)
    y = forward_ptychography(x_complex, n_positions=16)
    y = add_noise(y, photon_level=1000, read_noise=5.0)
    np.save(data_dir / "ptychography" / "ground_truth.npy", x_complex)
    np.save(data_dir / "ptychography" / "measurements.npy", y)

    # 13. Holography
    print("  13/16: holography")
    x = create_phantom_2d(64, seed=54)
    y = forward_holography(x)
    y = add_noise(y, photon_level=1000, read_noise=3.0)
    np.save(data_dir / "holography" / "ground_truth.npy", x)
    np.save(data_dir / "holography" / "measurements.npy", y)

    # 14. NeRF
    print("  14/16: nerf")
    x = create_phantom_3d(64, 64, seed=55)
    y = forward_nerf(x, n_views=10)
    y = add_noise(y, photon_level=1000, read_noise=5.0)
    np.save(data_dir / "nerf" / "ground_truth.npy", x)
    np.save(data_dir / "nerf" / "measurements.npy", y)

    # 15. Gaussian Splatting
    print("  15/16: gaussian_splatting")
    x = create_phantom_3d(64, 64, seed=56)
    y = forward_gaussian_splatting(x, n_views=10)
    y = add_noise(y, photon_level=1000, read_noise=5.0)
    np.save(data_dir / "gaussian_splatting" / "ground_truth.npy", x)
    np.save(data_dir / "gaussian_splatting" / "measurements.npy", y)

    # 16. Generic Matrix
    print("  16/16: matrix_generic")
    x = create_phantom_2d(64, seed=57)
    np.random.seed(999)
    A = np.random.randn(256, 64 * 64).astype(np.float32) / np.sqrt(64 * 64)
    y = forward_matrix(x, A)
    y = add_noise(y, photon_level=1000, read_noise=5.0)
    np.save(data_dir / "matrix_generic" / "ground_truth.npy", x)
    np.save(data_dir / "matrix_generic" / "measurements.npy", y)
    np.save(data_dir / "matrix_generic" / "forward_matrix.npy", A)

    print("\nDone! All example data generated.")


if __name__ == "__main__":
    generate_all_data()
