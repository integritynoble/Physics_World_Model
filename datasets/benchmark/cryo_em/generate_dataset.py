#!/usr/bin/env python3
"""Generate Cryo-Electron Microscopy (Cryo-EM) benchmark dataset.

Forward model (2D cryo-EM single-particle imaging):
    Y(k) = CTF(k) * P(k) + N(k)   (in Fourier domain)

where:
    P(k)   : Fourier transform of 2D projection of 3D density map
    CTF(k) : contrast transfer function
             CTF(k) = -sqrt(1-A^2)*sin(chi(k)) - A*cos(chi(k))
             chi(k) = pi*lambda*|k|^2*defocus - 0.5*pi*Cs*lambda^3*|k|^4
    N(k)   : additive Gaussian noise (very high noise, SNR ~ 0.01-0.1)
    lambda : electron wavelength (depends on accelerating voltage)

Mismatch parameters:
    defocus_um       : defocus value in micrometers (0.5-5.0 um)
    cs_mm            : spherical aberration in mm (1.0-3.0)
    amplitude_contrast : amplitude contrast ratio (0.05-0.15)
    ice_thickness_nm : vitreous ice thickness in nm (30-100)

Phantoms:
    Protein-like structures:
      - Sphere clusters (protein subunit complexes)
      - Cylindrical helices (filaments / helical assemblies)
      - Icosahedral shells (virus capsid-like)

Seeds: public=0, dev=10000, hidden=20000

CPU Baseline: CTF-corrected Wiener filter. Expected ~15-22 dB.

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate as nd_rotate, zoom as nd_zoom

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry ------------------------------------------------------------------

IMAGE_SIZE = 256
PIXEL_SIZE_A = 1.5   # Angstroms per pixel (typical cryo-EM)

# -- Electron optics constants ------------------------------------------------

ACCEL_VOLTAGE_KV = 300.0   # accelerating voltage
# Relativistic electron wavelength at 300 kV
ELECTRON_WAVELENGTH_A = 0.0197  # Angstroms (relativistic, 300 keV)

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "defocus_um":          {"min": 1.0,  "max": 3.0,  "unit": "um"},
        "cs_mm":               {"min": 2.0,  "max": 2.2,  "unit": "mm"},
        "amplitude_contrast":  {"min": 0.07, "max": 0.10, "unit": ""},
        "ice_thickness_nm":    {"min": 40.0, "max": 60.0, "unit": "nm"},
    },
    "dev": {
        "defocus_um":          {"min": 0.5,  "max": 4.0,  "unit": "um"},
        "cs_mm":               {"min": 1.5,  "max": 2.5,  "unit": "mm"},
        "amplitude_contrast":  {"min": 0.05, "max": 0.12, "unit": ""},
        "ice_thickness_nm":    {"min": 30.0, "max": 80.0, "unit": "nm"},
    },
    "hidden": {
        "defocus_um":          {"min": 0.5,  "max": 5.0,  "unit": "um"},
        "cs_mm":               {"min": 1.0,  "max": 3.0,  "unit": "mm"},
        "amplitude_contrast":  {"min": 0.05, "max": 0.15, "unit": ""},
        "ice_thickness_nm":    {"min": 30.0, "max": 100.0, "unit": "nm"},
    },
}


# -- CTF computation ----------------------------------------------------------

def compute_ctf(
    size: int,
    pixel_size_A: float,
    defocus_um: float,
    cs_mm: float,
    amplitude_contrast: float,
    wavelength_A: float = ELECTRON_WAVELENGTH_A,
) -> np.ndarray:
    """Compute the 2D Contrast Transfer Function.

    CTF(k) = -sqrt(1-A^2)*sin(chi(k)) - A*cos(chi(k))
    chi(k) = pi*lambda*defocus*|k|^2 - 0.5*pi*Cs*lambda^3*|k|^4

    Args:
        size: image size in pixels
        pixel_size_A: pixel size in Angstroms
        defocus_um: defocus in micrometers (positive = underfocus)
        cs_mm: spherical aberration in mm
        amplitude_contrast: amplitude contrast ratio (0-1)
        wavelength_A: electron wavelength in Angstroms

    Returns:
        ctf: (size, size) float64 CTF values in Fourier space
    """
    # Convert units
    defocus_A = defocus_um * 1e4     # um -> Angstroms
    cs_A = cs_mm * 1e7               # mm -> Angstroms

    # Frequency grid (1/Angstrom)
    freq_1d = np.fft.fftfreq(size, d=pixel_size_A)
    kx, ky = np.meshgrid(freq_1d, freq_1d)
    k2 = kx**2 + ky**2  # |k|^2

    # Phase shift chi(k)
    chi = (np.pi * wavelength_A * defocus_A * k2
           - 0.5 * np.pi * cs_A * wavelength_A**3 * k2**2)

    # CTF with amplitude contrast
    w = amplitude_contrast
    ctf = -np.sqrt(1.0 - w**2) * np.sin(chi) - w * np.cos(chi)

    return ctf


# -- 3D phantom generators (simple protein-like structures) -------------------

def _sphere_3d(
    grid_size: int, cx: float, cy: float, cz: float,
    radius: float, density: float,
) -> np.ndarray:
    """Create a 3D sphere mask with given density."""
    coords = np.linspace(-1.0, 1.0, grid_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
    dist2 = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    vol[dist2 <= radius**2] = density
    return vol


def _cylinder_3d(
    grid_size: int, cx: float, cy: float,
    radius: float, density: float,
    axis: str = 'z',
) -> np.ndarray:
    """Create a 3D cylinder along the specified axis."""
    coords = np.linspace(-1.0, 1.0, grid_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    if axis == 'z':
        dist2 = (xx - cx)**2 + (yy - cy)**2
    elif axis == 'x':
        dist2 = (yy - cy)**2 + (zz - cx)**2
    else:  # y
        dist2 = (xx - cx)**2 + (zz - cy)**2
    vol[dist2 <= radius**2] = density
    return vol


def _icosahedron_shell_3d(
    grid_size: int, center: float, radius: float,
    shell_thickness: float, density: float,
) -> np.ndarray:
    """Create a 3D icosahedral shell (sphere shell approximation)."""
    coords = np.linspace(-1.0, 1.0, grid_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
    dist = np.sqrt((xx - center)**2 + (yy - center)**2 + (zz - center)**2)
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    mask = (dist >= radius - shell_thickness) & (dist <= radius + shell_thickness)
    vol[mask] = density

    # Add icosahedral-like surface bumps using spherical harmonics approximation
    # Use 12 evenly-distributed points on the shell (icosahedron vertices)
    golden_ratio = (1 + np.sqrt(5)) / 2
    ico_vertices = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            ico_vertices.append((0, s1 * 1.0, s2 * golden_ratio))
            ico_vertices.append((s1 * 1.0, s2 * golden_ratio, 0))
            ico_vertices.append((s1 * golden_ratio, 0, s2 * 1.0))

    # Normalize vertices to unit sphere and scale to shell radius
    for vx, vy, vz in ico_vertices:
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        px = center + radius * vx / norm
        py = center + radius * vy / norm
        pz = center + radius * vz / norm
        bump_dist = np.sqrt((xx - px)**2 + (yy - py)**2 + (zz - pz)**2)
        bump_mask = bump_dist <= shell_thickness * 1.5
        vol[bump_mask] = density * 1.3

    return vol


def make_sphere_cluster_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate a protein-like structure: cluster of spheres (subunit complex).

    Returns:
        projection: (H, W) float64 [0, ~1]  -- 2D projection of 3D density
        name: scene name
    """
    rng = np.random.default_rng(seed)
    grid_3d = 64  # 3D grid size (lower res for speed, then interpolate)

    vol = np.zeros((grid_3d, grid_3d, grid_3d), dtype=np.float64)

    # Central subunit
    vol += _sphere_3d(grid_3d, 0.0, 0.0, 0.0,
                      radius=0.15 + variant * 0.005,
                      density=1.0)

    # Surrounding subunits (4-8 subunits)
    n_subunits = rng.integers(4, 9)
    for i in range(n_subunits):
        angle = 2 * np.pi * i / n_subunits + rng.uniform(-0.2, 0.2)
        dist = 0.30 + rng.uniform(-0.05, 0.05)
        z_off = rng.uniform(-0.15, 0.15)
        cx = dist * np.cos(angle)
        cy = dist * np.sin(angle)
        r = 0.08 + rng.uniform(-0.02, 0.02)
        d = 0.7 + rng.uniform(-0.1, 0.1)
        vol += _sphere_3d(grid_3d, cx, cy, z_off, radius=r, density=d)

    # Optional secondary ring
    if rng.random() < 0.4:
        n_sec = rng.integers(3, 6)
        for i in range(n_sec):
            angle = 2 * np.pi * i / n_sec + rng.uniform(-0.3, 0.3)
            dist = 0.18 + rng.uniform(-0.03, 0.03)
            z_off = rng.uniform(-0.05, 0.05) + 0.20
            cx = dist * np.cos(angle)
            cy = dist * np.sin(angle)
            r = 0.06 + rng.uniform(-0.01, 0.01)
            vol += _sphere_3d(grid_3d, cx, cy, z_off, radius=r, density=0.5)

    # Random 3D rotation for variety
    rot_angle_1 = rng.uniform(0, 360)
    rot_angle_2 = rng.uniform(0, 360)
    vol = nd_rotate(vol, rot_angle_1, axes=(0, 1), reshape=False,
                    order=1, mode='constant', cval=0.0)
    vol = nd_rotate(vol, rot_angle_2, axes=(0, 2), reshape=False,
                    order=1, mode='constant', cval=0.0)

    # Project along z-axis (sum)
    projection = vol.sum(axis=0)

    # Resize to target resolution
    projection = _resize_2d(projection, H, W)

    # Apply mild smoothing (cryo-EM has limited resolution)
    projection = gaussian_filter(projection, sigma=1.2)

    # Normalize to [0, 1]
    if projection.max() > 0:
        projection /= projection.max()

    return projection, f"sphere_cluster_{variant:02d}"


def make_helix_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate a helical protein assembly (like actin filament or microtubule).

    Returns:
        projection: (H, W) float64 [0, ~1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    grid_3d = 64

    vol = np.zeros((grid_3d, grid_3d, grid_3d), dtype=np.float64)

    # Helical backbone
    n_turns = rng.uniform(1.5, 3.5)
    n_subunits_per_turn = rng.integers(8, 16)
    helix_radius = 0.15 + variant * 0.005
    subunit_radius = 0.05 + rng.uniform(-0.01, 0.01)
    pitch = 0.30 + rng.uniform(-0.05, 0.05)

    total_subunits = int(n_turns * n_subunits_per_turn)
    for i in range(total_subunits):
        t = i / n_subunits_per_turn  # turn number
        angle = 2 * np.pi * i / n_subunits_per_turn
        cx = helix_radius * np.cos(angle)
        cy = helix_radius * np.sin(angle)
        cz = -0.5 + t * pitch  # along helix axis

        if abs(cz) < 0.9:  # keep within grid
            density = 0.6 + rng.uniform(-0.1, 0.1)
            vol += _sphere_3d(grid_3d, cx, cy, cz,
                              radius=subunit_radius, density=density)

    # Central hollow core (for microtubule-like structures)
    if rng.random() < 0.5:
        vol += _cylinder_3d(grid_3d, 0.0, 0.0,
                            radius=helix_radius * 0.4,
                            density=-0.3, axis='z')
        vol = np.maximum(vol, 0.0)

    # Random rotation
    rot_angle = rng.uniform(0, 360)
    tilt_angle = rng.uniform(-30, 30)
    vol = nd_rotate(vol, rot_angle, axes=(0, 1), reshape=False,
                    order=1, mode='constant', cval=0.0)
    vol = nd_rotate(vol, tilt_angle, axes=(0, 2), reshape=False,
                    order=1, mode='constant', cval=0.0)

    projection = vol.sum(axis=0)
    projection = _resize_2d(projection, H, W)
    projection = gaussian_filter(projection, sigma=1.0)

    if projection.max() > 0:
        projection /= projection.max()

    return projection, f"helix_{variant:02d}"


def make_icosahedral_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate an icosahedral shell (virus capsid-like particle).

    Returns:
        projection: (H, W) float64 [0, ~1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    grid_3d = 64

    # Outer shell
    shell_r = 0.30 + variant * 0.005
    shell_thick = 0.04 + rng.uniform(-0.01, 0.01)
    vol = _icosahedron_shell_3d(
        grid_3d, center=0.0, radius=shell_r,
        shell_thickness=shell_thick, density=1.0,
    )

    # Inner content (RNA/DNA-like density)
    if rng.random() < 0.6:
        inner_r = shell_r * 0.6
        inner_density = rng.uniform(0.2, 0.5)
        vol += _sphere_3d(grid_3d, 0.0, 0.0, 0.0,
                          radius=inner_r, density=inner_density)

    # Optional decoration (surface proteins / spikes)
    if rng.random() < 0.5:
        n_spikes = rng.integers(12, 30)
        for _ in range(n_spikes):
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            sx = (shell_r + shell_thick) * np.sin(theta) * np.cos(phi)
            sy = (shell_r + shell_thick) * np.sin(theta) * np.sin(phi)
            sz = (shell_r + shell_thick) * np.cos(theta)
            spike_r = rng.uniform(0.02, 0.04)
            vol += _sphere_3d(grid_3d, sx, sy, sz,
                              radius=spike_r, density=0.8)

    # Random rotation
    rot1 = rng.uniform(0, 360)
    rot2 = rng.uniform(0, 360)
    vol = nd_rotate(vol, rot1, axes=(0, 1), reshape=False,
                    order=1, mode='constant', cval=0.0)
    vol = nd_rotate(vol, rot2, axes=(1, 2), reshape=False,
                    order=1, mode='constant', cval=0.0)

    projection = vol.sum(axis=0)
    projection = _resize_2d(projection, H, W)
    projection = gaussian_filter(projection, sigma=1.5)

    if projection.max() > 0:
        projection /= projection.max()

    return projection, f"icosahedral_{variant:02d}"


def _resize_2d(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2D array to (H, W) using scipy zoom."""
    h, w = arr.shape
    zoom_y = H / h
    zoom_x = W / w
    return nd_zoom(arr, (zoom_y, zoom_x), order=1)


# -- Phantom diversity pools per tier -----------------------------------------

PHANTOM_GENERATORS = [
    make_sphere_cluster_phantom,
    make_helix_phantom,
    make_icosahedral_phantom,
]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, str]]:
    """Generate diverse public-tier phantoms:
    4 sphere clusters + 4 helices + 4 icosahedral."""
    phantoms = []
    for i in range(4):
        proj, name = make_sphere_cluster_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i)
        phantoms.append((proj, name))
    for i in range(4):
        proj, name = make_helix_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=200 + i, variant=i)
        phantoms.append((proj, name))
    for i in range(4):
        proj, name = make_icosahedral_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=300 + i, variant=i)
        phantoms.append((proj, name))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(15000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        proj, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=10500 + i, variant=i)
        # Augment: rotation + flip
        angle = float(rng.uniform(15, 345))
        proj = nd_rotate(proj, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.5:
            proj = np.fliplr(proj)
        if rng.random() < 0.3:
            proj = np.flipud(proj)
        # Mild zoom
        zoom_f = float(rng.uniform(0.85, 1.15))
        if zoom_f != 1.0:
            proj = _zoom_crop(proj, zoom_f, IMAGE_SIZE)
        proj = np.clip(proj, 0.0, None)
        if proj.max() > 0:
            proj /= proj.max()
        phantoms.append((proj, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """Generate hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(25000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        proj, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=20500 + i, variant=i + 10)

        # Aggressive augmentation
        angle = float(rng.uniform(20, 340))
        proj = nd_rotate(proj, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.7:
            proj = np.fliplr(proj)
        if rng.random() < 0.5:
            proj = np.flipud(proj)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.70, 1.30))
        proj = _zoom_crop(proj, zoom_f, IMAGE_SIZE)

        # Add subtle sub-structures (hard to reconstruct)
        n_micro = rng.integers(3, 8)
        for _ in range(n_micro):
            cy = rng.integers(60, IMAGE_SIZE - 60)
            cx = rng.integers(60, IMAGE_SIZE - 60)
            r = rng.integers(2, 5)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0 = max(0, cy - r)
            y1 = min(IMAGE_SIZE, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(IMAGE_SIZE, cx + r + 1)
            c_y0, c_y1 = r - (cy - y0), r + (y1 - cy)
            c_x0, c_x1 = r - (cx - x0), r + (x1 - cx)
            if proj[y0:y1, x0:x1].mean() > 0.05:
                intensity = rng.uniform(0.3, 0.8)
                proj[y0:y1, x0:x1] = np.maximum(
                    proj[y0:y1, x0:x1],
                    circle[c_y0:c_y1, c_x0:c_x1] * intensity,
                )

        proj = np.clip(proj, 0.0, None)
        if proj.max() > 0:
            proj /= proj.max()
        phantoms.append((proj, f"hidden_{name}"))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    zoomed = nd_zoom(arr, zoom_f, order=1)
    H, W = zoomed.shape
    if H >= size and W >= size:
        y0 = (H - size) // 2
        x0 = (W - size) // 2
        return zoomed[y0:y0 + size, x0:x0 + size]
    else:
        out = np.zeros((size, size), dtype=arr.dtype)
        y0 = (size - H) // 2
        x0 = (size - W) // 2
        out[y0:y0 + H, x0:x0 + W] = zoomed
        return out


# -- Cryo-EM Forward Model ----------------------------------------------------

def cryo_em_forward_model(
    projection: np.ndarray,
    defocus_um: float,
    cs_mm: float,
    amplitude_contrast: float,
    ice_thickness_nm: float,
    rng: np.random.Generator,
) -> dict:
    """Apply full cryo-EM forward model.

    Y(k) = CTF(k) * P(k) + N(k)

    The noise level is controlled by ice_thickness: thicker ice means more
    background scattering and thus lower SNR.

    Args:
        projection: (H, W) 2D projection of 3D density [0, 1]
        defocus_um: defocus in micrometers
        cs_mm: spherical aberration in mm
        amplitude_contrast: amplitude contrast ratio (0-1)
        ice_thickness_nm: ice thickness in nm (controls SNR)
        rng: random number generator

    Returns:
        dict with measurement, ctf, ideal_image, etc.
    """
    H, W = projection.shape

    # 1. Compute CTF
    ctf = compute_ctf(
        size=H,
        pixel_size_A=PIXEL_SIZE_A,
        defocus_um=defocus_um,
        cs_mm=cs_mm,
        amplitude_contrast=amplitude_contrast,
    )

    # 2. Apply CTF in Fourier domain
    P_fourier = np.fft.fft2(projection.astype(np.float64))
    Y_ideal = ctf * P_fourier   # CTF-modulated image

    # 3. Determine noise level from ice thickness
    # Thicker ice -> more background scattering -> lower SNR
    # Typical cryo-EM SNR is extremely low (0.01-0.1)
    # ice_thickness_nm ranges from 30-100 nm
    # SNR ~ 1 / (ice_thickness * constant)
    signal_power = np.mean(np.abs(Y_ideal)**2)
    # Target SNR: very low (0.01 to 0.15 depending on ice)
    target_snr = 0.15 * np.exp(-0.03 * ice_thickness_nm)
    target_snr = max(target_snr, 0.005)  # floor

    noise_power = signal_power / (target_snr + 1e-12)
    noise_std = np.sqrt(noise_power / 2.0)  # for complex noise

    # 4. Add complex Gaussian noise
    noise_fourier = (rng.normal(0, noise_std, (H, W))
                     + 1j * rng.normal(0, noise_std, (H, W)))
    Y_noisy = Y_ideal + noise_fourier

    # 5. Convert back to real space
    image_ideal = np.real(np.fft.ifft2(Y_ideal))
    image_noisy = np.real(np.fft.ifft2(Y_noisy))

    # Compute achieved SNR
    signal_var = np.var(image_ideal)
    noise_var = np.var(image_noisy - image_ideal)
    achieved_snr = signal_var / (noise_var + 1e-12)

    return {
        "image_ideal": image_ideal.astype(np.float32),
        "image_noisy": image_noisy.astype(np.float32),
        "ctf": ctf.astype(np.float32),
        "achieved_snr": float(achieved_snr),
        "target_snr": float(target_snr),
        "noise_std": float(noise_std),
    }


# -- Wiener filter baseline reconstruction ------------------------------------

def wiener_ctf_correction(
    image_noisy: np.ndarray,
    ctf: np.ndarray,
    wiener_param: float = 0.1,
) -> np.ndarray:
    """CTF-corrected Wiener filter reconstruction.

    recon(k) = CTF*(k) * Y(k) / (|CTF(k)|^2 + epsilon)

    This is the standard baseline for cryo-EM: a frequency-domain
    deconvolution that inverts the CTF with Wiener regularization.

    Args:
        image_noisy: (H, W) measured image
        ctf: (H, W) CTF in Fourier domain
        wiener_param: regularization strength (noise-to-signal ratio estimate)

    Returns:
        recon: (H, W) float64 reconstructed projection
    """
    Y = np.fft.fft2(image_noisy.astype(np.float64))
    ctf64 = ctf.astype(np.float64)

    # Wiener filter: CTF* / (|CTF|^2 + epsilon)
    ctf_conj = np.conj(ctf64)
    ctf_power = np.abs(ctf64)**2
    wiener_filter = ctf_conj / (ctf_power + wiener_param)

    recon_fourier = wiener_filter * Y
    recon = np.real(np.fft.ifft2(recon_fourier))

    return recon


# -- Metrics -------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64))**2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range**2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Structural Similarity Index (simplified global SSIM)."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# -- Image helpers -------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip:
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, img_ideal, img_noisy, recon, path: Path) -> None:
    """4-panel overview: GT | ideal image | noisy image | Wiener recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(img_ideal)
    ov[:, 2 * tw:3 * tw] = _r(img_noisy)
    ov[:, 3 * tw:4 * tw] = _r(recon)
    _save_png(ov, path)


# -- Tier generation -----------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the cryo-EM benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"cryo_em_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Cryo-EM benchmark -- {tier} tier "
            f"(CTF modulation + extreme Gaussian noise)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_A": PIXEL_SIZE_A,
            "accelerating_voltage_kV": ACCEL_VOLTAGE_KV,
            "electron_wavelength_A": ELECTRON_WAVELENGTH_A,
        })
        f.attrs["forward_model"] = (
            "Y(k) = CTF(k) * P(k) + N(k)  "
            "(Fourier domain: CTF-modulated projection + Gaussian noise)"
        )

        for idx, (projection, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = cryo_em_forward_model(
                projection,
                defocus_um=mis["defocus_um"],
                cs_mm=mis["cs_mm"],
                amplitude_contrast=mis["amplitude_contrast"],
                ice_thickness_nm=mis["ice_thickness_nm"],
                rng=rng,
            )

            image_ideal = result["image_ideal"]
            image_noisy = result["image_noisy"]
            ctf = result["ctf"]

            # Wiener filter baseline reconstruction
            # Try several Wiener parameters and pick the best
            best_psnr = -np.inf
            best_recon = None
            for wp in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                recon = wiener_ctf_correction(image_noisy, ctf,
                                              wiener_param=wp)
                p = compute_psnr(projection, recon)
                if p > best_psnr:
                    best_psnr = p
                    best_recon = recon

            recon_wiener = best_recon.astype(np.float32)
            psnr = compute_psnr(projection, recon_wiener)
            ssim = compute_ssim(projection, recon_wiener)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # H_ideal: store CTF as the ideal forward operator
            # The measurement model is y = CTF * FT(x) + noise
            # H_ideal here is the CTF (frequency-domain operator)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=projection.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=image_noisy.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("H_ideal", data=ctf.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("image_ideal",
                               data=image_ideal.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("reconstruction_wiener",
                               data=recon_wiener,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(projection.shape),
                "pixel_size_A": PIXEL_SIZE_A,
                "achieved_snr": result["achieved_snr"],
                "target_snr": result["target_snr"],
                "psnr_wiener": float(psnr),
                "ssim_wiener": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(projection, sample_dir / "ground_truth.png")
            _save_png(image_ideal, sample_dir / "image_ideal.png")
            _save_png(image_noisy, sample_dir / "image_noisy.png",
                      percentile_clip=True)
            _save_png(recon_wiener, sample_dir / "reconstruction_wiener.png",
                      percentile_clip=True)
            _save_overview(projection, image_ideal, image_noisy, recon_wiener,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_wiener": psnr,
                    "ssim_wiener": ssim,
                    "snr": result["achieved_snr"],
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"defocus={mis['defocus_um']:.2f} um  "
                  f"Cs={mis['cs_mm']:.2f} mm  "
                  f"A={mis['amplitude_contrast']:.3f}  "
                  f"ice={mis['ice_thickness_nm']:.0f} nm  "
                  f"SNR={result['achieved_snr']:.4f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# -- Gallery image generation --------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png, recon_III.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "cryo_em")

    h5_path = BENCHMARK_DIR / "public" / "cryo_em_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: sphere_cluster(0), helix(4), icosahedral(8),
    # sphere variant(3)
    gallery_sample_indices = [0, 4, 8, 3]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_sample_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found in HDF5, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            img_ideal = grp["image_ideal"][:]
            img_noisy = grp["y"][:]
            recon_wiener = grp["reconstruction_wiener"][:]
            ctf = grp["H_ideal"][:]

            # gt.png -- ground truth projection
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy micrograph
            _save_png(img_noisy, scene_dir / "measurement_I.png",
                      percentile_clip=True)

            # measurement_II.png -- power spectrum (shows CTF rings)
            power_spectrum = np.log1p(
                np.abs(np.fft.fftshift(np.fft.fft2(img_noisy)))**2)
            _save_png(power_spectrum, scene_dir / "measurement_II.png")

            # recon_I.png -- Wiener reconstruction
            _save_png(recon_wiener, scene_dir / "recon_I.png",
                      percentile_clip=True)

            # recon_II.png -- CTF visualization
            ctf_vis = np.fft.fftshift(ctf)
            _save_png(ctf_vis, scene_dir / "recon_II.png")

            # recon_III.png -- difference image |GT - Wiener|
            diff = np.abs(x_true - recon_wiener)
            _save_png(diff, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} images saved "
                  f"to {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print("Cryo-EM Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {IMAGE_SIZE}x{IMAGE_SIZE} images, "
          f"pixel={PIXEL_SIZE_A} A/px, {ACCEL_VOLTAGE_KV} kV\n")

    # -- Public tier (12 samples) -------------------------------------------
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=0)

    # -- Dev tier (20 samples) ----------------------------------------------
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # -- Hidden tier (20 samples) -------------------------------------------
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # -- Gallery images -----------------------------------------------------
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Cryo-EM benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
