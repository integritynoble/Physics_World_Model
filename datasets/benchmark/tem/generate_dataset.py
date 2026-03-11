#!/usr/bin/env python3
"""Generate Transmission Electron Microscopy (TEM) benchmark dataset.

Forward model:
    y = Poisson(I0 * exp(-mu*t*x) * |F^-1(CTF * F(exit_wave))|^2) + readout_noise

where:
    x            : specimen projected potential / density map [0, 1]
    mu*t         : mass-thickness contrast parameter
    exit_wave    : complex exit wavefunction after specimen
    CTF(k)       : contrast transfer function
                   CTF(k) = -sqrt(1-A^2)*sin(chi(k)) - A*cos(chi(k))
                   chi(k) = pi*lambda*|k|^2*defocus - 0.5*pi*Cs*lambda^3*|k|^4
    I0           : incident beam intensity (electrons/pixel)
    readout_noise: Gaussian detector noise

Mismatch parameters:
    defocus_nm      : objective lens defocus
    cs_mm           : spherical aberration
    specimen_thickness_nm : specimen thickness
    beam_coherence  : partial coherence envelope factor

Phantoms:
    Nanostructured materials:
      - Crystal lattice fringes (periodic atomic planes)
      - Nanoparticle distributions (metallic NP assemblies)
      - Layered thin films (multilayer heterostructures)
      - Biological sections with staining contrast

Seeds: public=0, dev=10000, hidden=20000

CPU Baseline: CTF correction + Wiener filter. Expected ~18-25 dB.

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
PIXEL_SIZE_NM = 0.05   # nm per pixel (0.5 Angstrom, HRTEM scale)

# -- Electron optics constants ------------------------------------------------

ACCEL_VOLTAGE_KV = 300.0
# Relativistic electron wavelength at 300 kV in nm
ELECTRON_WAVELENGTH_NM = 0.00197  # ~0.0197 Angstroms = 0.00197 nm

# -- Beam parameters -----------------------------------------------------------

I0_DEFAULT = 5000.0          # electrons per pixel (typical for conventional TEM)
READOUT_NOISE_STD = 5.0      # detector readout noise (electrons)

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "defocus_nm":             {"min": -100.0, "max": -30.0,  "unit": "nm"},
        "cs_mm":                  {"min": 0.8,    "max": 1.5,    "unit": "mm"},
        "specimen_thickness_nm":  {"min": 20.0,   "max": 50.0,   "unit": "nm"},
        "beam_coherence":         {"min": 0.85,   "max": 0.95,   "unit": ""},
    },
    "dev": {
        "defocus_nm":             {"min": -150.0, "max": -20.0,  "unit": "nm"},
        "cs_mm":                  {"min": 0.5,    "max": 2.0,    "unit": "mm"},
        "specimen_thickness_nm":  {"min": 15.0,   "max": 70.0,   "unit": "nm"},
        "beam_coherence":         {"min": 0.75,   "max": 0.95,   "unit": ""},
    },
    "hidden": {
        "defocus_nm":             {"min": -200.0, "max": -10.0,  "unit": "nm"},
        "cs_mm":                  {"min": 0.001,  "max": 2.5,    "unit": "mm"},
        "specimen_thickness_nm":  {"min": 10.0,   "max": 100.0,  "unit": "nm"},
        "beam_coherence":         {"min": 0.60,   "max": 0.98,   "unit": ""},
    },
}


# -- CTF computation ----------------------------------------------------------

def compute_ctf(
    size: int,
    pixel_size_nm: float,
    defocus_nm: float,
    cs_mm: float,
    beam_coherence: float,
    wavelength_nm: float = ELECTRON_WAVELENGTH_NM,
) -> np.ndarray:
    """Compute the 2D Contrast Transfer Function for TEM.

    CTF(k) = -sqrt(1-A^2)*sin(chi(k)) - A*cos(chi(k))
    chi(k) = pi*lambda*defocus*|k|^2 - 0.5*pi*Cs*lambda^3*|k|^4

    Includes a partial coherence envelope:
        E(k) = exp(-0.5*(pi*sigma_d*lambda*|k|^2)^2)

    Args:
        size: image size in pixels
        pixel_size_nm: pixel size in nm
        defocus_nm: defocus in nm (negative = underfocus for convention)
        cs_mm: spherical aberration in mm
        beam_coherence: coherence factor (0-1), controls envelope width
        wavelength_nm: electron wavelength in nm

    Returns:
        ctf: (size, size) float64 CTF values in Fourier space
    """
    # Convert units to nm
    defocus = defocus_nm           # already nm
    cs_nm = cs_mm * 1e6            # mm -> nm

    # Frequency grid (1/nm)
    freq_1d = np.fft.fftfreq(size, d=pixel_size_nm)
    kx, ky = np.meshgrid(freq_1d, freq_1d)
    k2 = kx**2 + ky**2  # |k|^2

    # Phase shift chi(k)
    lam = wavelength_nm
    chi = (np.pi * lam * defocus * k2
           - 0.5 * np.pi * cs_nm * lam**3 * k2**2)

    # Standard amplitude contrast ratio for TEM (~0.07-0.10)
    A = 0.07
    ctf = -np.sqrt(1.0 - A**2) * np.sin(chi) - A * np.cos(chi)

    # Partial coherence envelope (spatial)
    # sigma_d controls the spread; lower coherence = faster decay
    sigma_d = (1.0 - beam_coherence) * 100.0 + 1.0  # nm, defocus spread
    envelope = np.exp(-0.5 * (np.pi * sigma_d * lam * k2)**2)
    ctf *= envelope

    # Aperture cutoff at ~2/3 Nyquist
    k_nyquist = 1.0 / (2.0 * pixel_size_nm)
    k_max = 0.67 * k_nyquist
    aperture = np.sqrt(k2) <= k_max
    ctf *= aperture

    return ctf


# -- Phantom generators -------------------------------------------------------

def _make_coordinate_grid(size: int):
    """Create normalized coordinate grids [-1, 1]."""
    coords = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    return yy, xx


def make_crystal_lattice_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate crystal lattice fringes (periodic atomic planes).

    Simulates HRTEM lattice images of crystalline materials like Si, GaAs,
    or Au nanocrystals showing periodic atomic plane contrast.

    Returns:
        phantom: (H, W) float64 [0, 1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    yy, xx = _make_coordinate_grid(H)
    phantom = np.zeros((H, W), dtype=np.float64)

    # Base lattice parameters
    n_orientations = rng.integers(1, 4)  # 1-3 lattice orientations (grain boundaries)

    for g in range(n_orientations):
        # Lattice spacing (in normalized coords, ~3-10 px period)
        d_spacing = rng.uniform(0.02, 0.08)
        angle = rng.uniform(0, np.pi) + variant * 0.1

        # Rotated coordinates
        kx = np.cos(angle) / d_spacing
        ky = np.sin(angle) / d_spacing

        # Sinusoidal lattice fringes with harmonics
        lattice = 0.5 * (1.0 + np.cos(2.0 * np.pi * (kx * xx + ky * yy)))
        # Add second harmonic for more realistic atomic contrast
        lattice += 0.15 * np.cos(4.0 * np.pi * (kx * xx + ky * yy))

        # Grain region mask (circular or polygonal)
        cx = rng.uniform(-0.3, 0.3)
        cy = rng.uniform(-0.3, 0.3)
        grain_r = rng.uniform(0.3, 0.7)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        if n_orientations == 1:
            # Single crystal: fill most of FOV
            mask = dist < 0.9
        else:
            mask = dist < grain_r

        # Smooth grain boundary
        boundary_width = 0.03
        mask_soft = np.clip((grain_r - dist) / boundary_width, 0, 1)
        phantom += lattice * mask_soft * rng.uniform(0.3, 0.8)

    # Add amorphous background (support film / vacuum)
    bg_noise = gaussian_filter(rng.uniform(0, 0.15, (H, W)), sigma=5)
    phantom += bg_noise

    # Add some point defects / vacancies
    n_defects = rng.integers(3, 15)
    for _ in range(n_defects):
        dy = rng.integers(20, H - 20)
        dx = rng.integers(20, W - 20)
        r = rng.integers(1, 4)
        yy_l, xx_l = np.ogrid[-r:r + 1, -r:r + 1]
        circle = (yy_l**2 + xx_l**2 <= r**2).astype(np.float64)
        y0, y1 = max(0, dy - r), min(H, dy + r + 1)
        x0, x1 = max(0, dx - r), min(W, dx + r + 1)
        c_y0, c_y1 = r - (dy - y0), r + (y1 - dy)
        c_x0, c_x1 = r - (dx - x0), r + (x1 - dx)
        defect_val = rng.uniform(-0.3, 0.3)
        phantom[y0:y1, x0:x1] += circle[c_y0:c_y1, c_x0:c_x1] * defect_val

    phantom = np.clip(phantom, 0.0, None)
    phantom = gaussian_filter(phantom, sigma=0.8)
    if phantom.max() > 0:
        phantom /= phantom.max()

    return phantom, f"crystal_lattice_{variant:02d}"


def make_nanoparticle_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate nanoparticle distribution (metallic NP assemblies).

    Simulates TEM images of supported nanoparticle catalysts, quantum dots,
    or colloidal metallic nanoparticles on carbon support.

    Returns:
        phantom: (H, W) float64 [0, 1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    phantom = np.zeros((H, W), dtype=np.float64)

    # Carbon support film (amorphous background)
    support = gaussian_filter(rng.uniform(0.05, 0.20, (H, W)), sigma=8)
    # Add some larger-scale thickness variation
    support += 0.05 * gaussian_filter(rng.uniform(0, 1, (H, W)), sigma=30)
    phantom += support

    # Nanoparticle size distribution (log-normal is realistic)
    n_particles = rng.integers(15, 60) + variant * 3
    mean_radius = rng.uniform(4, 12)
    sigma_r = rng.uniform(0.2, 0.5)

    for _ in range(n_particles):
        r = max(2, int(rng.lognormal(np.log(mean_radius), sigma_r)))
        cy = rng.integers(r + 5, H - r - 5)
        cx = rng.integers(r + 5, W - r - 5)

        # NP projected density (approximately parabolic cross-section for sphere)
        yy_l, xx_l = np.ogrid[-r:r + 1, -r:r + 1]
        dist2 = (yy_l**2 + xx_l**2).astype(np.float64)
        r2 = float(r**2)
        # Projected thickness of sphere: ~ sqrt(R^2 - r^2)
        np_profile = np.sqrt(np.maximum(r2 - dist2, 0.0)) / r
        # Density depends on material (Au is very dense)
        density = rng.uniform(0.5, 1.0)
        np_profile *= density

        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        c_y0 = r - (cy - y0)
        c_y1 = c_y0 + (y1 - y0)
        c_x0 = r - (cx - x0)
        c_x1 = c_x0 + (x1 - x0)
        phantom[y0:y1, x0:x1] += np_profile[c_y0:c_y1, c_x0:c_x1]

    # Some particles may show internal lattice fringes
    # (simplified: add faint periodic modulation in larger particles)
    n_crystalline = rng.integers(2, min(8, n_particles // 3 + 1))
    for _ in range(n_crystalline):
        r = rng.integers(6, 14)
        cy = rng.integers(r + 10, H - r - 10)
        cx = rng.integers(r + 10, W - r - 10)
        angle = rng.uniform(0, np.pi)
        d_space = rng.uniform(0.015, 0.04)

        yy_l = np.arange(-r, r + 1).reshape(-1, 1)
        xx_l = np.arange(-r, r + 1).reshape(1, -1)
        dist2 = (yy_l**2 + xx_l**2).astype(np.float64)
        mask = dist2 <= r**2
        fringes = 0.1 * np.cos(
            2 * np.pi * (np.cos(angle) * xx_l + np.sin(angle) * yy_l)
            / (d_space * H)
        ) * mask
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        c_y0 = r - (cy - y0)
        c_y1 = c_y0 + (y1 - y0)
        c_x0 = r - (cx - x0)
        c_x1 = c_x0 + (x1 - x0)
        phantom[y0:y1, x0:x1] += fringes[c_y0:c_y1, c_x0:c_x1]

    phantom = np.clip(phantom, 0.0, None)
    phantom = gaussian_filter(phantom, sigma=0.5)
    if phantom.max() > 0:
        phantom /= phantom.max()

    return phantom, f"nanoparticle_{variant:02d}"


def make_layered_film_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate layered thin film heterostructure.

    Simulates cross-sectional TEM of multilayer thin films (e.g.,
    semiconductor heterostructures, optical coatings, battery electrodes).

    Returns:
        phantom: (H, W) float64 [0, 1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    phantom = np.zeros((H, W), dtype=np.float64)

    # Layer orientation (angle from horizontal)
    layer_angle = rng.uniform(-15, 15) + variant * 2.0
    angle_rad = np.radians(layer_angle)

    # Number of layers
    n_layers = rng.integers(5, 15)
    layer_thicknesses = rng.uniform(5, 30, size=n_layers)
    layer_densities = rng.uniform(0.2, 0.9, size=n_layers)

    # Coordinate system rotated by layer_angle
    yy, xx = _make_coordinate_grid(H)
    # Projected position along layer normal
    pos = np.cos(angle_rad) * yy + np.sin(angle_rad) * xx

    # Map position to pixel space
    pos_px = (pos + 1.0) / 2.0 * H  # [0, H]

    cumulative = 10.0  # start offset from edge
    for i in range(n_layers):
        thickness = layer_thicknesses[i]
        density = layer_densities[i]
        layer_start = cumulative
        layer_end = cumulative + thickness
        cumulative = layer_end

        if cumulative > H - 10:
            break

        # Smooth layer boundaries
        boundary = 1.5  # transition width in pixels
        mask = (
            0.5 * (1 + np.tanh((pos_px - layer_start) / boundary))
            * 0.5 * (1 + np.tanh((layer_end - pos_px) / boundary))
        )
        phantom += mask * density

        # Some layers have internal microstructure (columnar grains)
        if rng.random() < 0.3:
            freq = rng.uniform(0.05, 0.15)
            perp_angle = angle_rad + np.pi / 2
            perp_pos = np.cos(perp_angle) * yy + np.sin(perp_angle) * xx
            columnar = 0.05 * np.cos(2 * np.pi * freq * perp_pos * H)
            phantom += columnar * mask

    # Add interface roughness / interdiffusion
    roughness = gaussian_filter(rng.normal(0, 0.03, (H, W)), sigma=3)
    phantom += roughness

    # Substrate region
    substrate_mask = pos_px > cumulative
    phantom[substrate_mask] += rng.uniform(0.3, 0.5)

    phantom = np.clip(phantom, 0.0, None)
    phantom = gaussian_filter(phantom, sigma=0.7)
    if phantom.max() > 0:
        phantom /= phantom.max()

    return phantom, f"layered_film_{variant:02d}"


def make_biological_section_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> tuple[np.ndarray, str]:
    """Generate biological thin section with staining contrast.

    Simulates TEM of ultrathin biological sections stained with heavy metals
    (uranyl acetate, osmium tetroxide, lead citrate), showing cell
    ultrastructure: membranes, organelles, ribosomes.

    Returns:
        phantom: (H, W) float64 [0, 1]
        name: scene name
    """
    rng = np.random.default_rng(seed)
    phantom = np.zeros((H, W), dtype=np.float64)
    yy, xx = _make_coordinate_grid(H)

    # Cytoplasm background (lightly stained)
    cytoplasm = gaussian_filter(rng.uniform(0.15, 0.30, (H, W)), sigma=15)
    phantom += cytoplasm

    # Membranes (double bilayer lines)
    n_membranes = rng.integers(2, 6) + variant // 3
    for _ in range(n_membranes):
        # Curved membrane as parametric path
        t = np.linspace(0, 1, 200)
        cx_start = rng.uniform(-0.6, 0.6)
        cy_start = rng.uniform(-0.6, 0.6)
        cx_end = cx_start + rng.uniform(-0.5, 0.5)
        cy_end = cy_start + rng.uniform(-0.5, 0.5)
        # Bezier control point
        bx = rng.uniform(-0.8, 0.8)
        by = rng.uniform(-0.8, 0.8)
        curve_x = (1 - t)**2 * cx_start + 2 * (1 - t) * t * bx + t**2 * cx_end
        curve_y = (1 - t)**2 * cy_start + 2 * (1 - t) * t * by + t**2 * cy_end

        # Draw membrane as dark line (stained)
        membrane_width = rng.uniform(1.5, 3.0)  # in pixels
        for px, py in zip(curve_x, curve_y):
            dist = np.sqrt((xx - px)**2 + (yy - py)**2) * H / 2
            membrane_profile = np.exp(-dist**2 / (2 * membrane_width**2))
            phantom += membrane_profile * rng.uniform(0.3, 0.6)

    # Organelle-like structures (mitochondria, ER, vesicles)
    n_organelles = rng.integers(3, 8)
    for _ in range(n_organelles):
        org_type = rng.choice(["mito", "vesicle", "ribosome_cluster"])

        if org_type == "mito":
            # Elongated double-membrane organelle with cristae
            cx = rng.uniform(-0.5, 0.5)
            cy = rng.uniform(-0.5, 0.5)
            angle = rng.uniform(0, np.pi)
            length = rng.uniform(0.15, 0.35)
            width = rng.uniform(0.05, 0.12)

            # Rotated coordinates
            rx = np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)
            ry = -np.sin(angle) * (xx - cx) + np.cos(angle) * (yy - cy)

            # Outer membrane
            outer = np.exp(-(rx / length)**6 - (ry / width)**6)
            phantom += outer * 0.15

            # Inner membrane (slightly smaller)
            inner = np.exp(-(rx / (length * 0.85))**6 - (ry / (width * 0.7))**6)
            phantom += inner * 0.1

            # Cristae (internal folds)
            n_cristae = rng.integers(3, 8)
            for c in range(n_cristae):
                c_pos = -length * 0.7 + c * length * 1.4 / n_cristae
                c_mask = np.exp(-((rx - c_pos) / 0.01)**2) * inner
                phantom += c_mask * 0.2

        elif org_type == "vesicle":
            # Circular membrane-bound vesicle
            cx = rng.uniform(-0.6, 0.6)
            cy = rng.uniform(-0.6, 0.6)
            r = rng.uniform(0.03, 0.10)
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            # Membrane ring
            ring = np.exp(-((dist - r) / 0.008)**2) * 0.4
            phantom += ring
            # Interior (lighter or darker depending on content)
            interior = np.exp(-(dist / (r * 0.8))**4) * rng.uniform(0.05, 0.20)
            phantom += interior

        else:  # ribosome_cluster
            # Cluster of small dense particles
            cx = rng.uniform(-0.5, 0.5)
            cy = rng.uniform(-0.5, 0.5)
            n_ribosomes = rng.integers(5, 20)
            for _ in range(n_ribosomes):
                rx = cx + rng.normal(0, 0.03)
                ry = cy + rng.normal(0, 0.03)
                r = rng.uniform(0.005, 0.012)
                dist = np.sqrt((xx - rx)**2 + (yy - ry)**2)
                ribo = np.exp(-(dist / r)**2) * rng.uniform(0.2, 0.5)
                phantom += ribo

    # Add staining artifacts (uneven staining)
    stain_gradient = gaussian_filter(rng.uniform(0, 0.1, (H, W)), sigma=40)
    phantom += stain_gradient

    phantom = np.clip(phantom, 0.0, None)
    phantom = gaussian_filter(phantom, sigma=0.6)
    if phantom.max() > 0:
        phantom /= phantom.max()

    return phantom, f"bio_section_{variant:02d}"


# -- Phantom pool generators per tier ----------------------------------------

PHANTOM_GENERATORS = [
    make_crystal_lattice_phantom,
    make_nanoparticle_phantom,
    make_layered_film_phantom,
    make_biological_section_phantom,
]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, str]]:
    """Generate diverse public-tier phantoms:
    3 crystal lattice + 3 nanoparticle + 3 layered film + 3 bio section."""
    phantoms = []
    per_type = n // 4
    remainder = n - per_type * 4
    counts = [per_type] * 4
    for i in range(remainder):
        counts[i] += 1

    sample_idx = 0
    for gen_idx, gen_fn in enumerate(PHANTOM_GENERATORS):
        for v in range(counts[gen_idx]):
            proj, name = gen_fn(
                IMAGE_SIZE, IMAGE_SIZE, seed=100 + sample_idx, variant=v)
            phantoms.append((proj, name))
            sample_idx += 1
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(15000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 4]
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
        gen_fn = PHANTOM_GENERATORS[i % 4]
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
        n_micro = rng.integers(3, 10)
        for _ in range(n_micro):
            cy = rng.integers(30, IMAGE_SIZE - 30)
            cx = rng.integers(30, IMAGE_SIZE - 30)
            r = rng.integers(1, 4)
            yy_l, xx_l = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy_l**2 + xx_l**2 <= r**2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0 = r - (cy - y0)
            c_y1 = c_y0 + (y1 - y0)
            c_x0 = r - (cx - x0)
            c_x1 = c_x0 + (x1 - x0)
            if proj[y0:y1, x0:x1].mean() > 0.05:
                intensity = rng.uniform(0.2, 0.7)
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


# -- TEM Forward Model --------------------------------------------------------

def tem_forward_model(
    x_true: np.ndarray,
    defocus_nm: float,
    cs_mm: float,
    specimen_thickness_nm: float,
    beam_coherence: float,
    rng: np.random.Generator,
    I0: float = I0_DEFAULT,
    readout_std: float = READOUT_NOISE_STD,
) -> dict:
    """Apply the full TEM forward model.

    y = Poisson(I0 * exp(-mu*t*x) * |F^-1(CTF * F(exit_wave))|^2) + readout

    For reconstruction purposes we use the linearized (weak-phase-object)
    contrast model:
        image_contrast(k) ~ CTF(k) * sigma(k)
    where sigma is the projected potential, so that Wiener deconvolution
    of the CTF is meaningful.

    The full nonlinear intensity is computed for the measurement y, while
    H_ideal stores the CTF for the linear inverse problem.

    Args:
        x_true: (H, W) specimen projected density [0, 1]
        defocus_nm: defocus in nm
        cs_mm: spherical aberration in mm
        specimen_thickness_nm: thickness in nm (controls absorption strength)
        beam_coherence: partial coherence factor (0-1)
        rng: random number generator
        I0: beam intensity (electrons/pixel)
        readout_std: readout noise standard deviation

    Returns:
        dict with measurement, ctf, H_ideal, etc.
    """
    H, W = x_true.shape

    # 1. Mass-thickness contrast: mu*t depends on specimen thickness
    mu = 0.03  # absorption coefficient (1/nm)
    mu_t = mu * specimen_thickness_nm  # dimensionless

    # 2. Compute CTF
    ctf = compute_ctf(
        size=H,
        pixel_size_nm=PIXEL_SIZE_NM,
        defocus_nm=defocus_nm,
        cs_mm=cs_mm,
        beam_coherence=beam_coherence,
    )

    # 3. Weak-phase-object (WPO) linearized contrast image
    # Under WPO approximation for thin specimens:
    #   I(r) = I0 * (1 - 2*sigma*CTF_imag)  approximately
    # where sigma ~ mu_t * x is the projected potential.
    # We compute: contrast = F^-1(CTF * F(x_true))
    # This gives the image contrast due to phase + absorption.
    X_fourier = np.fft.fft2(x_true.astype(np.float64))
    contrast_fourier = ctf * X_fourier
    contrast_image = np.real(np.fft.ifft2(contrast_fourier))

    # Scale contrast by mass-thickness
    contrast_scaled = mu_t * contrast_image

    # 4. Ideal intensity: I = I0 * (1 + contrast_scaled)
    # The contrast_scaled modulates around the mean intensity I0
    intensity_ideal = I0 * (1.0 + contrast_scaled)

    # Ensure non-negative before Poisson
    intensity_ideal = np.maximum(intensity_ideal, 1.0)

    # 5. Poisson noise (shot noise)
    intensity_poisson = rng.poisson(intensity_ideal).astype(np.float64)

    # 6. Readout noise (Gaussian)
    readout_noise = rng.normal(0, readout_std, (H, W))
    y = intensity_poisson + readout_noise

    # Compute achieved SNR (contrast SNR)
    contrast_signal = intensity_ideal - I0  # the contrast part
    noise = y - intensity_ideal
    signal_var = np.var(contrast_signal)
    noise_var = np.var(noise)
    achieved_snr = signal_var / (noise_var + 1e-12)

    return {
        "y": y.astype(np.float32),
        "intensity_ideal": intensity_ideal.astype(np.float32),
        "ctf": ctf.astype(np.float32),
        "achieved_snr": float(achieved_snr),
        "mu_t": float(mu_t),
        "I0": float(I0),
    }


# -- Wiener filter baseline reconstruction ------------------------------------

def wiener_ctf_correction(
    y: np.ndarray,
    ctf: np.ndarray,
    I0: float,
    mu_t: float,
    wiener_param: float = 0.1,
) -> np.ndarray:
    """CTF-corrected Wiener filter reconstruction for TEM.

    Under the WPO linearized model:
        y ~ I0 * (1 + mu_t * F^-1(CTF * F(x)))  + noise
    So:
        contrast = (y - I0) / (I0 * mu_t)
        F(contrast) ~ CTF * F(x)  + noise term
    Then Wiener deconvolution of CTF recovers x.

    Args:
        y: (H, W) measured TEM image (noisy)
        ctf: (H, W) CTF in Fourier domain
        I0: beam intensity
        mu_t: mass-thickness parameter
        wiener_param: regularization strength

    Returns:
        recon: (H, W) float64 reconstructed projected density [~0, ~1]
    """
    H, W = y.shape

    # Step 1: Extract contrast image from measurement
    # y = I0 * (1 + mu_t * contrast_image) + noise
    # => contrast_image ~ (y - I0) / (I0 * mu_t)
    scale = I0 * mu_t
    if scale < 1e-6:
        scale = 1.0
    contrast = (y.astype(np.float64) - I0) / scale

    # Step 2: Wiener deconvolution to undo CTF
    # F(contrast) ~ CTF * F(x) + noise
    C_fourier = np.fft.fft2(contrast)
    ctf64 = ctf.astype(np.float64)

    ctf_conj = np.conj(ctf64)
    ctf_power = np.abs(ctf64)**2
    wiener_filter = ctf_conj / (ctf_power + wiener_param)

    recon_fourier = wiener_filter * C_fourier
    x_recon = np.real(np.fft.ifft2(recon_fourier))

    # Step 3: Normalize to [0, 1]
    x_recon = np.clip(x_recon, 0.0, None)
    if x_recon.max() > 0:
        p99 = np.percentile(x_recon, 99)
        if p99 > 0:
            x_recon /= p99
    x_recon = np.clip(x_recon, 0.0, 1.0)

    return x_recon


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


def _save_overview(x_true, intensity_ideal, y_noisy, recon, path: Path) -> None:
    """4-panel overview: GT | ideal intensity | noisy measurement | Wiener recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(intensity_ideal)
    ov[:, 2 * tw:3 * tw] = _r(y_noisy)
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
    """Generate one tier of the TEM benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"tem_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM TEM benchmark -- {tier} tier "
            f"(Beer-Lambert absorption + CTF + Poisson noise)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "accelerating_voltage_kV": ACCEL_VOLTAGE_KV,
            "electron_wavelength_nm": ELECTRON_WAVELENGTH_NM,
        })
        f.attrs["forward_model"] = (
            "y = Poisson(I0 * exp(-mu*t*x) * |F^-1(CTF * F(exit_wave))|^2) "
            "+ readout_noise"
        )

        for idx, (phantom, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = tem_forward_model(
                phantom,
                defocus_nm=mis["defocus_nm"],
                cs_mm=mis["cs_mm"],
                specimen_thickness_nm=mis["specimen_thickness_nm"],
                beam_coherence=mis["beam_coherence"],
                rng=rng,
            )

            y = result["y"]
            intensity_ideal = result["intensity_ideal"]
            ctf = result["ctf"]
            mu_t = result["mu_t"]
            I0_used = result["I0"]

            # Wiener filter baseline -- try several parameters
            best_psnr = -np.inf
            best_recon = None
            for wp in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
                       0.1, 0.5, 1.0, 5.0, 10.0]:
                recon = wiener_ctf_correction(
                    y, ctf, I0=I0_used, mu_t=mu_t, wiener_param=wp)
                p = compute_psnr(phantom, recon)
                if p > best_psnr:
                    best_psnr = p
                    best_recon = recon

            recon_wiener = best_recon.astype(np.float32)
            psnr = compute_psnr(phantom, recon_wiener)
            ssim = compute_ssim(phantom, recon_wiener)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=phantom.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("H_ideal", data=ctf.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("intensity_ideal",
                               data=intensity_ideal.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("reconstruction_wiener",
                               data=recon_wiener,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(phantom.shape),
                "pixel_size_nm": PIXEL_SIZE_NM,
                "achieved_snr": result["achieved_snr"],
                "mu_t": mu_t,
                "psnr_wiener": float(psnr),
                "ssim_wiener": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save preview images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(phantom, sample_dir / "ground_truth.png")
            _save_png(intensity_ideal, sample_dir / "intensity_ideal.png")
            _save_png(y, sample_dir / "measurement.png",
                      percentile_clip=True)
            _save_png(recon_wiener, sample_dir / "reconstruction_wiener.png",
                      percentile_clip=True)
            _save_overview(phantom, intensity_ideal, y, recon_wiener,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_wiener": psnr,
                    "ssim_wiener": ssim,
                    "snr": result["achieved_snr"],
                    "mu_t": mu_t,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"defocus={mis['defocus_nm']:.1f} nm  "
                  f"Cs={mis['cs_mm']:.3f} mm  "
                  f"thickness={mis['specimen_thickness_nm']:.1f} nm  "
                  f"coherence={mis['beam_coherence']:.3f}  "
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
                    "benchmark_gallery" / "tem")

    h5_path = BENCHMARK_DIR / "public" / "tem_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: crystal(0), nanoparticle(3), layered(6), bio(9)
    gallery_sample_indices = [0, 3, 6, 9]

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
            intensity_ideal = grp["intensity_ideal"][:]
            y = grp["y"][:]
            recon_wiener = grp["reconstruction_wiener"][:]
            ctf = grp["H_ideal"][:]

            # gt.png -- ground truth projected density
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy TEM image
            _save_png(y, scene_dir / "measurement_I.png",
                      percentile_clip=True)

            # measurement_II.png -- power spectrum (shows CTF rings)
            power_spectrum = np.log1p(
                np.abs(np.fft.fftshift(np.fft.fft2(y)))**2)
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
    print("TEM Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {IMAGE_SIZE}x{IMAGE_SIZE} images, "
          f"pixel={PIXEL_SIZE_NM} nm/px, {ACCEL_VOLTAGE_KV} kV")
    print(f"Forward: y = Poisson(I0*exp(-mu*t*x)*|F^-1(CTF*F(psi))|^2) + readout")
    print()

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
    print("TEM benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
