#!/usr/bin/env python3
"""Generate the Ptychography benchmark dataset.

Forward model (scanning coherent diffractive imaging):
    For each scan position j:
        y_j = |F{ P(r - r_j) * O(r) }|^2 + noise

    P(r - r_j) : shifted complex probe function (Gaussian or structured)
    O(r)        : complex-valued object transmission function
    F{.}        : 2D Fourier transform (far-field diffraction)
    |.|^2       : intensity measurement (phase lost)

Mismatch parameters:
    probe_position_error : random offset in probe positions (pixels)
    probe_shape_error    : probe width perturbation factor
    detector_saturation  : saturation threshold for detector counts
    noise_level          : Poisson noise photon flux (lower = noisier)

Phantoms:
    Complex-valued thin specimens with both amplitude and phase:
    - circuit_pattern  : IC-like features (sharp edges, rectangular structures)
    - biological_cell  : cell-like structures (membranes, organelles)
    - crystalline      : periodic lattice with defects
    - mixed_media      : combination of sharp and smooth features

CPU Baseline:
    ePIE (extended Ptychographic Iterative Engine) - alternating projection.
    Expected: ~22-30 dB PSNR on amplitude recovery.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/ptychography
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry ----------------------------------------------------------------

IMAGE_SIZE = 256            # object pixel grid
PROBE_SIZE = 64             # probe support diameter in pixels
SCAN_STEP = 20              # step size between scan positions (pixels)
OVERLAP_RATIO = 0.6875      # 1 - 20/64 = 0.6875
DETECTOR_SIZE = 64          # diffraction pattern size per position
WAVELENGTH_NM = 0.15        # X-ray wavelength (hard X-ray, ~8 keV)
PIXEL_SIZE_NM = 10.0        # real-space pixel size

# -- Mismatch ranges per tier -----------------------------------------------

SPEC = {
    "public": {
        "probe_position_error": {"min": 0.0, "max": 1.5, "unit": "pixels"},
        "probe_shape_error":    {"min": 0.90, "max": 1.10, "unit": "factor"},
        "detector_saturation":  {"min": 50000, "max": 100000, "unit": "counts"},
        "noise_level":          {"min": 1e5, "max": 1e6, "unit": "photons"},
    },
    "dev": {
        "probe_position_error": {"min": 0.5, "max": 3.0, "unit": "pixels"},
        "probe_shape_error":    {"min": 0.85, "max": 1.15, "unit": "factor"},
        "detector_saturation":  {"min": 30000, "max": 80000, "unit": "counts"},
        "noise_level":          {"min": 5e4, "max": 5e5, "unit": "photons"},
    },
    "hidden": {
        "probe_position_error": {"min": 1.0, "max": 5.0, "unit": "pixels"},
        "probe_shape_error":    {"min": 0.80, "max": 1.25, "unit": "factor"},
        "detector_saturation":  {"min": 20000, "max": 60000, "unit": "counts"},
        "noise_level":          {"min": 1e4, "max": 2e5, "unit": "photons"},
    },
}


# -- Complex phantom generators ---------------------------------------------

def make_circuit_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """IC circuit pattern: sharp rectangular features with phase variation.

    Returns complex-valued (H, W) object transmission O(r).
    Amplitude in [0.3, 1.0], phase in [-pi/2, pi/2].
    """
    amplitude = np.ones((H, W), dtype=np.float64) * 0.8
    phase = np.zeros((H, W), dtype=np.float64)

    n_features = rng.integers(15, 35)
    for _ in range(n_features):
        feat_type = rng.choice(["rect", "line_h", "line_v", "pad"])
        if feat_type == "rect":
            y0 = rng.integers(10, H - 30)
            x0 = rng.integers(10, W - 30)
            rh = rng.integers(5, 40)
            rw = rng.integers(5, 40)
            y1, x1 = min(y0 + rh, H), min(x0 + rw, W)
            amp_val = rng.uniform(0.3, 0.95)
            phase_val = rng.uniform(-np.pi / 3, np.pi / 3)
            amplitude[y0:y1, x0:x1] = amp_val
            phase[y0:y1, x0:x1] = phase_val
        elif feat_type == "line_h":
            y0 = rng.integers(5, H - 5)
            x0 = rng.integers(0, W // 2)
            length = rng.integers(30, W - x0)
            thickness = rng.integers(1, 4)
            y1 = min(y0 + thickness, H)
            x1 = min(x0 + length, W)
            amplitude[y0:y1, x0:x1] = rng.uniform(0.4, 0.7)
            phase[y0:y1, x0:x1] = rng.uniform(-np.pi / 4, np.pi / 4)
        elif feat_type == "line_v":
            y0 = rng.integers(0, H // 2)
            x0 = rng.integers(5, W - 5)
            length = rng.integers(30, H - y0)
            thickness = rng.integers(1, 4)
            y1 = min(y0 + length, H)
            x1 = min(x0 + thickness, W)
            amplitude[y0:y1, x0:x1] = rng.uniform(0.4, 0.7)
            phase[y0:y1, x0:x1] = rng.uniform(-np.pi / 4, np.pi / 4)
        else:  # pad
            cy = rng.integers(20, H - 20)
            cx = rng.integers(20, W - 20)
            radius = rng.integers(3, 12)
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            amplitude[mask] = rng.uniform(0.5, 1.0)
            phase[mask] = rng.uniform(-np.pi / 2, np.pi / 2)

    # Slight smoothing for realism
    amplitude = gaussian_filter(amplitude, sigma=0.8)
    phase = gaussian_filter(phase, sigma=0.5)
    amplitude = np.clip(amplitude, 0.3, 1.0)

    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex128)


def make_biological_cell_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Biological cell: membrane boundary, nucleus, organelles.

    Returns complex-valued (H, W) object transmission.
    """
    amplitude = np.ones((H, W), dtype=np.float64) * 0.95
    phase = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.ogrid[:H, :W]

    # Cell boundary (large ellipse)
    cy, cx = H // 2 + rng.integers(-10, 10), W // 2 + rng.integers(-10, 10)
    a = rng.uniform(H * 0.30, H * 0.42)
    b = rng.uniform(W * 0.28, W * 0.38)
    angle = rng.uniform(0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    yr = (yy - cy) * cos_a - (xx - cx) * sin_a
    xr = (yy - cy) * sin_a + (xx - cx) * cos_a
    cell_mask = (yr / a) ** 2 + (xr / b) ** 2 <= 1.0
    amplitude[cell_mask] = rng.uniform(0.6, 0.85)
    phase[cell_mask] = rng.uniform(0.3, 0.8)

    # Nucleus (smaller ellipse inside cell)
    na = a * rng.uniform(0.25, 0.40)
    nb = b * rng.uniform(0.25, 0.40)
    nuc_mask = (yr / na) ** 2 + (xr / nb) ** 2 <= 1.0
    amplitude[nuc_mask] = rng.uniform(0.45, 0.65)
    phase[nuc_mask] = rng.uniform(0.8, 1.5)

    # Organelles (small blobs inside cell)
    n_organelles = rng.integers(5, 15)
    for _ in range(n_organelles):
        oy = cy + rng.uniform(-a * 0.7, a * 0.7)
        ox = cx + rng.uniform(-b * 0.7, b * 0.7)
        # Check inside cell
        yr_o = (oy - cy) * cos_a - (ox - cx) * sin_a
        xr_o = (oy - cy) * sin_a + (ox - cx) * cos_a
        if (yr_o / a) ** 2 + (xr_o / b) ** 2 > 0.8:
            continue
        r_org = rng.uniform(3, 10)
        org_mask = (yy - oy) ** 2 + (xx - ox) ** 2 <= r_org ** 2
        amplitude[org_mask] = rng.uniform(0.4, 0.7)
        phase[org_mask] = rng.uniform(0.5, 1.2)

    # Membrane (ring at cell boundary)
    membrane_outer = (yr / a) ** 2 + (xr / b) ** 2 <= 1.0
    membrane_inner = (yr / (a - 2)) ** 2 + (xr / (b - 2)) ** 2 <= 1.0
    membrane = membrane_outer & ~membrane_inner
    amplitude[membrane] = rng.uniform(0.35, 0.55)
    phase[membrane] = rng.uniform(1.0, 1.8)

    amplitude = gaussian_filter(amplitude, sigma=1.0)
    phase = gaussian_filter(phase, sigma=0.8)
    amplitude = np.clip(amplitude, 0.3, 1.0)

    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex128)


def make_crystalline_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Crystalline lattice with defects: periodic atomic columns.

    Returns complex-valued (H, W) object transmission.
    """
    amplitude = np.ones((H, W), dtype=np.float64) * 0.9
    phase = np.zeros((H, W), dtype=np.float64)

    # Lattice parameters
    a1 = rng.uniform(8, 16)  # lattice constant 1
    a2 = rng.uniform(8, 16)  # lattice constant 2
    theta = rng.uniform(np.pi / 6, np.pi / 2)  # lattice angle

    # Generate lattice sites
    for ny in range(-20, H // int(a1) + 20):
        for nx in range(-20, W // int(a2) + 20):
            cy = ny * a1 + nx * a2 * np.cos(theta)
            cx = nx * a2 * np.sin(theta)
            cy_int = int(round(cy))
            cx_int = int(round(cx))
            if 0 <= cy_int < H and 0 <= cx_int < W:
                # Atomic column as a small Gaussian
                r = 3
                y0 = max(0, cy_int - r)
                y1 = min(H, cy_int + r + 1)
                x0 = max(0, cx_int - r)
                x1 = min(W, cx_int + r + 1)
                yy = np.arange(y0, y1)[:, None]
                xx = np.arange(x0, x1)[None, :]
                g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2))
                amplitude[y0:y1, x0:x1] -= 0.3 * g
                phase[y0:y1, x0:x1] += rng.uniform(0.3, 0.8) * g

    # Add defects: vacancies, dislocations
    n_defects = rng.integers(3, 10)
    for _ in range(n_defects):
        dy = rng.integers(20, H - 20)
        dx = rng.integers(20, W - 20)
        dr = rng.integers(5, 20)
        yy, xx = np.ogrid[:H, :W]
        defect_mask = (yy - dy) ** 2 + (xx - dx) ** 2 <= dr ** 2
        amplitude[defect_mask] = rng.uniform(0.5, 0.85)
        phase[defect_mask] = rng.uniform(-0.5, 0.5)

    # Grain boundary
    if rng.random() < 0.6:
        boundary_y = rng.integers(H // 4, 3 * H // 4)
        thickness = rng.integers(2, 5)
        amplitude[boundary_y:boundary_y + thickness, :] = rng.uniform(0.4, 0.6)
        phase[boundary_y:boundary_y + thickness, :] = rng.uniform(0.8, 1.5)

    amplitude = np.clip(amplitude, 0.3, 1.0)
    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex128)


def make_mixed_media_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Mixed media: composite of sharp edges and smooth gradients.

    Returns complex-valued (H, W) object transmission.
    """
    amplitude = np.ones((H, W), dtype=np.float64) * 0.85
    phase = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.ogrid[:H, :W]

    # Smooth background phase gradient
    grad_angle = rng.uniform(0, 2 * np.pi)
    grad_mag = rng.uniform(0.005, 0.02)
    phase += grad_mag * (np.arange(H)[:, None] * np.cos(grad_angle) +
                          np.arange(W)[None, :] * np.sin(grad_angle))

    # Sharp-edged polygonal regions
    n_polygons = rng.integers(4, 10)
    for _ in range(n_polygons):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(20, W - 20)
        n_sides = rng.integers(3, 8)
        radius = rng.uniform(10, 40)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        angles += rng.uniform(0, 2 * np.pi / n_sides)

        # Create polygon mask using ray casting
        vertices_y = cy + radius * np.sin(angles)
        vertices_x = cx + radius * np.cos(angles)
        # Simple approximation: filled circle inscribed in polygon
        effective_r = radius * np.cos(np.pi / n_sides)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= effective_r ** 2
        amplitude[mask] = rng.uniform(0.35, 0.75)
        phase[mask] = rng.uniform(-np.pi / 3, np.pi / 3)

    # Smooth Gaussian blobs (phase objects)
    n_blobs = rng.integers(3, 8)
    for _ in range(n_blobs):
        by = rng.uniform(20, H - 20)
        bx = rng.uniform(20, W - 20)
        bsigma = rng.uniform(8, 25)
        blob = np.exp(-((np.arange(H)[:, None] - by) ** 2 +
                        (np.arange(W)[None, :] - bx) ** 2) / (2 * bsigma ** 2))
        phase += rng.uniform(0.3, 1.0) * blob
        amplitude -= rng.uniform(0.05, 0.2) * blob

    # Thin film interference fringes
    if rng.random() < 0.5:
        freq = rng.uniform(0.05, 0.15)
        fringe_angle = rng.uniform(0, np.pi)
        fringe = np.sin(freq * (np.arange(H)[:, None] * np.cos(fringe_angle) +
                                 np.arange(W)[None, :] * np.sin(fringe_angle)))
        phase += 0.3 * fringe

    amplitude = gaussian_filter(np.clip(amplitude, 0.3, 1.0), sigma=0.5)
    phase = gaussian_filter(phase, sigma=0.3)
    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex128)


PHANTOM_FNS = [
    make_circuit_phantom,
    make_biological_cell_phantom,
    make_crystalline_phantom,
    make_mixed_media_phantom,
]

PHANTOM_NAMES = ["circuit_pattern", "biological_cell", "crystalline", "mixed_media"]


# -- Probe generation -------------------------------------------------------

def make_gaussian_probe(
    size: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a complex-valued Gaussian illumination probe.

    The probe has a Gaussian amplitude envelope and a mild quadratic
    phase curvature (defocus).

    Args:
        size: probe array size (pixels)
        sigma: Gaussian width in pixels
        rng: random generator

    Returns:
        (size, size) complex128 probe array, normalized so |P|^2 sums to 1.
    """
    yy = np.arange(size) - size / 2.0
    xx = np.arange(size) - size / 2.0
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    r2 = Y ** 2 + X ** 2

    # Amplitude: Gaussian envelope
    amp = np.exp(-r2 / (2 * sigma ** 2))

    # Phase: mild defocus (quadratic) + small aberration
    defocus = rng.uniform(-0.5, 0.5)
    phase = defocus * r2 / (size ** 2)

    # Small astigmatism
    astig = rng.uniform(-0.1, 0.1)
    phase += astig * (Y ** 2 - X ** 2) / (size ** 2)

    probe = amp * np.exp(1j * phase)

    # Normalize: |P|^2 sums to 1
    norm = np.sqrt(np.sum(np.abs(probe) ** 2))
    if norm > 0:
        probe /= norm

    return probe.astype(np.complex128)


# -- Scan positions ---------------------------------------------------------

def generate_scan_positions(
    obj_size: int,
    probe_size: int,
    step: int,
) -> np.ndarray:
    """Generate a regular raster scan grid of probe positions.

    Positions are the top-left corner of each probe window.

    Returns:
        (J, 2) int array of (y, x) scan positions.
    """
    positions = []
    # Ensure full coverage: scan from 0 to obj_size - probe_size
    for y in range(0, obj_size - probe_size + 1, step):
        for x in range(0, obj_size - probe_size + 1, step):
            positions.append([y, x])
    return np.array(positions, dtype=np.int32)


# -- Ptychographic forward model --------------------------------------------

def ptychography_forward(
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    noise_level: float,
    detector_saturation: float,
    probe_position_error: float,
    probe_shape_error: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ptychographic forward model with mismatch and noise.

    For each scan position j:
        exit_wave_j = P_perturbed(r - r_j_perturbed) * O(r)
        I_j = |F{exit_wave_j}|^2
        y_j = Poisson(I_j * noise_level / max(I_j)) clipped at saturation

    Args:
        obj: (H, W) complex object transmission
        probe: (Pd, Pd) complex probe function
        positions: (J, 2) scan positions (y, x)
        noise_level: total photon flux per pattern
        detector_saturation: max counts per pixel
        probe_position_error: std of position perturbation (pixels)
        probe_shape_error: multiplicative factor for probe width
        rng: random generator

    Returns:
        y: (J, Pd, Pd) float32 measured diffraction patterns
        H_ideal: (J, Pd, Pd) float32 ideal (noiseless) intensities
        positions_perturbed: (J, 2) float32 actual scan positions used
    """
    J = len(positions)
    Pd = probe.shape[0]
    H, W = obj.shape

    # Perturb probe shape (width scaling)
    if abs(probe_shape_error - 1.0) > 1e-6:
        yy = np.arange(Pd) - Pd / 2.0
        xx = np.arange(Pd) - Pd / 2.0
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        # Re-evaluate Gaussian with modified sigma
        base_sigma = Pd / 6.0  # approximate original sigma
        new_sigma = base_sigma * probe_shape_error
        amp = np.exp(-(Y ** 2 + X ** 2) / (2 * new_sigma ** 2))
        # Keep original phase structure
        probe_perturbed = amp * np.exp(1j * np.angle(probe))
        norm = np.sqrt(np.sum(np.abs(probe_perturbed) ** 2))
        if norm > 0:
            probe_perturbed /= norm
    else:
        probe_perturbed = probe.copy()

    # Perturb positions
    pos_offsets = rng.normal(0, probe_position_error, (J, 2))
    positions_perturbed = positions.astype(np.float64) + pos_offsets

    y = np.zeros((J, Pd, Pd), dtype=np.float32)
    H_ideal = np.zeros((J, Pd, Pd), dtype=np.float32)

    for j in range(J):
        py = int(round(positions_perturbed[j, 0]))
        px = int(round(positions_perturbed[j, 1]))

        # Clamp to valid range
        py = max(0, min(py, H - Pd))
        px = max(0, min(px, W - Pd))

        # Extract object patch at scan position
        obj_patch = obj[py:py + Pd, px:px + Pd]

        # Exit wave = probe * object
        exit_wave = probe_perturbed * obj_patch

        # Far-field diffraction: FFT
        far_field = np.fft.fft2(exit_wave)
        far_field = np.fft.fftshift(far_field)

        # Ideal intensity
        intensity = np.abs(far_field) ** 2

        # Scale to photon level
        total_intensity = intensity.sum()
        if total_intensity > 0:
            intensity_scaled = intensity * (noise_level / total_intensity)
        else:
            intensity_scaled = intensity

        H_ideal[j] = intensity_scaled.astype(np.float32)

        # Poisson noise
        noisy = rng.poisson(np.maximum(intensity_scaled, 1e-6)).astype(np.float64)

        # Detector saturation
        noisy = np.minimum(noisy, detector_saturation)

        y[j] = noisy.astype(np.float32)

    return y, H_ideal, positions_perturbed.astype(np.float32)


# -- ePIE baseline reconstruction ------------------------------------------

def epie_reconstruct(
    y: np.ndarray,
    positions: np.ndarray,
    probe_init: np.ndarray,
    obj_shape: tuple[int, int],
    n_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """ePIE (extended Ptychographic Iterative Engine) reconstruction.

    Alternating projection: update object and probe sequentially for each
    scan position within each iteration.  Uses standard Maiden & Rodenburg
    (2009) update rules with regularised denominators.

    Args:
        y: (J, Pd, Pd) measured diffraction patterns (intensity)
        positions: (J, 2) scan positions
        probe_init: (Pd, Pd) initial probe estimate
        obj_shape: (H, W) object array shape
        n_iterations: number of ePIE iterations
        alpha: object update step size
        beta: probe update step size

    Returns:
        obj_recon: (H, W) complex reconstructed object
        probe_recon: (Pd, Pd) complex reconstructed probe
    """
    J, Pd, _ = y.shape
    H, W = obj_shape

    # Initialize object as uniform complex field
    obj = np.ones((H, W), dtype=np.complex128) * 0.8

    probe = probe_init.copy().astype(np.complex128)

    # Precompute amplitude from measured intensities
    amp_measured = np.sqrt(np.maximum(y, 0)).astype(np.float64)

    for iteration in range(n_iterations):
        # Shuffle scan order each iteration for better convergence
        order = np.arange(J)
        rng_shuffle = np.random.default_rng(iteration * 1000 + 42)
        rng_shuffle.shuffle(order)

        for j in order:
            py = int(round(positions[j, 0]))
            px = int(round(positions[j, 1]))
            py = max(0, min(py, H - Pd))
            px = max(0, min(px, W - Pd))

            # Current object patch
            obj_patch = obj[py:py + Pd, px:px + Pd].copy()

            # Exit wave
            exit_wave = probe * obj_patch

            # Propagate to detector
            G = np.fft.fftshift(np.fft.fft2(exit_wave))

            # Modulus constraint: replace amplitude with measured, keep phase
            G_amp = np.abs(G)
            # Avoid division by zero
            G_updated = np.where(
                G_amp > 1e-12,
                amp_measured[j] * G / (G_amp + 1e-12),
                G,
            )

            # Back-propagate
            exit_wave_updated = np.fft.ifft2(np.fft.ifftshift(G_updated))

            # ePIE update: object (Maiden & Rodenburg 2009 Eq. 9)
            diff = exit_wave_updated - exit_wave
            probe_conj = np.conj(probe)
            probe_abs2_max = np.max(np.abs(probe) ** 2)
            if probe_abs2_max > 1e-12:
                obj[py:py + Pd, px:px + Pd] += (
                    alpha * probe_conj / probe_abs2_max * diff
                )

            # ePIE update: probe (Maiden & Rodenburg 2009 Eq. 10)
            obj_conj = np.conj(obj_patch)
            obj_abs2_max = np.max(np.abs(obj_patch) ** 2)
            if obj_abs2_max > 1e-12:
                probe += beta * obj_conj / obj_abs2_max * diff

    return obj, probe


def _align_complex_recon(
    gt_amp: np.ndarray, recon_amp: np.ndarray,
) -> np.ndarray:
    """Align reconstructed amplitude to ground truth by removing global
    scale/offset ambiguity.

    Solves  recon_aligned = a * recon_amp + b  minimising MSE vs gt_amp.
    """
    r = recon_amp.ravel().astype(np.float64)
    g = gt_amp.ravel().astype(np.float64)
    n = len(r)
    sr = r.sum()
    sg = g.sum()
    srr = (r * r).sum()
    srg = (r * g).sum()
    denom = n * srr - sr * sr
    if abs(denom) < 1e-12:
        return recon_amp
    a = (n * srg - sr * sg) / denom
    b = (sg - a * sr) / n
    aligned = (a * recon_amp + b).astype(np.float32)
    return np.clip(aligned, 0, 1)


# -- Metrics ----------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """PSNR between two real-valued images."""
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    mse = np.mean((gt64 - recon64) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt64.max() - gt64.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray, win_size: int = 7) -> float:
    """Windowed structural similarity (mean SSIM over local patches)."""
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    data_range = gt64.max() - gt64.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Use uniform window for local statistics
    from scipy.ndimage import uniform_filter
    mu_x = uniform_filter(gt64, win_size)
    mu_y = uniform_filter(recon64, win_size)
    mu_x2 = uniform_filter(gt64 ** 2, win_size)
    mu_y2 = uniform_filter(recon64 ** 2, win_size)
    mu_xy = uniform_filter(gt64 * recon64, win_size)

    var_x = mu_x2 - mu_x ** 2
    var_y = mu_y2 - mu_y ** 2
    cov_xy = mu_xy - mu_x * mu_y

    # Clamp negative variances from numerical precision
    var_x = np.maximum(var_x, 0)
    var_y = np.maximum(var_y, 0)

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ssim_map = num / den

    # Crop borders
    pad = win_size // 2
    return float(ssim_map[pad:-pad, pad:-pad].mean())


# -- Image helpers ----------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            lo, hi = np.percentile(nonzero, [1, 99])
            arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# -- Phantom generation pool ------------------------------------------------

def generate_phantoms(
    n: int, seed_offset: int, tier: str,
) -> list[tuple[np.ndarray, str]]:
    """Generate complex-valued phantom objects for a tier.

    Returns list of (complex_object, scene_name).
    """
    phantoms = []
    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]

        phantom_rng = np.random.default_rng(seed_offset + i)
        obj = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((obj, scene_name))

    return phantoms


# -- Mismatch sampling ------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        if k in ("detector_saturation",):
            val = round(val)
        result[k] = val
    return result


# -- Tier generation ---------------------------------------------------------

def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the ptychography benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"ptychography_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    # Generate scan positions (same for all samples)
    scan_positions = generate_scan_positions(IMAGE_SIZE, PROBE_SIZE, SCAN_STEP)
    n_scan = len(scan_positions)

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Ptychography benchmark -- {tier} tier "
            f"(scanning coherent diffractive imaging)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y_j = |F{ P(r - r_j) * O(r) }|^2 + Poisson noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "probe_size": PROBE_SIZE,
            "scan_step": SCAN_STEP,
            "n_scan_positions": int(n_scan),
            "overlap_ratio": float(OVERLAP_RATIO),
            "detector_size": DETECTOR_SIZE,
            "wavelength_nm": WAVELENGTH_NM,
            "pixel_size_nm": PIXEL_SIZE_NM,
        })

        for idx, (obj, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name}, "
                  f"{n_scan} scan positions)...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_scan_positions": n_scan}

            # Generate probe
            probe_sigma = PROBE_SIZE / 6.0
            probe = make_gaussian_probe(PROBE_SIZE, probe_sigma, rng)

            # Forward model with mismatch
            y, H_ideal, pos_perturbed = ptychography_forward(
                obj, probe, scan_positions,
                noise_level=mis["noise_level"],
                detector_saturation=mis["detector_saturation"],
                probe_position_error=mis["probe_position_error"],
                probe_shape_error=mis["probe_shape_error"],
                rng=rng,
            )

            # x_true: amplitude of complex object, normalized to [0, 1]
            x_true_amp = np.abs(obj).astype(np.float32)
            x_true_phase = np.angle(obj).astype(np.float32)
            # Normalize amplitude to [0, 1]
            amp_min, amp_max = x_true_amp.min(), x_true_amp.max()
            if amp_max - amp_min > 1e-8:
                x_true_amp_norm = (x_true_amp - amp_min) / (amp_max - amp_min)
            else:
                x_true_amp_norm = x_true_amp

            # ePIE baseline reconstruction (use nominal positions)
            probe_init = make_gaussian_probe(PROBE_SIZE, PROBE_SIZE / 6.0,
                                             np.random.default_rng(999))
            obj_recon, _ = epie_reconstruct(
                y, scan_positions.astype(np.float32),
                probe_init, (IMAGE_SIZE, IMAGE_SIZE),
                n_iterations=50, alpha=1.0, beta=0.8,
            )

            # Reconstruct amplitude and normalize to [0, 1]
            recon_amp = np.abs(obj_recon).astype(np.float32)
            r_min, r_max = recon_amp.min(), recon_amp.max()
            if r_max - r_min > 1e-8:
                recon_amp_norm = (recon_amp - r_min) / (r_max - r_min)
            else:
                recon_amp_norm = recon_amp

            # Align reconstruction to ground truth (remove global ambiguity)
            recon_amp_norm = _align_complex_recon(x_true_amp_norm, recon_amp_norm)

            psnr = compute_psnr(x_true_amp_norm, recon_amp_norm)
            ssim = compute_ssim(x_true_amp_norm, recon_amp_norm)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            # x_true stored as the amplitude image (256x256) float32
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true_amp_norm, compression="gzip")
            grp.create_dataset("x_true_phase", data=x_true_phase, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("scan_positions", data=scan_positions,
                               compression="gzip")
            grp.create_dataset("probe", data=np.abs(probe).astype(np.float32),
                               compression="gzip")
            grp.create_dataset("reconstruction_baseline",
                               data=recon_amp_norm, compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true_amp_norm.shape),
                "n_scan_positions": int(n_scan),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(
                {**mis, "n_scan_positions": int(n_scan)}
            )
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true_amp_norm, sample_dir / "gt_amplitude.png")
            _save_png(x_true_phase, sample_dir / "gt_phase.png")
            # Save a representative diffraction pattern (middle scan position)
            mid_j = n_scan // 2
            _save_png(np.log1p(y[mid_j]), sample_dir / "diffraction.png")
            _save_png(recon_amp_norm, sample_dir / "recon.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": {**mis, "n_scan_positions": int(n_scan)},
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"pos_err={mis['probe_position_error']:.2f}  "
                  f"shape_err={mis['probe_shape_error']:.3f}  "
                  f"sat={mis['detector_saturation']:.0f}  "
                  f"flux={mis['noise_level']:.0e}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    if tier == "public":
        with open(tier_dir / "true_spec.json", "w") as tf:
            json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs) if all_psnrs else 0.0
    mean_ssim = np.mean(all_ssims) if all_ssims else 0.0
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# -- Gallery image generation -----------------------------------------------

def generate_gallery_images() -> None:
    """Generate/update gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "ptychography"
    )

    h5_path = BENCHMARK_DIR / "public" / "ptychography_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    gallery_indices = [0, 1, 2, 3]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            y = grp["y"][:]
            H_ideal_data = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]
            x_true_phase = grp["x_true_phase"][:]

            # gt.png -- ground truth amplitude
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- representative diffraction pattern (log scale)
            mid_j = y.shape[0] // 2
            _save_png(np.log1p(y[mid_j]), scene_dir / "measurement_I.png")

            # measurement_II.png -- ground truth phase
            _save_png(x_true_phase, scene_dir / "measurement_II.png")

            # recon_I.png -- ePIE baseline reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- |GT - recon| error map
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# -- Main -------------------------------------------------------------------

def main() -> None:
    print("Ptychography Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Object: {IMAGE_SIZE}x{IMAGE_SIZE}, "
          f"Probe: {PROBE_SIZE}x{PROBE_SIZE}, "
          f"Step: {SCAN_STEP} px, "
          f"Overlap: {OVERLAP_RATIO:.1%}\n")

    # Generate scan positions info
    scan_pos = generate_scan_positions(IMAGE_SIZE, PROBE_SIZE, SCAN_STEP)
    print(f"Scan grid: {len(scan_pos)} positions "
          f"({int(np.sqrt(len(scan_pos)))}x{int(np.sqrt(len(scan_pos)))} grid)\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0, tier="public")
    generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000, tier="dev")
    generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000, tier="hidden")
    generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Ptychography benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
