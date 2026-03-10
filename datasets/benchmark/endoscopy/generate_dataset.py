#!/usr/bin/env python3
"""Generate fiber-bundle endoscopy benchmark dataset.

Forward model (wide-angle fiber-bundle endoscopic imaging):

    y = V(r) * D[H * x] + specular + n

where:
    x_true          : 2D tissue surface reflectance (256x256), range [0, 1]
    H               : blur PSF (Gaussian motion/defocus blur)
    D               : barrel distortion (Brown-Conrady radial model, k1 coeff)
    V(r)            : radial vignetting (cos^4 falloff with tuneable strength)
    specular        : bright specular highlight spots (Cook-Torrance BRDF approx)
    n               : additive Gaussian sensor noise

Ground truth phantoms (256x256 grayscale):
    Simulated mucosal tissue surfaces with:
    - Mucosal background texture (fine granular pattern via multi-octave noise)
    - Blood vessel networks (branching vascular tree)
    - Folds / rugae (curved ridges, typical of gastric/colonic mucosa)
    - Lesions: polyps (raised bumps) and ulcers (dark depressions) for dev/hidden

Phantoms per tier:
    Public  : 12 samples (4 normal mucosa + 4 with vessels + 4 with folds)
    Dev     : 20 samples (augmented, includes polyps)
    Hidden  : 20 samples (adversarial, polyps + ulcers + extreme degradations)

Mismatch parameters:
    distortion_k1       : barrel distortion coefficient (0-0.1 public, 0-0.3 hidden)
    vignetting_strength : edge darkening strength (0.1-0.3 public, 0.1-0.6 hidden)
    blur_sigma          : motion/defocus blur sigma (0.5-1.5 public, 0.5-4.0 hidden)
    specular_fraction   : fraction of pixels with specular highlights (0-0.05 public, 0-0.15 hidden)
    noise_sigma         : additive Gaussian sensor noise sigma (0.01-0.03 public, 0.01-0.08 hidden)

CPU reconstruction: Distortion correction + Wiener deconvolution + TV denoising

Usage:
    cd datasets/benchmark/endoscopy
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, map_coordinates
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# ── Mismatch spec ranges per tier ────────────────────────────────────────────

SPEC = {
    "public": {
        "distortion_k1":       {"min": 0.0,  "max": 0.10, "unit": ""},
        "vignetting_strength": {"min": 0.1,  "max": 0.30, "unit": ""},
        "blur_sigma":          {"min": 0.5,  "max": 1.5,  "unit": "pixels"},
        "specular_fraction":   {"min": 0.0,  "max": 0.05, "unit": ""},
        "noise_sigma":         {"min": 0.01, "max": 0.03, "unit": ""},
    },
    "dev": {
        "distortion_k1":       {"min": 0.0,  "max": 0.20, "unit": ""},
        "vignetting_strength": {"min": 0.1,  "max": 0.45, "unit": ""},
        "blur_sigma":          {"min": 0.5,  "max": 2.5,  "unit": "pixels"},
        "specular_fraction":   {"min": 0.0,  "max": 0.10, "unit": ""},
        "noise_sigma":         {"min": 0.01, "max": 0.05, "unit": ""},
    },
    "hidden": {
        "distortion_k1":       {"min": 0.0,  "max": 0.30, "unit": ""},
        "vignetting_strength": {"min": 0.1,  "max": 0.60, "unit": ""},
        "blur_sigma":          {"min": 0.5,  "max": 4.0,  "unit": "pixels"},
        "specular_fraction":   {"min": 0.0,  "max": 0.15, "unit": ""},
        "noise_sigma":         {"min": 0.01, "max": 0.08, "unit": ""},
    },
}

# ── Multi-octave noise for tissue texture ────────────────────────────────────


def _octave_noise(H: int, W: int, rng: np.random.Generator,
                  octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
    """Generate multi-octave smooth noise (Perlin-like) for tissue texture.

    Returns array in [0, 1].
    """
    result = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        freq = 2 ** octave
        sigma = max(H, W) / (freq * 4.0)
        noise = rng.standard_normal((H, W))
        smooth = gaussian_filter(noise, sigma=max(sigma, 1.0))
        result += amplitude * smooth
        total_amp += amplitude
        amplitude *= persistence
    result /= total_amp
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-10:
        result = (result - lo) / (hi - lo)
    return result


# ── Blood vessel network ─────────────────────────────────────────────────────


def _vessel_network(H: int, W: int, rng: np.random.Generator,
                    n_vessels: int = 6, depth: int = 4) -> np.ndarray:
    """Generate a branching blood vessel network.

    Vessels appear as dark linear structures on the mucosal surface.
    Uses a recursive branching model with Gaussian-drawn lines.

    Returns vessel map in [0, 1] where 1 = full vessel.
    """
    vessel_map = np.zeros((H, W), dtype=np.float64)

    def _draw_line(y0, x0, y1, x1, width):
        """Draw a soft Gaussian line on vessel_map."""
        length = max(int(np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)), 1)
        ts = np.linspace(0, 1, length * 3)
        ys = y0 + ts * (y1 - y0)
        xs = x0 + ts * (x1 - x0)
        for y, x in zip(ys, xs):
            iy, ix = int(round(y)), int(round(x))
            hw = int(np.ceil(width * 2.5))
            for dy in range(-hw, hw + 1):
                for dx in range(-hw, hw + 1):
                    py, px = iy + dy, ix + dx
                    if 0 <= py < H and 0 <= px < W:
                        dist2 = dy ** 2 + dx ** 2
                        val = np.exp(-dist2 / (2 * width ** 2))
                        vessel_map[py, px] = max(vessel_map[py, px], val)

    def _branch(y0, x0, angle, length, width, remaining_depth):
        if remaining_depth <= 0 or width < 0.3 or length < 3:
            return
        y1 = y0 + length * np.sin(angle)
        x1 = x0 + length * np.cos(angle)
        # Clip to image
        y1 = np.clip(y1, 2, H - 3)
        x1 = np.clip(x1, 2, W - 3)
        _draw_line(y0, x0, y1, x1, width)
        # Branch
        n_branches = rng.integers(1, 4)
        for _ in range(n_branches):
            branch_angle = angle + rng.uniform(-0.8, 0.8)
            branch_len = length * rng.uniform(0.5, 0.8)
            branch_width = width * rng.uniform(0.5, 0.8)
            _branch(y1, x1, branch_angle, branch_len, branch_width,
                    remaining_depth - 1)

    for _ in range(n_vessels):
        # Start from a random edge or interior point
        side = rng.integers(0, 4)
        if side == 0:
            y0, x0 = float(rng.integers(10, H - 10)), 5.0
        elif side == 1:
            y0, x0 = float(rng.integers(10, H - 10)), float(W - 5)
        elif side == 2:
            y0, x0 = 5.0, float(rng.integers(10, W - 10))
        else:
            y0, x0 = float(H - 5), float(rng.integers(10, W - 10))

        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(30, 80)
        width = rng.uniform(1.0, 2.5)
        _branch(y0, x0, angle, length, width, depth)

    return np.clip(vessel_map, 0, 1)


# ── Folds / rugae ────────────────────────────────────────────────────────────


def _folds(H: int, W: int, rng: np.random.Generator,
           n_folds: int = 5) -> np.ndarray:
    """Generate curved fold/rugae ridges on mucosal surface.

    Folds appear as bright ridges with dark shadows alongside.
    Returns fold map in [-0.3, 0.3] (signed: positive = ridge, negative = valley).
    """
    fold_map = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]

    for _ in range(n_folds):
        # Random curved line (quadratic Bezier)
        p0 = rng.uniform(0, [H, W])
        p1 = rng.uniform(0, [H, W])
        p2 = rng.uniform(0, [H, W])
        width = rng.uniform(3, 10)
        amplitude = rng.uniform(0.10, 0.25)

        ts = np.linspace(0, 1, 200)
        curve_y = (1 - ts) ** 2 * p0[0] + 2 * (1 - ts) * ts * p1[0] + ts ** 2 * p2[0]
        curve_x = (1 - ts) ** 2 * p0[1] + 2 * (1 - ts) * ts * p1[1] + ts ** 2 * p2[1]

        for cy, cx in zip(curve_y, curve_x):
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            # Ridge profile: Gaussian positive with negative flanks
            profile = np.exp(-dist ** 2 / (2 * width ** 2)) - \
                0.3 * np.exp(-dist ** 2 / (2 * (width * 2.5) ** 2))
            fold_map += amplitude * profile / len(ts) * 10

    fold_map = np.clip(fold_map, -0.3, 0.3)
    return fold_map


# ── Lesions: polyps and ulcers ───────────────────────────────────────────────


def _polyps(H: int, W: int, rng: np.random.Generator,
            n_polyps: int = 3) -> np.ndarray:
    """Generate raised polyp bumps (bright circular/oval structures).

    Returns polyp map in [0, 0.4].
    """
    polyp_map = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]

    for _ in range(n_polyps):
        cy = rng.uniform(30, H - 30)
        cx = rng.uniform(30, W - 30)
        ry = rng.uniform(8, 25)
        rx = rng.uniform(8, 25)
        amplitude = rng.uniform(0.15, 0.35)
        angle = rng.uniform(0, np.pi)

        # Rotated ellipse distance
        dy = yy - cy
        dx = xx - cx
        dy_r = dy * np.cos(angle) + dx * np.sin(angle)
        dx_r = -dy * np.sin(angle) + dx * np.cos(angle)
        dist2 = (dy_r / ry) ** 2 + (dx_r / rx) ** 2
        polyp = amplitude * np.exp(-dist2 * 2)
        polyp_map += polyp

    return np.clip(polyp_map, 0, 0.4)


def _ulcers(H: int, W: int, rng: np.random.Generator,
            n_ulcers: int = 2) -> np.ndarray:
    """Generate ulcer depressions (dark patches with bright rim).

    Returns ulcer map in [-0.3, 0.15].
    """
    ulcer_map = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]

    for _ in range(n_ulcers):
        cy = rng.uniform(40, H - 40)
        cx = rng.uniform(40, W - 40)
        radius = rng.uniform(10, 30)
        depth = rng.uniform(0.15, 0.30)

        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        # Dark center with bright rim
        center = -depth * np.exp(-dist ** 2 / (2 * (radius * 0.6) ** 2))
        rim = (depth * 0.4) * np.exp(-(dist - radius) ** 2 / (2 * (radius * 0.25) ** 2))
        ulcer_map += center + rim

    return np.clip(ulcer_map, -0.3, 0.15)


# ── Phantom generators ───────────────────────────────────────────────────────


def _circular_mask(H: int, W: int, margin: float = 0.02) -> np.ndarray:
    """Circular field-of-view mask (endoscope has circular aperture).

    Returns float mask in [0, 1] with soft edge.
    """
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    r = np.sqrt(yy ** 2 + xx ** 2)
    # Hard radius at ~0.95, soft falloff
    radius = 0.95
    mask = np.clip((radius - r) / (margin + 1e-8), 0, 1)
    return mask


def make_normal_phantom(H: int, W: int, seed: int = 0,
                        variant: int = 0) -> tuple[np.ndarray, str]:
    """Normal mucosal tissue: texture + sparse vessels, no lesions."""
    rng = np.random.default_rng(seed)
    circ = _circular_mask(H, W)

    # Base mucosal texture
    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.55)
    # Shift to pinkish mucosal tone: base reflectance ~ 0.35-0.65
    base = 0.35 + 0.30 * texture

    # Sparse vessels (dark lines)
    vessels = _vessel_network(H, W, rng, n_vessels=3 + variant % 3, depth=3)
    base -= 0.20 * vessels

    # Fine granular texture (mucosal pattern)
    fine = _octave_noise(H, W, rng, octaves=3, persistence=0.4)
    fine = gaussian_filter(fine, sigma=1.5)
    base += 0.05 * (fine - 0.5)

    phantom = np.clip(base, 0, 1) * circ
    name = f"normal_{variant:02d}"
    return phantom.astype(np.float64), name


def make_vessel_phantom(H: int, W: int, seed: int = 0,
                        variant: int = 0) -> tuple[np.ndarray, str]:
    """Tissue with prominent blood vessel network."""
    rng = np.random.default_rng(seed)
    circ = _circular_mask(H, W)

    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.5)
    base = 0.35 + 0.25 * texture

    # Dense vessel network
    vessels = _vessel_network(H, W, rng, n_vessels=6 + variant % 4, depth=4)
    base -= 0.30 * vessels

    # Fine texture
    fine = _octave_noise(H, W, rng, octaves=3, persistence=0.4)
    base += 0.04 * (fine - 0.5)

    phantom = np.clip(base, 0, 1) * circ
    name = f"vessel_{variant:02d}"
    return phantom.astype(np.float64), name


def make_fold_phantom(H: int, W: int, seed: int = 0,
                      variant: int = 0) -> tuple[np.ndarray, str]:
    """Tissue with prominent folds/rugae (gastric or colonic mucosa)."""
    rng = np.random.default_rng(seed)
    circ = _circular_mask(H, W)

    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.5)
    base = 0.35 + 0.25 * texture

    # Vessels
    vessels = _vessel_network(H, W, rng, n_vessels=3, depth=3)
    base -= 0.15 * vessels

    # Folds (bright ridges)
    folds = _folds(H, W, rng, n_folds=4 + variant % 3)
    base += folds

    phantom = np.clip(base, 0, 1) * circ
    name = f"fold_{variant:02d}"
    return phantom.astype(np.float64), name


def make_polyp_phantom(H: int, W: int, seed: int = 0,
                       variant: int = 0) -> tuple[np.ndarray, str]:
    """Tissue with polyps (for dev/hidden tiers)."""
    rng = np.random.default_rng(seed)
    circ = _circular_mask(H, W)

    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.5)
    base = 0.35 + 0.25 * texture

    vessels = _vessel_network(H, W, rng, n_vessels=4, depth=3)
    base -= 0.15 * vessels

    folds = _folds(H, W, rng, n_folds=2 + variant % 2)
    base += folds

    polyps = _polyps(H, W, rng, n_polyps=1 + variant % 3)
    base += polyps

    phantom = np.clip(base, 0, 1) * circ
    name = f"polyp_{variant:02d}"
    return phantom.astype(np.float64), name


def make_ulcer_phantom(H: int, W: int, seed: int = 0,
                       variant: int = 0) -> tuple[np.ndarray, str]:
    """Tissue with ulcers and polyps (for hidden tier)."""
    rng = np.random.default_rng(seed)
    circ = _circular_mask(H, W)

    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.5)
    base = 0.35 + 0.25 * texture

    vessels = _vessel_network(H, W, rng, n_vessels=5, depth=3)
    base -= 0.20 * vessels

    folds = _folds(H, W, rng, n_folds=2)
    base += folds

    polyps = _polyps(H, W, rng, n_polyps=1 + variant % 2)
    base += polyps

    ulcers = _ulcers(H, W, rng, n_ulcers=1 + variant % 2)
    base += ulcers

    phantom = np.clip(base, 0, 1) * circ
    name = f"ulcer_{variant:02d}"
    return phantom.astype(np.float64), name


# ── Forward model ─────────────────────────────────────────────────────────────


def _barrel_distortion(image: np.ndarray, k1: float) -> np.ndarray:
    """Apply Brown-Conrady barrel distortion.

    Maps undistorted coords to distorted coords:
        r_d = r_u * (1 + k1 * r_u^2)

    For the inverse (correction), we invert this mapping.
    """
    if abs(k1) < 1e-6:
        return image.copy()

    H, W = image.shape
    cy, cx = H / 2.0, W / 2.0
    # Normalise coordinates to [-1, 1]
    yy, xx = np.mgrid[0:H, 0:W]
    yn = (yy - cy) / cy
    xn = (xx - cx) / cx

    r_u = np.sqrt(yn ** 2 + xn ** 2)
    # Distorted radius
    r_d = r_u * (1 + k1 * r_u ** 2)

    # Scale factor
    scale = np.where(r_u > 1e-8, r_d / r_u, 1.0)
    yn_d = yn * scale
    xn_d = xn * scale

    # Convert back to pixel coordinates
    yy_d = yn_d * cy + cy
    xx_d = xn_d * cx + cx

    # Remap using map_coordinates
    distorted = map_coordinates(image, [yy_d, xx_d], order=3, mode='constant',
                                cval=0.0)
    return distorted


def _inverse_barrel_distortion(image: np.ndarray, k1: float) -> np.ndarray:
    """Correct barrel distortion (inverse mapping).

    Given distorted image, produce undistorted image.
    Uses iterative Newton's method to invert r_d = r_u * (1 + k1 * r_u^2).
    """
    if abs(k1) < 1e-6:
        return image.copy()

    H, W = image.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    yn = (yy - cy) / cy
    xn = (xx - cx) / cx

    r_d = np.sqrt(yn ** 2 + xn ** 2)

    # Solve r_d = r_u * (1 + k1 * r_u^2) for r_u using Newton's method
    r_u = r_d.copy()  # initial guess
    for _ in range(10):
        f = r_u * (1 + k1 * r_u ** 2) - r_d
        fp = 1 + 3 * k1 * r_u ** 2
        r_u = r_u - f / (fp + 1e-10)
        r_u = np.maximum(r_u, 0)

    scale = np.where(r_d > 1e-8, r_u / r_d, 1.0)
    yn_u = yn * scale
    xn_u = xn * scale

    yy_u = yn_u * cy + cy
    xx_u = xn_u * cx + cx

    corrected = map_coordinates(image, [yy_u, xx_u], order=3, mode='constant',
                                cval=0.0)
    return corrected


def _vignetting(H: int, W: int, strength: float) -> np.ndarray:
    """Generate cos^4 vignetting pattern.

    V(r) = (1 - strength) + strength * cos^4(theta)
    where theta = atan(r / f), with r in normalized coords and f ~ 1.

    Returns 2D vignetting map in [1-strength, 1].
    """
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    r = np.sqrt(yy ** 2 + xx ** 2)
    # cos^4 law approximation
    cos_theta = 1.0 / np.sqrt(1.0 + r ** 2)
    v = cos_theta ** 4
    # Scale: v goes from ~0.25 at corners to 1.0 at center
    v_map = (1.0 - strength) + strength * v
    return v_map


def _specular_highlights(H: int, W: int, rng: np.random.Generator,
                         fraction: float) -> np.ndarray:
    """Generate specular highlight spots (bright saturated regions).

    Models direct reflection of the endoscope light source off the wet
    mucosal surface. Appears as small bright blobs.

    Returns specular map in [0, ~1].
    """
    if fraction < 1e-4:
        return np.zeros((H, W), dtype=np.float64)

    n_spots = max(1, int(fraction * H * W / 50))
    spec_map = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]

    for _ in range(n_spots):
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        # Check if within circular FOV
        yn = (cy - H / 2) / (H / 2)
        xn = (cx - W / 2) / (W / 2)
        if yn ** 2 + xn ** 2 > 0.85:
            continue
        radius = rng.uniform(1.5, 5.0)
        intensity = rng.uniform(0.5, 1.0)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        spot = intensity * np.exp(-dist2 / (2 * radius ** 2))
        spec_map = np.maximum(spec_map, spot)

    return spec_map


def endoscopy_forward_model(
    x_true: np.ndarray,
    distortion_k1: float,
    vignetting_strength: float,
    blur_sigma: float,
    specular_fraction: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> dict:
    """Apply the endoscopy forward model.

    Pipeline:
        1. Blur (motion/defocus Gaussian PSF)
        2. Barrel distortion
        3. Vignetting (cos^4 radial falloff)
        4. Add specular highlights
        5. Add Gaussian sensor noise

    Returns dict with image_ideal and image_measured.
    """
    H, W = x_true.shape

    # 1. Blur (Gaussian PSF for motion/defocus)
    if blur_sigma > 0.3:
        blurred = gaussian_filter(x_true, sigma=blur_sigma)
    else:
        blurred = x_true.copy()

    # image_ideal = blurred + distorted + vignetted (but no noise/specular)
    # 2. Barrel distortion
    distorted = _barrel_distortion(blurred, distortion_k1)

    # 3. Vignetting
    vig = _vignetting(H, W, vignetting_strength)
    image_ideal = distorted * vig
    image_ideal = np.clip(image_ideal, 0, 1).astype(np.float32)

    # 4. Add specular highlights
    specular = _specular_highlights(H, W, rng, specular_fraction)
    image_with_spec = image_ideal + specular

    # 5. Gaussian sensor noise
    noise = rng.normal(0, noise_sigma, size=(H, W))
    image_measured = image_with_spec + noise
    image_measured = np.clip(image_measured, 0, 1).astype(np.float32)

    return {
        "image_ideal": image_ideal,
        "image_measured": image_measured,
    }


# ── CPU Reconstruction ───────────────────────────────────────────────────────


def _tv_denoise(image: np.ndarray, weight: float = 0.1,
                n_iter: int = 50) -> np.ndarray:
    """Total Variation denoising (Chambolle 2004 dual projection).

    Minimizes: ||u - image||^2 / 2 + weight * TV(u)
    """
    H, W = image.shape
    px = np.zeros((H, W), dtype=np.float64)
    py = np.zeros((H, W), dtype=np.float64)
    tau = 0.25

    for _ in range(n_iter):
        div = np.zeros((H, W), dtype=np.float64)
        div[1:, :] += px[1:, :] - px[:-1, :]
        div[:, 1:] += py[:, 1:] - py[:, :-1]

        u = image + weight * div

        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]

        norm = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        px = (px + tau * gx / weight) / (1.0 + tau * norm / weight)
        py = (py + tau * gy / weight) / (1.0 + tau * norm / weight)

    div = np.zeros((H, W), dtype=np.float64)
    div[1:, :] += px[1:, :] - px[:-1, :]
    div[:, 1:] += py[:, 1:] - py[:, :-1]
    return image + weight * div


def reconstruct_endoscopy(image_measured: np.ndarray,
                          distortion_k1: float,
                          vignetting_strength: float,
                          blur_sigma: float,
                          noise_sigma: float) -> np.ndarray:
    """CPU baseline reconstruction for endoscopy.

    Pipeline:
        1. Inverse barrel distortion correction
        2. Vignetting compensation (divide by V(r))
        3. Wiener deconvolution for blur PSF
        4. TV denoising for noise reduction
        5. Specular highlight removal (clip outliers)

    Returns:
        recon: (H, W) float64 -- estimated tissue reflectance [0, 1]
    """
    img = image_measured.astype(np.float64)
    H, W = img.shape

    # 0. Specular highlight removal: clip bright outliers
    p95 = np.percentile(img[img > 0], 95)
    bright_mask = img > p95 * 1.3
    if bright_mask.any():
        # Replace specular spots with local median
        from scipy.ndimage import median_filter
        med = median_filter(img, size=7)
        img[bright_mask] = med[bright_mask]

    # 1. Inverse barrel distortion
    img = _inverse_barrel_distortion(img, distortion_k1)

    # 2. Vignetting compensation
    vig = _vignetting(H, W, vignetting_strength)
    vig = np.maximum(vig, 0.05)  # avoid division by zero
    img = img / vig

    # 3. Wiener deconvolution for blur
    if blur_sigma > 0.3:
        k_size = int(np.ceil(blur_sigma * 6)) | 1
        k_half = k_size // 2
        y_k, x_k = np.mgrid[-k_half:k_half + 1, -k_half:k_half + 1]
        psf = np.exp(-(x_k ** 2 + y_k ** 2) / (2 * blur_sigma ** 2))
        psf /= psf.sum()

        pad_shape = (H + psf.shape[0], W + psf.shape[1])
        psf_pad = np.zeros(pad_shape, dtype=np.float64)
        psf_pad[:psf.shape[0], :psf.shape[1]] = psf
        psf_pad = np.roll(psf_pad, -k_half, axis=0)
        psf_pad = np.roll(psf_pad, -k_half, axis=1)

        img_pad = np.zeros(pad_shape, dtype=np.float64)
        img_pad[:H, :W] = img

        PSF_F = np.fft.fft2(psf_pad)
        IMG_F = np.fft.fft2(img_pad)

        noise_power = max(noise_sigma ** 2, 1e-4)
        wiener = np.conj(PSF_F) / (np.abs(PSF_F) ** 2 + noise_power)
        result_f = IMG_F * wiener
        result = np.real(np.fft.ifft2(result_f))
        img = result[:H, :W]

    # 4. TV denoising
    tv_weight = min(0.05 + noise_sigma * 2, 0.3)
    img = _tv_denoise(img, weight=tv_weight, n_iter=60)

    img = np.clip(img, 0, 1)
    return img.astype(np.float64)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Windowed SSIM (11x11 blocks)."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    win_size = 11

    mu_x = uniform_filter(gt, size=win_size, mode='reflect')
    mu_y = uniform_filter(recon, size=win_size, mode='reflect')
    mu_x2 = uniform_filter(gt ** 2, size=win_size, mode='reflect')
    mu_y2 = uniform_filter(recon ** 2, size=win_size, mode='reflect')
    mu_xy = uniform_filter(gt * recon, size=win_size, mode='reflect')

    var_x = mu_x2 - mu_x ** 2
    var_y = mu_y2 - mu_y ** 2
    cov_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ── Image helpers ────────────────────────────────────────────────────────────


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, img_ideal, img_meas, recon, path: Path) -> None:
    """4-panel overview: GT | ideal | measured | recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(img_ideal)
    ov[:, 2 * tw:3 * tw] = _r(img_meas)
    ov[:, 3 * tw:4 * tw] = _r(recon)
    _save_png(ov, path)


# ── Phantom pools per tier ───────────────────────────────────────────────────


def generate_phantoms_public(n: int = 12
                             ) -> list[tuple[np.ndarray, str]]:
    """12 public phantoms: 4 normal mucosa + 4 with vessels + 4 with folds."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_normal_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                            seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_vessel_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                            seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_fold_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                          seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20
                          ) -> list[tuple[np.ndarray, str]]:
    """20 dev phantoms: augmented with polyps, rotation, flip."""
    from scipy.ndimage import rotate as nd_rotate

    generators = [make_normal_phantom, make_vessel_phantom,
                  make_fold_phantom, make_polyp_phantom]
    phantoms = []
    rng = np.random.default_rng(5000)

    for i in range(n):
        gen_fn = generators[i % len(generators)]
        phantom, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=500 + i, variant=i)

        # Augment: rotation
        angle = float(rng.uniform(15, 345))
        phantom = nd_rotate(phantom, angle, reshape=False, mode='constant',
                            cval=0.0)

        # Augment: flip
        if rng.random() < 0.5:
            phantom = np.fliplr(phantom)
        if rng.random() < 0.3:
            phantom = np.flipud(phantom)

        # Augment: zoom
        zoom_f = float(rng.uniform(0.85, 1.15))
        if abs(zoom_f - 1.0) > 0.02:
            phantom = _zoom_crop(phantom, zoom_f, IMAGE_SIZE)

        # Re-apply circular mask after augmentation
        circ = _circular_mask(IMAGE_SIZE, IMAGE_SIZE)
        phantom = np.clip(phantom, 0, 1) * circ
        phantoms.append((phantom, f"dev_{name}"))

    return phantoms


def generate_phantoms_hidden(n: int = 20
                             ) -> list[tuple[np.ndarray, str]]:
    """20 hidden phantoms: adversarial with ulcers, polyps, extreme augments."""
    from scipy.ndimage import rotate as nd_rotate

    generators = [make_vessel_phantom, make_fold_phantom,
                  make_polyp_phantom, make_ulcer_phantom]
    phantoms = []
    rng = np.random.default_rng(8000)

    for i in range(n):
        gen_fn = generators[i % len(generators)]
        phantom, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=800 + i,
                               variant=i + 10)

        # Aggressive rotation + flip
        angle = float(rng.uniform(20, 340))
        phantom = nd_rotate(phantom, angle, reshape=False, mode='constant',
                            cval=0.0)
        if rng.random() < 0.7:
            phantom = np.fliplr(phantom)
        if rng.random() < 0.5:
            phantom = np.flipud(phantom)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.70, 1.30))
        phantom = _zoom_crop(phantom, zoom_f, IMAGE_SIZE)

        # Add extra tiny polyps in some hidden samples
        if rng.random() < 0.4:
            extra = _polyps(IMAGE_SIZE, IMAGE_SIZE, rng, n_polyps=2)
            phantom += extra * 0.3
        # Add extra ulcers in some hidden samples
        if rng.random() < 0.3:
            extra = _ulcers(IMAGE_SIZE, IMAGE_SIZE, rng, n_ulcers=1)
            phantom += extra * 0.5

        # Re-apply circular mask
        circ = _circular_mask(IMAGE_SIZE, IMAGE_SIZE)
        phantom = np.clip(phantom, 0, 1) * circ
        phantoms.append((phantom, f"hidden_{name}"))

    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    from scipy.ndimage import zoom as nd_zoom
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


# ── Tier generation ──────────────────────────────────────────────────────────


def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the endoscopy benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"endoscopy_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM endoscopy benchmark -- {tier} tier "
            f"(barrel distortion + vignetting + blur + specular + noise)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "field_of_view": "circular (endoscope aperture)",
            "forward_model": "y = V(r) * D[H * x] + specular + noise",
        })
        f.attrs["forward_model"] = (
            "y = V(r) * D[H * x] + specular + noise"
        )

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = endoscopy_forward_model(
                x_true,
                distortion_k1=mis["distortion_k1"],
                vignetting_strength=mis["vignetting_strength"],
                blur_sigma=mis["blur_sigma"],
                specular_fraction=mis["specular_fraction"],
                noise_sigma=mis["noise_sigma"],
                rng=rng,
            )
            img_ideal = result["image_ideal"]
            img_meas = result["image_measured"]

            # Baseline reconstruction
            recon = reconstruct_endoscopy(
                img_meas,
                distortion_k1=mis["distortion_k1"],
                vignetting_strength=mis["vignetting_strength"],
                blur_sigma=mis["blur_sigma"],
                noise_sigma=mis["noise_sigma"],
            )
            recon = recon.astype(np.float32)

            # Metrics
            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("image_ideal", data=img_ideal,
                               compression="gzip")
            grp.create_dataset("image_measured", data=img_meas,
                               compression="gzip")
            grp.create_dataset("reconstruction", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(img_ideal, sample_dir / "image_ideal.png")
            _save_png(img_meas, sample_dir / "image_measured.png")
            _save_png(recon, sample_dir / "reconstruction.png")
            _save_overview(x_true, img_ideal, img_meas, recon,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"k1={mis['distortion_k1']:.3f}  "
                  f"vig={mis['vignetting_strength']:.2f}  "
                  f"blur={mis['blur_sigma']:.2f} px  "
                  f"spec={mis['specular_fraction']:.3f}  "
                  f"noise={mis['noise_sigma']:.4f}")

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


# ── README ───────────────────────────────────────────────────────────────────


def _write_top_readme() -> None:
    txt = """# Endoscopy -- Fiber-Bundle Endoscopic Imaging

## Overview

Fiber-bundle endoscopy benchmark with realistic mucosal tissue phantoms
and clinical-grade degradation physics: barrel distortion, cos^4 vignetting,
motion/defocus blur, specular highlights, and Gaussian sensor noise.

## Forward Model

```
y = V(r) * D[H * x] + specular + noise

where:
    x_true              : 2D tissue reflectance map (256x256), range [0, 1]
    H                   : Gaussian blur PSF (motion/defocus)
    D                   : barrel distortion (Brown-Conrady, k1 coefficient)
    V(r)                : cos^4 radial vignetting
    specular            : bright specular highlight spots
    noise               : additive Gaussian sensor noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| Field of View | circular (endoscope aperture) |
| Distortion model | Brown-Conrady radial |
| Vignetting model | cos^4 law |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| distortion_k1 | Barrel distortion | 0-0.10 | 0-0.20 | 0-0.30 |
| vignetting_strength | Edge darkening | 0.1-0.3 | 0.1-0.45 | 0.1-0.6 |
| blur_sigma | Motion/defocus blur | 0.5-1.5 px | 0.5-2.5 px | 0.5-4.0 px |
| specular_fraction | Bright spots | 0-0.05 | 0-0.10 | 0-0.15 |
| noise_sigma | Sensor noise | 0.01-0.03 | 0.01-0.05 | 0.01-0.08 |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Normal mucosa | 4/tier | Base tissue texture + sparse vessels |
| Vessel-rich | 4/tier | Dense branching vascular network |
| Fold/rugae | 4/tier | Curved mucosal ridges (gastric/colonic) |
| Polyps (dev/hidden) | varies | Raised bumps (sessile/pedunculated) |
| Ulcers (hidden) | varies | Dark depressions with bright rim |

## Dataset Structure

```
endoscopy/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal + true spec visible)
+-- dev/       20 samples (blind eval, augmented + polyps)
+-- hidden/    20 samples (adversarial: ulcers + extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32          # Ground truth tissue reflectance [0, 1]
+-- image_ideal (256, 256) float32     # Degraded but noiseless image
+-- image_measured (256, 256) float32  # Fully degraded measurement
+-- reconstruction (256, 256) float32  # Baseline distortion-corr + Wiener + TV
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Brown, D.C. (1966) "Decentering distortion of lenses,"
   Photogrammetric Engineering 32, 444-462.
2. Vigneras, F. et al. (2020) "Endoscopic image enhancement: A review,"
   Computers in Biology and Medicine 120, 103738.
3. Ozyoruk, K.B. et al. (2021) "EndoSLAM dataset and unsupervised monocular
   visual odometry and depth estimation approach for endoscopic videos,"
   Medical Image Analysis 71, 102058.
4. PWM Benchmark: https://pwm.platformai.org/benchmark/endoscopy
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Gallery image generation ─────────────────────────────────────────────────


def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png, recon_III.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "endoscopy")

    h5_path = BENCHMARK_DIR / "public" / "endoscopy_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: normal(0), vessel(4), fold(8), normal_variant(1)
    gallery_sample_indices = [0, 4, 8, 1]

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
            img_meas = grp["image_measured"][:]
            recon = grp["reconstruction"][:]

            # gt.png -- ground truth tissue reflectance
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- measured (degraded) image
            _save_png(img_meas, scene_dir / "measurement_I.png")

            # measurement_II.png -- ideal (no noise/specular) image
            _save_png(img_ideal, scene_dir / "measurement_II.png")

            # recon_I.png -- baseline reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            # recon_III.png -- enhanced measured image (histogram-stretched)
            stretched = _norm(img_meas)
            _save_png(stretched, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} saved to {scene_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Endoscopy Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")

    # ── Public tier (12 samples) ────────────────────────────────────────────
    print("Generating public tier (12 samples: 4 normal + 4 vessel + 4 fold)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=1000)

    # ── Dev tier (20 samples) ──────────────────────────────────────────────
    print("\nGenerating dev tier (20 samples, augmented + polyps)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=5000)

    # ── Hidden tier (20 samples) ──────────────────────────────────────────
    print("\nGenerating hidden tier (20 samples, adversarial + ulcers)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=8000)

    # ── README ──────────────────────────────────────────────────────────────
    _write_top_readme()

    # ── Gallery images ──────────────────────────────────────────────────────
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Endoscopy benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
