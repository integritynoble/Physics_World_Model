"""Run reconstruction algorithms for lensless system-design modalities.

Modalities (each recovers a different physical dimension from a single 2D shot):
  Existing:
    1. Lensless (C -> D): phase-mask PSF, 1:1
    2. 3D Lensless (Phi_z -> Sigma -> D): diffuser depth encoding, Nz=8 (8:1)
    3. Temporal-Coded Video (M -> C -> Sigma -> D): mask + diffuser, Nt=8 (8:1)
    4. Spectral Lensless (M -> W -> C -> Sigma -> D): mask + dispersion + diffuser, Nb=8 (8:1)
  Multidimensional (depth via diffuser-encoded light field):
    5. 4D Spectral-Depth (M -> W_l -> Phi_z -> Sigma -> D): passive, Nz*Nl:1
    6. 4D Temporal-Depth DMD (M -> Phi_z -> Sigma -> D): active, Nz*Nt:1
    7. 4D Temporal-Depth Streak (M -> W_t -> Phi_z -> Sigma -> D): passive, Nz*Nt:1
    8. 5D Full DMD (M -> W_l -> Phi_z -> Sigma -> D): active, Nz*Nl*Nt:1
    9. 5D Full Streak (M -> W_l -> W_t -> Phi_z -> Sigma -> D): passive, Nz*Nl*Nt:1

Key: Depth encoded via diffuser-encoded light field (depth-dependent PSFs from
     wave propagation through random phase plate + defocus). Each depth plane
     produces a unique speckle pattern, providing maximum measurement diversity.
     Ref: Optica 13(2), 2026 — "Single-shot diffuser-encoded light field imaging"
Each modality runs 5 algorithms with PSNR/SSIM evaluation.
"""
import numpy as np
from scipy import ndimage, fft as sp_fft
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.data import camera, astronaut
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle
import time
import json
import warnings
warnings.filterwarnings("ignore")

import sys
np.random.seed(42)
N = 128            # Spatial resolution
N_SAMPLES = 3      # Number of test samples


# ============================================================================
# PSF generation: phase-mask-based (well-conditioned)
# ============================================================================

def generate_phase_mask_psf(size, seed=42, feature_scale=2.5):
    """Generate a diffuser PSF with good frequency coverage.

    Models a thin random phase mask where PSF = |FT{exp(i*phi)}|^2.
    The phase is smoothed to control feature scale, giving a caustic
    pattern with near-flat |H(f)|^2 spectrum.
    """
    rng = np.random.RandomState(seed)
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    phase = ndimage.gaussian_filter(phase, sigma=feature_scale)
    field = np.exp(1j * phase)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    psf /= psf.sum()
    return psf


def generate_diffuser_depth_psfs(n_depths, size, seed=42, feature_scale=1.0,
                                  defocus_max=40.0):
    """Generate depth-dependent PSFs from diffuser-encoded light field.

    Models wave propagation through a random phase diffuser at varying depths.
    Uses ASYMMETRIC defocus (z=0 in focus, z=N-1 maximally defocused) to avoid
    symmetric PSF pairs and maximize depth diversity.

        PSF(z) = |FT{exp(i*phi_diffuser + i*defocus(z)*r^2)}|^2

    The defocus creates depth-dependent PSFs where each plane produces a
    different speckle/caustic pattern. Moderate feature_scale balances
    PSF structure (OTF bandwidth) and inter-depth diversity.

    Ref: Optica 13(2), 2026 — "Single-shot diffuser-encoded light field imaging"

    Args:
        n_depths: Number of depth planes
        size: Spatial resolution (size x size)
        seed: Random seed for diffuser phase
        feature_scale: Gaussian smoothing sigma for diffuser (controls OTF)
        defocus_max: Maximum defocus at z=N-1 in radians (controls diversity)

    Returns:
        H_ffts: List of FFT transfer functions per depth (complex, size x size)
        psfs: List of PSFs per depth (real, size x size)
    """
    rng = np.random.RandomState(seed)
    phase_diffuser = rng.uniform(0, 2 * np.pi, (size, size))
    if feature_scale > 0:
        phase_diffuser = ndimage.gaussian_filter(phase_diffuser, sigma=feature_scale)

    # Spatial coordinates for defocus quadratic phase
    y_arr, x_arr = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    r2 = (x_arr ** 2 + y_arr ** 2) / (size / 2.0) ** 2

    psfs = []
    H_ffts = []
    for z in range(n_depths):
        # Asymmetric defocus: z=0 is in focus, z=N-1 has maximum defocus
        dz = z / max(n_depths - 1, 1)  # 0 to 1
        defocus_phase = defocus_max * dz * np.pi * r2

        field = np.exp(1j * (phase_diffuser + defocus_phase))
        field = np.fft.ifftshift(field)
        psf = np.abs(sp_fft.fft2(field)) ** 2
        psf /= psf.sum()
        psfs.append(psf)
        H_ffts.append(sp_fft.fft2(psf))

    return H_ffts, psfs


def compute_shift_phase(dx, dy, size):
    """Compute FFT phase ramp for sub-pixel shift (dispersion/streak)."""
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    return np.exp(-2j * np.pi * (fx * dx + fy * dy))


# ============================================================================
# Utility functions
# ============================================================================

def generate_binary_mask(size, fill_factor=0.5, seed=None):
    """Generate a binary random coded aperture mask."""
    rng = np.random.RandomState(seed)
    return (rng.rand(size, size) < fill_factor).astype(np.float64)


def fft_convolve2d(image, kernel):
    """FFT-based 2D convolution with same-size output."""
    s1, s2 = image.shape
    return np.real(sp_fft.ifft2(sp_fft.fft2(image, s=(s1, s2)) *
                                sp_fft.fft2(kernel, s=(s1, s2))))


def add_noise(y, mean_photons=5000, read_noise=3.0):
    """Add Poisson + Gaussian noise."""
    y_photon = np.clip(y * mean_photons, 0, None)
    y_noisy = np.random.poisson(y_photon.astype(np.float64)).astype(np.float64)
    y_noisy += np.random.normal(0, read_noise, y.shape)
    return np.clip(y_noisy / mean_photons, 0, None)


def wiener_deconv_fft(y, H_fft, snr=500):
    """Wiener deconvolution in FFT domain."""
    Y = sp_fft.fft2(y)
    H_conj = np.conj(H_fft)
    return np.real(sp_fft.ifft2(H_conj * Y / (np.abs(H_fft) ** 2 + 1.0 / snr)))


def apply_shift(image, dx, dy):
    """Apply lateral shift (dispersion operator W). Uses np.roll for speed."""
    out = image
    if dx != 0:
        out = np.roll(out, int(dx), axis=1)
    if dy != 0:
        out = np.roll(out, int(dy), axis=0)
    return out


def apply_shift_adjoint(image, dx, dy):
    """Adjoint of lateral shift (reverse W)."""
    return apply_shift(image, -dx, -dy)


def get_test_images(n_samples, size):
    """Generate n test images from standard sources + phantoms."""
    images = []
    cam = resize(camera() / 255.0, (size, size), anti_aliasing=True)
    astro = resize(astronaut().mean(axis=2) / 255.0, (size, size), anti_aliasing=True)
    images.extend([cam, astro])

    for i in range(n_samples - 2):
        img = np.zeros((size, size))
        rng = np.random.RandomState(300 + i)
        y, x = np.mgrid[0:size, 0:size]
        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.randint(20, size - 20, 2)
            rx, ry = rng.randint(8, 50, 2)
            intensity = rng.uniform(0.2, 1.0)
            mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
            img[mask] = intensity
        img = ndimage.gaussian_filter(img, sigma=1.5)
        img = img / max(img.max(), 1e-10)
        images.append(img)
    return images[:n_samples]


def generate_3d_volumes(n_depths, size, n_samples):
    """Generate 3D volumes with objects at different depth planes."""
    volumes = []
    for s in range(n_samples):
        vol = np.zeros((n_depths, size, size))
        rng = np.random.RandomState(200 + s)
        n_objects = rng.randint(5, 10)
        for j in range(n_objects):
            z = rng.randint(0, n_depths)
            cx, cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(6, 25, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            vol[z] += obj * rng.uniform(0.3, 0.9)
        for z in range(n_depths):
            bg = rng.rand(size, size) * 0.03
            vol[z] += ndimage.gaussian_filter(bg, sigma=3)
            mx = vol[z].max()
            if mx > 1e-10:
                vol[z] = np.clip(vol[z] / mx, 0, 1)
        volumes.append(vol)
    return volumes


def generate_video_frames(n_frames, size, n_samples):
    """Generate video sequences with moving objects."""
    videos = []
    for s in range(n_samples):
        frames = np.zeros((n_frames, size, size))
        rng = np.random.RandomState(400 + s)
        base = ndimage.gaussian_filter(rng.rand(size, size) * 0.2, sigma=5)
        n_obj = rng.randint(2, 5)
        for t in range(n_frames):
            frame = base.copy()
            for j in range(n_obj):
                cx = int(size * (0.2 + 0.6 * t / n_frames * (1 + 0.3 * np.sin(j * 1.5)))) % size
                cy = int(size * (0.3 + 0.4 * np.sin(2 * np.pi * t / n_frames + j))) % size
                rx, ry = rng.randint(8, 22, 2)
                yy, xx = np.mgrid[0:size, 0:size]
                obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
                frame += obj * rng.uniform(0.3, 0.8)
            frame = np.clip(frame / max(frame.max(), 1e-10), 0, 1)
            frames[t] = frame
        videos.append(frames)
    return videos


def generate_spectral_cube(n_bands, size, n_samples):
    """Generate hyperspectral data cubes."""
    cubes = []
    for s in range(n_samples):
        cube = np.zeros((n_bands, size, size))
        rng = np.random.RandomState(500 + s)
        base = ndimage.gaussian_filter(rng.rand(size, size) * 0.15, sigma=5)
        n_obj = rng.randint(3, 7)
        for j in range(n_obj):
            cx, cy = rng.randint(20, size - 20, 2)
            rx, ry = rng.randint(12, 40, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            spatial = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            spectrum = ndimage.gaussian_filter1d(rng.rand(n_bands), sigma=1.2)
            spectrum /= max(spectrum.max(), 1e-10)
            for b in range(n_bands):
                cube[b] += spatial * spectrum[b] * rng.uniform(0.3, 0.9)
        for b in range(n_bands):
            cube[b] += base
            cube[b] = np.clip(cube[b] / max(cube[b].max(), 1e-10), 0, 1)
        cubes.append(cube)
    return cubes


def generate_dispersion_shifts(n_bands, max_shift_px=15):
    """Generate wavelength-dependent lateral shifts (prism W_lambda)."""
    return [(int(max_shift_px * (b - n_bands / 2) / n_bands), 0) for b in range(n_bands)]


def generate_streak_shifts(n_frames, max_shift_px=15):
    """Generate time-dependent lateral shifts (streak camera W_t).

    Streak disperses along y-axis (orthogonal to spectral x-axis).
    """
    return [(0, int(max_shift_px * (t - n_frames / 2) / n_frames)) for t in range(n_frames)]


# ============================================================================
# Generic multi-dimensional algorithms
# ============================================================================

def gap_tv(y, forward_fn, adjoint_fn, n_slices, n_iter=80, tv_weight=0.008):
    """GAP-TV for any lensless system.

    forward_fn(x_flat) -> y_est  (datacube -> measurement)
    adjoint_fn(r)      -> x_flat (measurement residual -> datacube update)
    n_slices: number of 2D spatial slices in the datacube

    Uses AtA normalization for stable convergence with diverse operators.
    """
    x = adjoint_fn(y)
    # Compute normalization: AtA(ones) gives per-voxel scaling
    ones = np.ones_like(x)
    AtA_ones = adjoint_fn(forward_fn(ones))
    norm = np.maximum(np.abs(AtA_ones), 1e-6)
    # Normalize initial estimate
    x = x / norm
    x = np.clip(x, 0, None)

    for _ in range(n_iter):
        residual = y - forward_fn(x)
        update = adjoint_fn(residual) / norm
        v = x + update
        for s in range(n_slices):
            v[s] = denoise_tv_chambolle(np.clip(v[s], 0, None), weight=tv_weight)
        x = v
    return np.clip(x, 0, 1)


def fista_tv(y, forward_fn, adjoint_fn, n_slices, step, n_iter=80, tv_weight=0.006):
    """FISTA+TV for any lensless system."""
    x = adjoint_fn(y)
    z = x.copy()
    z_prev = x.copy()
    t_f = 1.0
    for _ in range(n_iter):
        residual = forward_fn(x) - y
        grad = adjoint_fn(residual)
        v = x - step * grad
        for s in range(n_slices):
            z[s] = denoise_tv_chambolle(np.clip(v[s], 0, None), weight=tv_weight)
        t_new = (1 + np.sqrt(1 + 4 * t_f ** 2)) / 2
        x = z + ((t_f - 1) / t_new) * (z - z_prev)
        z_prev = z.copy()
        t_f = t_new
    return np.clip(z, 0, 1)


def admm_tv(y, forward_fn, adjoint_fn, n_slices, n_iter=60, rho=0.5, tv_weight=0.008):
    """ADMM+TV for any lensless system with normalized gradient."""
    x = adjoint_fn(y)
    # Compute normalization
    ones = np.ones_like(x)
    AtA_ones = adjoint_fn(forward_fn(ones))
    norm = np.maximum(np.abs(AtA_ones), 1e-6)
    x = x / norm
    x = np.clip(x, 0, None)

    z = x.copy()
    u = np.zeros_like(x)
    for _ in range(n_iter):
        # x-update: normalized gradient step + penalty
        residual = forward_fn(x) - y
        grad = adjoint_fn(residual) / norm
        x = x - 0.5 * grad + rho * (z - u - x)
        # z-update: TV proximal
        for s in range(n_slices):
            z[s] = denoise_tv_chambolle(np.clip(x[s] + u[s], 0, None),
                                        weight=tv_weight / max(rho, 1e-8))
        u = u + x - z
    return np.clip(z, 0, 1)


def rl_deconv(y, forward_fn, adjoint_fn, n_slices, n_iter=60):
    """Richardson-Lucy for any lensless system."""
    x = np.ones_like(adjoint_fn(y)) * 0.5 / max(n_slices, 1)
    eps = 1e-10
    for _ in range(n_iter):
        y_est = forward_fn(x) + eps
        ratio = y / y_est
        correction = adjoint_fn(ratio)
        x = x * np.clip(correction, 0.1, 10.0)
        x = np.clip(x, 0, 1)
    return x


# ============================================================================
# Modality 1: Lensless (C -> D) — basic 2D deconvolution
# ============================================================================

def run_lensless():
    """Basic lensless imaging with phase-mask PSF."""
    print("=" * 70)
    print("LENSLESS (C -> D) — 2D deconvolution, 1:1")
    print("=" * 70)

    psf = generate_phase_mask_psf(N, seed=42)
    H_fft = sp_fft.fft2(psf)
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) + 1e-8
    images = get_test_images(N_SAMPLES, N)

    def forward(x):
        return fft_convolve2d(x, psf)

    def forward_flat(x_flat):
        return fft_convolve2d(x_flat[0], psf)[np.newaxis]

    algorithms = {
        "Wiener": lambda y: np.clip(wiener_deconv_fft(y, H_fft, snr=800), 0, 1),
        "Wiener+TV": lambda y: denoise_tv_chambolle(
            np.clip(wiener_deconv_fft(y, H_fft, snr=500), 0, None), weight=0.003),
        "FISTA+TV": lambda y: _lensless_fista(y, H_fft, HT_fft, L),
        "ADMM+TV": lambda y: _lensless_admm(y, H_fft, HT_fft),
        "R-L": lambda y: _lensless_rl(y, H_fft, HT_fft),
    }

    return _run_modality("Lensless", images, psf, algorithms,
                         forward_fn=forward, is_2d=True)


def _lensless_fista(y, H_fft, HT_fft, L, n_iter=80, tv_weight=0.002):
    Y_fft = sp_fft.fft2(y)
    x = np.real(sp_fft.ifft2(HT_fft * Y_fft))
    z = x.copy()
    z_prev = x.copy()
    step = 1.0 / L
    t_f = 1.0
    for _ in range(n_iter):
        res_fft = sp_fft.fft2(np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(x))) - y)
        grad = np.real(sp_fft.ifft2(HT_fft * res_fft))
        z = denoise_tv_chambolle(np.clip(x - step * grad, 0, None), weight=tv_weight)
        t_new = (1 + np.sqrt(1 + 4 * t_f ** 2)) / 2
        x = z + ((t_f - 1) / t_new) * (z - z_prev)
        z_prev = z.copy()
        t_f = t_new
    return np.clip(z, 0, 1)


def _lensless_admm(y, H_fft, HT_fft, n_iter=60, rho=1.0, tv_weight=0.003):
    Y_fft = sp_fft.fft2(y)
    x = np.real(sp_fft.ifft2(HT_fft * Y_fft))
    z = x.copy()
    u = np.zeros_like(x)
    denom = np.abs(H_fft) ** 2 + rho
    for _ in range(n_iter):
        rhs = HT_fft * Y_fft + rho * sp_fft.fft2(z - u)
        x = np.real(sp_fft.ifft2(rhs / denom))
        z = denoise_tv_chambolle(np.clip(x + u, 0, None), weight=tv_weight / rho)
        u = u + x - z
    return np.clip(z, 0, 1)


def _lensless_rl(y, H_fft, HT_fft, n_iter=80):
    eps = 1e-10
    x = np.ones((N, N)) * 0.5
    for _ in range(n_iter):
        y_est = np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(x))) + eps
        ratio = y / y_est
        correction = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(ratio)))
        x = x * np.clip(correction, 0.1, 10.0)
        x = np.clip(x, 0, 1)
    return x


# ============================================================================
# Modality 2: 3D Lensless (Phi_z -> Sigma -> D) — diffuser depth encoding
# ============================================================================

NZ = 8  # Depth planes for 3D systems


def run_lensless3d():
    """3D Lensless with diffuser-encoded light field depth encoding.

    Each depth plane sees a unique speckle PSF from diffuser + defocus.
    Forward: y = sum_z conv(x_z, psf_z) / Nz
    """
    print("\n" + "=" * 70)
    print("3D LENSLESS (Phi_z -> Sigma -> D) — diffuser depth, 8:1")
    print("=" * 70)

    H_ffts, psfs = generate_diffuser_depth_psfs(NZ, N, seed=100)
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / NZ + 1e-8
    volumes = generate_3d_volumes(NZ, N, N_SAMPLES)

    def forward(vol):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ):
            Y_fft += H_ffts[z] * sp_fft.fft2(vol[z])
        return np.real(sp_fft.ifft2(Y_fft)) / NZ

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((NZ, N, N))
        for z in range(NZ):
            out[z] = np.real(sp_fft.ifft2(np.conj(H_ffts[z]) * R_fft))
        return out / NZ

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        vol = np.zeros((NZ, N, N))
        for z in range(NZ):
            vol[z] = np.real(sp_fft.ifft2(
                np.conj(H_ffts[z]) * Y_fft / (np.abs(H_ffts[z]) ** 2 + 1.0 / 200)))
        return np.clip(vol / NZ, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, NZ, n_iter=120, tv_weight=0.005),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, NZ, 0.8 / L, n_iter=120, tv_weight=0.003),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, NZ, n_iter=100, rho=0.5, tv_weight=0.005),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, NZ, n_iter=100),
    }

    return _run_modality_3d("3D Lensless", volumes, None, algorithms, forward)


# ============================================================================
# Modality 3: Temporal-Coded (M -> C -> Sigma -> D) — time recovery
# ============================================================================

NT = 8  # Temporal frames


def run_temporal_coded():
    """Temporal-coded lensless video (mask + diffuser)."""
    print("\n" + "=" * 70)
    print("TEMPORAL-CODED LENSLESS (M -> C -> Sigma -> D) — time, 8:1")
    print("=" * 70)

    psf = generate_phase_mask_psf(N, seed=50)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) / NT + 1e-8
    masks = [generate_binary_mask(N, 0.5, seed=600 + t) for t in range(NT)]
    videos = generate_video_frames(NT, N, N_SAMPLES)

    def forward(frames):
        y = np.zeros((N, N))
        for t in range(NT):
            y += np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(masks[t] * frames[t])))
        return y / NT

    def adjoint(r):
        HT_r = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(r)))
        out = np.zeros((NT, N, N))
        for t in range(NT):
            out[t] = masks[t] * HT_r / NT
        return out

    def wiener(y):
        x_dec = wiener_deconv_fft(y, H_fft, snr=300)
        mask_sum = np.clip(sum(masks), 1, None)
        return np.clip(np.array([masks[t] * x_dec / mask_sum * NT for t in range(NT)]), 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, NT, n_iter=80, tv_weight=0.008),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, NT, 0.8 / L, n_iter=80, tv_weight=0.006),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, NT, n_iter=60, rho=0.4, tv_weight=0.008),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, NT, n_iter=60),
    }

    return _run_modality_3d("Temporal-coded", videos, psf, algorithms, forward)


# ============================================================================
# Modality 4: Spectral Lensless (M -> W -> C -> Sigma -> D)
# ============================================================================

NB = 8  # Spectral bands


def run_spectral():
    """Spectral lensless (mask + prism + diffuser)."""
    print("\n" + "=" * 70)
    print("SPECTRAL LENSLESS (M -> W -> C -> Sigma -> D) — wavelength, 8:1")
    print("=" * 70)

    psf = generate_phase_mask_psf(N, seed=55)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) / NB + 1e-8
    mask = generate_binary_mask(N, 0.5, seed=700)
    shifts = generate_dispersion_shifts(NB, max_shift_px=12)
    cubes = generate_spectral_cube(NB, N, N_SAMPLES)

    def forward(cube):
        y = np.zeros((N, N))
        for b in range(NB):
            coded = mask * cube[b]
            dispersed = apply_shift(coded, shifts[b][0], shifts[b][1])
            y += np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(dispersed)))
        return y / NB

    def adjoint(r):
        HT_r = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(r)))
        out = np.zeros((NB, N, N))
        for b in range(NB):
            out[b] = mask * apply_shift_adjoint(HT_r, shifts[b][0], shifts[b][1]) / NB
        return out

    def wiener(y):
        x_dec = wiener_deconv_fft(y, H_fft, snr=200)
        cube_rec = np.zeros((NB, N, N))
        for b in range(NB):
            cube_rec[b] = mask * apply_shift_adjoint(x_dec, shifts[b][0], shifts[b][1]) / NB
        return np.clip(cube_rec, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, NB, n_iter=80, tv_weight=0.008),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, NB, 0.8 / L, n_iter=80, tv_weight=0.005),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, NB, n_iter=60, rho=0.4, tv_weight=0.008),
        "TwoStep": lambda y: _spectral_twostep(y, H_fft, HT_fft, mask, shifts),
    }

    return _run_modality_3d("Spectral", cubes, psf, algorithms, forward)


def _spectral_twostep(y, H_fft, HT_fft, mask, shifts, n_iter=40, tv_weight=0.006):
    x_dec = wiener_deconv_fft(y, H_fft, snr=300)
    cube = np.zeros((NB, N, N))
    for b in range(NB):
        cube[b] = mask * apply_shift_adjoint(x_dec, shifts[b][0], shifts[b][1]) / NB
    for _ in range(n_iter):
        y_est = np.zeros((N, N))
        for b in range(NB):
            y_est += apply_shift(mask * cube[b], shifts[b][0], shifts[b][1])
        y_est /= NB
        residual = x_dec - y_est
        for b in range(NB):
            v = cube[b] + mask * apply_shift_adjoint(residual, shifts[b][0], shifts[b][1]) / NB
            cube[b] = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)
    return np.clip(cube, 0, 1)


# ============================================================================
# Modality 5: 4D Spectral-Depth (M -> W_l -> Phi_z -> Sigma -> D) — PASSIVE
# ============================================================================

NZ4 = 4  # Depth planes for 4D/5D
NL4 = 4  # Spectral bands for 4D/5D
NT4 = 4  # Temporal frames for 4D/5D


def run_4d_spectral_depth():
    """4D Spectral-Depth: mask + prism + diffuser depth (passive).

    Fixed coded aperture mask for spectral dispersion diversity.
    Diffuser provides depth encoding via depth-dependent PSFs.
    """
    print("\n" + "=" * 70)
    print("4D SPECTRAL-DEPTH (M -> W_l -> Phi_z -> Sigma -> D) — passive, 16:1")
    print("=" * 70)

    H_ffts, _ = generate_diffuser_depth_psfs(NZ4, N, seed=200)
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / (NZ4 * NL4) + 1e-8
    shifts = generate_dispersion_shifts(NL4, max_shift_px=12)
    disp_phase = [compute_shift_phase(dx, dy, N) for dx, dy in shifts]
    mask = generate_binary_mask(N, 0.5, seed=750)
    n_slices = NZ4 * NL4
    data = _generate_4d_zl(NZ4, NL4, N, N_SAMPLES)

    def forward(cube_flat):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ4):
            for b in range(NL4):
                idx = z * NL4 + b
                coded_fft = sp_fft.fft2(mask * cube_flat[idx]) * disp_phase[b]
                Y_fft += H_ffts[z] * coded_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = R_fft * np.conj(H_ffts[z])
            for b in range(NL4):
                idx = z * NL4 + b
                out[idx] = mask * np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(disp_phase[b])))
        return out / n_slices

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        vol = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = np.conj(H_ffts[z]) * Y_fft / (
                np.abs(H_ffts[z]) ** 2 + 1.0 / 100)
            for b in range(NL4):
                idx = z * NL4 + b
                vol[idx] = mask * np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(disp_phase[b])))
        return np.clip(vol / n_slices, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.008),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, n_slices, 0.8 / L, n_iter=80, tv_weight=0.006),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, n_slices, n_iter=60, rho=0.5, tv_weight=0.008),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, n_slices, n_iter=60),
    }

    return _run_modality_3d("4D Spectral-Depth", data, None, algorithms, forward)


def _generate_4d_zl(nz, nl, size, n_samples):
    """Generate 4D (z, lambda) datacubes flattened to (nz*nl, size, size)."""
    cubes = []
    for s in range(n_samples):
        cube = np.zeros((nz * nl, size, size))
        rng = np.random.RandomState(800 + s)
        n_obj = rng.randint(4, 8)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            cx, cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(8, 25, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            spatial = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            spectrum = ndimage.gaussian_filter1d(rng.rand(nl), sigma=0.8)
            spectrum /= max(spectrum.max(), 1e-10)
            for b in range(nl):
                cube[z * nl + b] += spatial * spectrum[b] * rng.uniform(0.3, 0.9)
        for idx in range(nz * nl):
            bg = rng.rand(size, size) * 0.02
            cube[idx] += ndimage.gaussian_filter(bg, sigma=3)
            mx = cube[idx].max()
            if mx > 1e-10:
                cube[idx] = np.clip(cube[idx] / mx, 0, 1)
        cubes.append(cube)
    return cubes


# ============================================================================
# Modality 6: 4D Temporal-Depth DMD (M -> Phi_z -> Sigma -> D) — ACTIVE
# ============================================================================

def run_4d_temporal_dmd():
    """4D Temporal-Depth DMD: DMD mask + diffuser depth (active).

    DMD masks provide temporal coding. Diffuser PSFs encode depth.
    """
    print("\n" + "=" * 70)
    print("4D TEMPORAL-DEPTH DMD (M -> Phi_z -> Sigma -> D) — active, 16:1")
    print("=" * 70)

    H_ffts, _ = generate_diffuser_depth_psfs(NZ4, N, seed=300)
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / (NZ4 * NT4) + 1e-8
    masks = [generate_binary_mask(N, 0.5, seed=900 + t) for t in range(NT4)]
    n_slices = NZ4 * NT4
    data = _generate_4d_zt(NZ4, NT4, N, N_SAMPLES)

    def forward(cube_flat):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ4):
            slice_fft = np.zeros((N, N), dtype=complex)
            for t in range(NT4):
                idx = z * NT4 + t
                slice_fft += sp_fft.fft2(masks[t] * cube_flat[idx])
            Y_fft += H_ffts[z] * slice_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z = np.real(sp_fft.ifft2(np.conj(H_ffts[z]) * R_fft))
            for t in range(NT4):
                out[z * NT4 + t] = masks[t] * dec_z
        return out / n_slices

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        mask_sum = np.clip(sum(masks), 1, None)
        vol = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z = np.real(sp_fft.ifft2(
                np.conj(H_ffts[z]) * Y_fft / (np.abs(H_ffts[z]) ** 2 + 1.0 / 80)))
            for t in range(NT4):
                vol[z * NT4 + t] = masks[t] * dec_z / mask_sum
        return np.clip(vol / n_slices * NT4, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.008),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, n_slices, 0.8 / L, n_iter=80, tv_weight=0.006),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, n_slices, n_iter=60, rho=0.5, tv_weight=0.008),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, n_slices, n_iter=60),
    }

    return _run_modality_3d("4D Temporal DMD", data, None, algorithms, forward)


def _generate_4d_zt(nz, nt, size, n_samples):
    """Generate 4D (z, t) datacubes: objects at specific depths with motion."""
    cubes = []
    for s in range(n_samples):
        cube = np.zeros((nz * nt, size, size))
        rng = np.random.RandomState(1000 + s)
        n_obj = rng.randint(3, 6)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            base_cx, base_cy = rng.randint(20, size - 20, 2)
            rx, ry = rng.randint(6, 20, 2)
            for t in range(nt):
                cx = int(base_cx + 8 * np.sin(2 * np.pi * t / nt + j)) % size
                cy = int(base_cy + 5 * np.cos(2 * np.pi * t / nt + j * 0.7)) % size
                yy, xx = np.mgrid[0:size, 0:size]
                obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
                cube[z * nt + t] += obj * rng.uniform(0.3, 0.9)
        for idx in range(nz * nt):
            bg = rng.rand(size, size) * 0.02
            cube[idx] += ndimage.gaussian_filter(bg, sigma=3)
            mx = cube[idx].max()
            if mx > 1e-10:
                cube[idx] = np.clip(cube[idx] / mx, 0, 1)
        cubes.append(cube)
    return cubes


# ============================================================================
# Modality 7: 4D Temporal-Depth Streak (M -> W_t -> Phi_z -> Sigma -> D) — PASSIVE
# ============================================================================

def run_4d_temporal_streak():
    """4D Temporal-Depth Streak: mask + streak + diffuser depth (passive).

    Fixed mask for streak dispersion diversity. Diffuser for depth.
    """
    print("\n" + "=" * 70)
    print("4D TEMPORAL-DEPTH STREAK (M -> W_t -> Phi_z -> Sigma -> D) — passive, 16:1")
    print("=" * 70)

    H_ffts, _ = generate_diffuser_depth_psfs(NZ4, N, seed=400)
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / (NZ4 * NT4) + 1e-8
    t_shifts = generate_streak_shifts(NT4, max_shift_px=12)
    streak_phase = [compute_shift_phase(dx, dy, N) for dx, dy in t_shifts]
    mask = generate_binary_mask(N, 0.5, seed=760)
    n_slices = NZ4 * NT4
    data = _generate_4d_zt(NZ4, NT4, N, N_SAMPLES)

    def forward(cube_flat):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ4):
            for t in range(NT4):
                idx = z * NT4 + t
                coded_fft = sp_fft.fft2(mask * cube_flat[idx]) * streak_phase[t]
                Y_fft += H_ffts[z] * coded_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = R_fft * np.conj(H_ffts[z])
            for t in range(NT4):
                idx = z * NT4 + t
                out[idx] = mask * np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(streak_phase[t])))
        return out / n_slices

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        vol = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = np.conj(H_ffts[z]) * Y_fft / (
                np.abs(H_ffts[z]) ** 2 + 1.0 / 80)
            for t in range(NT4):
                idx = z * NT4 + t
                vol[idx] = mask * np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(streak_phase[t])))
        return np.clip(vol / n_slices, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.008),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, n_slices, 0.8 / L, n_iter=80, tv_weight=0.006),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, n_slices, n_iter=60, rho=0.5, tv_weight=0.008),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, n_slices, n_iter=60),
    }

    return _run_modality_3d("4D Temporal Streak", data, None, algorithms, forward)


# ============================================================================
# Modality 8: 5D Full DMD (M -> W_l -> Phi_z -> Sigma -> D) — ACTIVE
# ============================================================================

def run_5d_dmd():
    """5D Full DMD: DMD mask + prism + diffuser depth (active).

    DMD masks for temporal coding, prism for spectral dispersion,
    diffuser for depth encoding via depth-dependent PSFs.
    """
    print("\n" + "=" * 70)
    print("5D FULL DMD (M -> W_l -> Phi_z -> Sigma -> D) — active, 64:1")
    print("=" * 70)

    H_ffts, _ = generate_diffuser_depth_psfs(NZ4, N, seed=500)
    masks = [generate_binary_mask(N, 0.5, seed=1100 + t) for t in range(NT4)]
    l_shifts = generate_dispersion_shifts(NL4, max_shift_px=12)
    disp_phase = [compute_shift_phase(dx, dy, N) for dx, dy in l_shifts]
    n_slices = NZ4 * NL4 * NT4
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_slices + 1e-8
    data = _generate_5d(NZ4, NL4, NT4, N, N_SAMPLES)

    def forward(cube_flat):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ4):
            for b in range(NL4):
                slice_fft = np.zeros((N, N), dtype=complex)
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    slice_fft += sp_fft.fft2(masks[t] * cube_flat[idx])
                Y_fft += H_ffts[z] * disp_phase[b] * slice_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = R_fft * np.conj(H_ffts[z])
            for b in range(NL4):
                undis = np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(disp_phase[b])))
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    out[idx] = masks[t] * undis
        return out / n_slices

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        mask_sum = np.clip(sum(masks), 1, None)
        vol = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = np.conj(H_ffts[z]) * Y_fft / (
                np.abs(H_ffts[z]) ** 2 + 1.0 / 50)
            for b in range(NL4):
                undis = np.real(sp_fft.ifft2(
                    dec_z_fft * np.conj(disp_phase[b])))
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    vol[idx] = masks[t] * undis / mask_sum
        return np.clip(vol / n_slices * NT4, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, n_slices, n_iter=60, tv_weight=0.010),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, n_slices, 0.8 / L, n_iter=60, tv_weight=0.008),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, n_slices, n_iter=40, rho=0.5, tv_weight=0.010),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, n_slices, n_iter=40),
    }

    return _run_modality_3d("5D Full DMD", data, None, algorithms, forward)


# ============================================================================
# Modality 9: 5D Full Streak (M -> W_l -> W_t -> Phi_z -> Sigma -> D) — PASSIVE
# ============================================================================

def run_5d_streak():
    """5D Full Streak: mask + prism + streak + diffuser depth (passive).

    Fixed mask for dispersion diversity. Diffuser for depth.
    """
    print("\n" + "=" * 70)
    print("5D FULL STREAK (M -> W_l -> W_t -> Phi_z -> Sigma -> D) — PASSIVE, 64:1")
    print("=" * 70)

    H_ffts, _ = generate_diffuser_depth_psfs(NZ4, N, seed=600)
    l_shifts = generate_dispersion_shifts(NL4, max_shift_px=12)
    t_shifts = generate_streak_shifts(NT4, max_shift_px=12)
    disp_phase = [compute_shift_phase(dx, dy, N) for dx, dy in l_shifts]
    streak_phase = [compute_shift_phase(dx, dy, N) for dx, dy in t_shifts]
    mask = generate_binary_mask(N, 0.5, seed=770)
    n_slices = NZ4 * NL4 * NT4
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_slices + 1e-8
    data = _generate_5d(NZ4, NL4, NT4, N, N_SAMPLES)

    def forward(cube_flat):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(NZ4):
            for b in range(NL4):
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    coded_fft = sp_fft.fft2(mask * cube_flat[idx])
                    phase = disp_phase[b] * streak_phase[t]
                    Y_fft += H_ffts[z] * phase * coded_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = R_fft * np.conj(H_ffts[z])
            for b in range(NL4):
                undis_fft = dec_z_fft * np.conj(disp_phase[b])
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    out[idx] = mask * np.real(sp_fft.ifft2(
                        undis_fft * np.conj(streak_phase[t])))
        return out / n_slices

    def wiener(y):
        Y_fft = sp_fft.fft2(y)
        vol = np.zeros((n_slices, N, N))
        for z in range(NZ4):
            dec_z_fft = np.conj(H_ffts[z]) * Y_fft / (
                np.abs(H_ffts[z]) ** 2 + 1.0 / 50)
            for b in range(NL4):
                undis_fft = dec_z_fft * np.conj(disp_phase[b])
                for t in range(NT4):
                    idx = z * NL4 * NT4 + b * NT4 + t
                    vol[idx] = mask * np.real(sp_fft.ifft2(
                        undis_fft * np.conj(streak_phase[t])))
        return np.clip(vol / n_slices, 0, 1)

    algorithms = {
        "Wiener": wiener,
        "GAP-TV": lambda y: gap_tv(y, forward, adjoint, n_slices, n_iter=60, tv_weight=0.010),
        "FISTA+TV": lambda y: fista_tv(y, forward, adjoint, n_slices, 0.8 / L, n_iter=60, tv_weight=0.008),
        "ADMM+TV": lambda y: admm_tv(y, forward, adjoint, n_slices, n_iter=40, rho=0.5, tv_weight=0.010),
        "R-L": lambda y: rl_deconv(y, forward, adjoint, n_slices, n_iter=40),
    }

    return _run_modality_3d("5D Full Streak", data, None, algorithms, forward)


def _generate_5d(nz, nl, nt, size, n_samples):
    """Generate 5D (z, lambda, t) datacubes flattened to (nz*nl*nt, size, size)."""
    cubes = []
    for s in range(n_samples):
        n_total = nz * nl * nt
        cube = np.zeros((n_total, size, size))
        rng = np.random.RandomState(1200 + s)
        n_obj = rng.randint(3, 6)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            base_cx, base_cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(6, 18, 2)
            spectrum = ndimage.gaussian_filter1d(rng.rand(nl), sigma=0.8)
            spectrum /= max(spectrum.max(), 1e-10)
            for b in range(nl):
                for t in range(nt):
                    cx = int(base_cx + 6 * np.sin(2 * np.pi * t / nt + j)) % size
                    cy = int(base_cy + 4 * np.cos(2 * np.pi * t / nt + j * 0.7)) % size
                    yy, xx = np.mgrid[0:size, 0:size]
                    obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
                    idx = z * nl * nt + b * nt + t
                    cube[idx] += obj * spectrum[b] * rng.uniform(0.3, 0.9)
        for idx in range(n_total):
            bg = rng.rand(size, size) * 0.02
            cube[idx] += ndimage.gaussian_filter(bg, sigma=3)
            mx = cube[idx].max()
            if mx > 1e-10:
                cube[idx] = np.clip(cube[idx] / mx, 0, 1)
        cubes.append(cube)
    return cubes


# ============================================================================
# Runner helpers
# ============================================================================

def _run_modality(name, images, psf, algorithms, forward_fn, is_2d=False):
    """Run all algorithms on a 2D modality and report results."""
    results = {alg: {"psnr": [], "ssim": [], "time": []} for alg in algorithms}

    for i, img in enumerate(images):
        print(f"  Sample {i+1}/{len(images)}...", end=" ", flush=True)
        y = add_noise(forward_fn(img), mean_photons=5000, read_noise=3.0)
        for alg_name, algo in algorithms.items():
            t0 = time.time()
            rec = algo(y)
            dt = time.time() - t0
            p = psnr(img, rec, data_range=1.0)
            s = ssim(img, rec, data_range=1.0)
            results[alg_name]["psnr"].append(p)
            results[alg_name]["ssim"].append(s)
            results[alg_name]["time"].append(dt)
        print("done", flush=True)

    _print_results(name, algorithms, results)
    return results


def _run_modality_3d(name, data_list, psfs, algorithms, forward_fn):
    """Run all algorithms on a multi-slice modality and report results."""
    results = {alg: {"psnr": [], "ssim": [], "time": []} for alg in algorithms}

    for i, data in enumerate(data_list):
        print(f"  Sample {i+1}/{len(data_list)}...", end=" ", flush=True)
        n_sl = data.shape[0]
        y = add_noise(forward_fn(data), mean_photons=5000, read_noise=3.0)
        for alg_name, algo in algorithms.items():
            t0 = time.time()
            rec = algo(y)
            dt = time.time() - t0
            sl_psnr = [psnr(data[s], rec[s], data_range=1.0) for s in range(n_sl)]
            sl_ssim = [ssim(data[s], rec[s], data_range=1.0) for s in range(n_sl)]
            results[alg_name]["psnr"].append(np.mean(sl_psnr))
            results[alg_name]["ssim"].append(np.mean(sl_ssim))
            results[alg_name]["time"].append(dt)
        print("done", flush=True)

    _print_results(name, algorithms, results)
    return results


def _print_results(name, algorithms, results):
    """Print formatted results table."""
    print(f"\n{'Method':<14} {'PSNR (dB)':<18} {'SSIM':<14} {'Time (s)':<10}")
    print("-" * 56)
    all_psnr = []
    for alg_name in algorithms:
        p_mean = np.mean(results[alg_name]["psnr"])
        p_std = np.std(results[alg_name]["psnr"])
        s_mean = np.mean(results[alg_name]["ssim"])
        t_mean = np.mean(results[alg_name]["time"])
        print(f"{alg_name:<14} {p_mean:.1f} +/- {p_std:.1f}    {s_mean:.3f}         {t_mean:.1f}")
        all_psnr.append(p_mean)
    mean_p = np.mean(all_psnr)
    std_p = np.std(all_psnr)
    cov = std_p / mean_p * 100 if mean_p > 0 else 0
    print(f"\nMean PSNR: {mean_p:.1f} +/- {std_p:.1f} dB, CoV: {cov:.1f}%")
    print(f"Best: {max(all_psnr):.1f} dB")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTIDIMENSIONAL LENSLESS IMAGING — Expert Study")
    print(f"Spatial: {N}x{N}, Samples: {N_SAMPLES}")
    print("Depth: Diffuser-encoded light field, 2D: Phase-mask PSF")
    print("=" * 70)

    all_results = {}

    # Existing modalities (improved PSF)
    all_results["lensless"] = run_lensless()
    all_results["3d_lensless"] = run_lensless3d()
    all_results["temporal_coded"] = run_temporal_coded()
    all_results["spectral"] = run_spectral()

    # New multidimensional modalities
    all_results["4d_spectral_depth"] = run_4d_spectral_depth()
    all_results["4d_temporal_dmd"] = run_4d_temporal_dmd()
    all_results["4d_temporal_streak"] = run_4d_temporal_streak()
    all_results["5d_dmd"] = run_5d_dmd()
    all_results["5d_streak"] = run_5d_streak()

    # Save JSON results
    json_results = {}
    for mod_name, mod_results in all_results.items():
        json_results[mod_name] = {}
        for alg_name, alg_data in mod_results.items():
            json_results[mod_name][alg_name] = {
                "psnr_mean": float(np.mean(alg_data["psnr"])),
                "psnr_std": float(np.std(alg_data["psnr"])),
                "ssim_mean": float(np.mean(alg_data["ssim"])),
                "time_mean": float(np.mean(alg_data["time"])),
            }

    with open("results/new_modalities_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("\nResults saved to results/new_modalities_results.json")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — All Lensless Modalities")
    print("=" * 70)
    print(f"{'Modality':<28} {'Chain':<28} {'Best PSNR':<12} {'Compression'}")
    print("-" * 80)

    summary = [
        ("Lensless", "C->D", "lensless", "1:1"),
        ("3D Lensless", "Phi_z->Sigma->D", "3d_lensless", "8:1"),
        ("Temporal-coded", "M->C->Sigma->D", "temporal_coded", "8:1"),
        ("Spectral", "M->W->C->Sigma->D", "spectral", "8:1"),
        ("4D Spectral-Depth", "M->W_l->Phi_z->Sigma->D", "4d_spectral_depth", "16:1"),
        ("4D Temporal DMD", "M->Phi_z->Sigma->D", "4d_temporal_dmd", "16:1"),
        ("4D Temporal Streak", "M->W_t->Phi_z->Sigma->D", "4d_temporal_streak", "16:1"),
        ("5D Full DMD", "M->W_l->Phi_z->Sigma->D", "5d_dmd", "64:1"),
        ("5D Full Streak", "M->W_l->W_t->Phi_z->Sigma->D", "5d_streak", "64:1"),
    ]

    for label, chain, key, comp in summary:
        best = max(np.mean(all_results[key][a]["psnr"]) for a in all_results[key])
        print(f"{label:<28} {chain:<28} {best:.1f} dB      {comp}")
