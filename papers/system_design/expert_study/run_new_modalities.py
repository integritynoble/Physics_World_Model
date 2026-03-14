"""Run reconstruction algorithms for new system-design modalities.

New modalities:
  1. Coded Lensless (M -> C -> D): binary mask + diffuser PSF
  2. Temporal-Coded Lensless Video (M -> C -> Sigma -> D): temporal mask + diffuser + accumulation
  3. Snapshot Spectral Lensless (M -> W -> C -> Sigma -> D): coded mask + dispersion + diffuser + accumulation

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
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N = 256
N_SAMPLES = 10


# ============================================================================
# Utility functions
# ============================================================================

def generate_random_psf(size=256, spread=30):
    """Generate a diffuse caustic-like PSF."""
    psf = np.random.rand(size, size) ** 2
    psf = ndimage.gaussian_filter(psf, sigma=spread)
    # Add some caustic structure
    for _ in range(5):
        cx, cy = np.random.randint(20, size - 20, 2)
        sx, sy = np.random.uniform(3, 10, 2)
        y, x = np.mgrid[0:size, 0:size]
        caustic = np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))
        psf += caustic * np.random.uniform(0.5, 2.0)
    psf = psf / psf.sum()
    return psf


def generate_binary_mask(size=256, fill_factor=0.5):
    """Generate a binary random coded aperture mask."""
    return (np.random.rand(size, size) < fill_factor).astype(np.float64)


def fft_convolve2d(image, kernel):
    """FFT-based 2D convolution with same-size output."""
    s1, s2 = image.shape
    fft_image = sp_fft.fft2(image, s=(s1, s2))
    fft_kernel = sp_fft.fft2(kernel, s=(s1, s2))
    return np.real(sp_fft.ifft2(fft_image * fft_kernel))


def add_noise(y, mean_photons=5000, read_noise=3.0):
    """Add Poisson + Gaussian noise."""
    # Scale to photon counts
    y_photon = np.clip(y * mean_photons, 0, None)
    y_noisy = np.random.poisson(y_photon.astype(np.float64)).astype(np.float64)
    y_noisy += np.random.normal(0, read_noise, y.shape)
    y_noisy = np.clip(y_noisy / mean_photons, 0, None)
    return y_noisy


def get_test_images(n_samples=10, size=256):
    """Generate n test images."""
    images = []
    # Use standard test images + random phantoms
    cam = resize(camera() / 255.0, (size, size), anti_aliasing=True)
    astro = resize(astronaut().mean(axis=2) / 255.0, (size, size), anti_aliasing=True)
    images.append(cam)
    images.append(astro)

    # Generate additional phantoms
    for i in range(n_samples - 2):
        img = np.zeros((size, size))
        # Random ellipses
        y, x = np.mgrid[0:size, 0:size]
        for _ in range(np.random.randint(3, 8)):
            cx = np.random.randint(30, size - 30)
            cy = np.random.randint(30, size - 30)
            rx = np.random.randint(10, 60)
            ry = np.random.randint(10, 60)
            intensity = np.random.uniform(0.2, 1.0)
            mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
            img[mask] = intensity
        img = ndimage.gaussian_filter(img, sigma=2)
        img = img / max(img.max(), 1e-10)
        images.append(img)
    return images[:n_samples]


def wiener_deconv_fft(y, H_fft, snr=200):
    """Wiener deconvolution in FFT domain."""
    Y = sp_fft.fft2(y)
    H_conj = np.conj(H_fft)
    H_sq = np.abs(H_fft) ** 2
    X = H_conj * Y / (H_sq + 1.0 / snr)
    return np.real(sp_fft.ifft2(X))


# ============================================================================
# Modality 1: Coded Lensless (M -> C -> D)
# ============================================================================

def coded_lensless_forward(x, mask, psf):
    """Forward model: y = H * (M ⊙ x) + noise."""
    coded = mask * x
    y_clean = fft_convolve2d(coded, psf)
    y_noisy = add_noise(y_clean, mean_photons=5000, read_noise=3.0)
    return y_noisy, y_clean


def coded_lensless_wiener(y, mask, psf, snr=150):
    """Wiener deconvolution for coded lensless."""
    # Effective operator in FFT domain: H * diag(M)
    # First deconvolve PSF, then divide by mask
    H_fft = sp_fft.fft2(psf, s=y.shape)
    x_deconv = wiener_deconv_fft(y, H_fft, snr=snr)
    # Unmask (avoid division by zero)
    x_rec = np.where(mask > 0.5, x_deconv / np.clip(mask, 0.5, 1.0), 0.0)
    return np.clip(x_rec, 0, 1)


def coded_lensless_fista_tv(y, mask, psf, n_iter=80, tv_weight=0.005):
    """FISTA+TV for coded lensless."""
    H_fft = sp_fft.fft2(psf, s=y.shape)
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) + 1e-8
    step = 1.0 / L

    # Initialize with Wiener
    x = coded_lensless_wiener(y, mask, psf, snr=150)
    z = x.copy()
    z_prev = x.copy()
    t = 1.0

    for _ in range(n_iter):
        # Gradient: M ⊙ H^T(H(M⊙x) - y)
        Mx = mask * x
        HMx = np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(Mx)))
        residual = HMx - y
        grad = mask * np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(residual)))

        # Proximal step (TV)
        v = x - step * grad
        z = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        x = z + ((t - 1) / t_new) * (z - z_prev)
        z_prev = z.copy()
        t = t_new

    return np.clip(z, 0, 1)


def coded_lensless_admm(y, mask, psf, n_iter=60, rho=0.5, tv_weight=0.005):
    """ADMM with TV denoiser for coded lensless."""
    H_fft = sp_fft.fft2(psf, s=y.shape)
    HT_fft = np.conj(H_fft)
    Y_fft = sp_fft.fft2(y)

    # Initialize
    x = coded_lensless_wiener(y, mask, psf, snr=150)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        # x-update: solve (H^T H M^T M + rho I) x = H^T M^T y + rho(z - u)
        # Approximate: use FFT-domain solve ignoring mask interaction
        rhs_fft = HT_fft * Y_fft + rho * sp_fft.fft2(z - u)
        denom = np.abs(H_fft) ** 2 + rho
        x = np.real(sp_fft.ifft2(rhs_fft / denom))

        # z-update: TV denoising
        z = denoise_tv_chambolle(np.clip(x + u, 0, None), weight=tv_weight / rho)

        # Dual update
        u = u + x - z

    return np.clip(z, 0, 1)


def coded_lensless_pnp_tv(y, mask, psf, n_iter=40, rho=0.5, tv_weight=0.008):
    """PnP-ADMM with TV denoiser for coded lensless."""
    return coded_lensless_admm(y, mask, psf, n_iter=n_iter, rho=rho, tv_weight=tv_weight)


def coded_lensless_rl(y, mask, psf, n_iter=60):
    """Richardson-Lucy adapted for coded lensless."""
    H_fft = sp_fft.fft2(psf, s=y.shape)
    HT_fft = np.conj(H_fft)

    x = np.ones_like(y) * 0.5
    eps = 1e-10

    for _ in range(n_iter):
        # Forward
        Mx = mask * x
        HMx = np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(Mx)))
        ratio = y / (HMx + eps)
        correction = mask * np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(ratio)))
        # RL update (multiplicative)
        x = x * correction
        x = np.clip(x, 0, 1)

    return x


def run_coded_lensless():
    """Run all 5 algorithms on coded lensless and report results."""
    print("=" * 60)
    print("CODED LENSLESS (M -> C -> D)")
    print("=" * 60)

    psf = generate_random_psf(N, spread=25)
    mask = generate_binary_mask(N, fill_factor=0.5)
    images = get_test_images(N_SAMPLES, N)

    algorithms = [
        ("Wiener", lambda y: coded_lensless_wiener(y, mask, psf)),
        ("FISTA+TV", lambda y: coded_lensless_fista_tv(y, mask, psf)),
        ("ADMM+TV", lambda y: coded_lensless_admm(y, mask, psf)),
        ("PnP+TV", lambda y: coded_lensless_pnp_tv(y, mask, psf)),
        ("R-L", lambda y: coded_lensless_rl(y, mask, psf)),
    ]

    results = {name: {"psnr": [], "ssim": [], "time": []} for name, _ in algorithms}

    for i, x_true in enumerate(images):
        y, _ = coded_lensless_forward(x_true, mask, psf)
        for name, algo in algorithms:
            t0 = time.time()
            x_rec = algo(y)
            dt = time.time() - t0
            p = psnr(x_true, x_rec, data_range=1.0)
            s = ssim(x_true, x_rec, data_range=1.0)
            results[name]["psnr"].append(p)
            results[name]["ssim"].append(s)
            results[name]["time"].append(dt)

    print(f"\n{'Method':<12} {'PSNR (dB)':<16} {'SSIM':<14} {'Time (s)':<10}")
    print("-" * 52)
    for name, _ in algorithms:
        p_mean = np.mean(results[name]["psnr"])
        p_std = np.std(results[name]["psnr"])
        s_mean = np.mean(results[name]["ssim"])
        t_mean = np.mean(results[name]["time"])
        print(f"{name:<12} {p_mean:.1f} ± {p_std:.1f}    {s_mean:.3f}         {t_mean:.1f}")

    all_psnr = [np.mean(results[n]["psnr"]) for n, _ in algorithms]
    print(f"\nMean PSNR: {np.mean(all_psnr):.1f} ± {np.std(all_psnr):.1f} dB")
    print(f"CoV: {np.std(all_psnr)/np.mean(all_psnr)*100:.1f}%")

    return results


# ============================================================================
# Modality 2: Temporal-Coded Lensless Video (M -> C -> Sigma -> D)
# ============================================================================

def generate_video_frames(n_frames=8, size=256, n_samples=1):
    """Generate video sequences with moving objects."""
    videos = []
    for _ in range(n_samples):
        frames = []
        # Base scene
        base = np.random.rand(size, size) * 0.3
        base = ndimage.gaussian_filter(base, sigma=5)

        # Moving objects
        n_objects = np.random.randint(2, 5)
        for t in range(n_frames):
            frame = base.copy()
            for j in range(n_objects):
                cx = int(size * 0.2 + size * 0.6 * (t / n_frames) * (1 + 0.3 * np.sin(j * 1.5)))
                cy = int(size * 0.3 + size * 0.4 * np.sin(2 * np.pi * t / n_frames + j))
                cx = cx % size
                cy = cy % size
                rx = np.random.randint(8, 25)
                ry = np.random.randint(8, 25)
                yy, xx = np.mgrid[0:size, 0:size]
                obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
                frame += obj * np.random.uniform(0.3, 0.8)
            frame = np.clip(frame / max(frame.max(), 1e-10), 0, 1)
            frames.append(frame)
        videos.append(np.array(frames))
    return videos


def temporal_coded_forward(video, masks, psf):
    """Forward model: y = sum_t H * (m_t ⊙ x_t) + noise."""
    T = video.shape[0]
    y_clean = np.zeros((N, N))
    for t in range(T):
        coded = masks[t] * video[t]
        y_clean += fft_convolve2d(coded, psf)
    y_clean /= T  # normalize
    y_noisy = add_noise(y_clean, mean_photons=3000, read_noise=3.0)
    return y_noisy, y_clean


def temporal_coded_gap_tv(y, masks, psf, n_iter=60, tv_weight=0.01):
    """GAP-TV for temporal-coded lensless video."""
    T = len(masks)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)

    # Initialize: back-projection
    frames = []
    HT_y = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(y)))
    for t in range(T):
        frames.append(masks[t] * HT_y / T)
    frames = np.array(frames)

    for _ in range(n_iter):
        # Forward projection
        y_est = np.zeros((N, N))
        for t in range(T):
            coded = masks[t] * frames[t]
            y_est += fft_convolve2d(coded, psf)
        y_est /= T

        # GAP update per frame
        residual = y - y_est
        HT_res = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(residual)))
        for t in range(T):
            v = frames[t] + masks[t] * HT_res / T
            frames[t] = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)

    return np.clip(frames, 0, 1)


def temporal_coded_fista_tv(y, masks, psf, n_iter=60, tv_weight=0.008):
    """FISTA+TV for temporal-coded lensless video."""
    T = len(masks)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) / T + 1e-8
    step = 0.8 / L

    # Initialize
    HT_y = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(y)))
    frames = np.array([masks[t] * HT_y / T for t in range(T)])
    z = frames.copy()
    z_prev = frames.copy()
    t_fista = 1.0

    for _ in range(n_iter):
        # Forward
        y_est = np.zeros((N, N))
        for t in range(T):
            y_est += fft_convolve2d(masks[t] * frames[t], psf)
        y_est /= T
        residual = y_est - y

        # Gradient per frame
        HT_res = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(residual)))
        grads = np.array([masks[t] * HT_res / T for t in range(T)])

        # Proximal step (TV per frame)
        v = frames - step * grads
        for t in range(T):
            z[t] = denoise_tv_chambolle(np.clip(v[t], 0, None), weight=tv_weight)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t_fista ** 2)) / 2
        frames = z + ((t_fista - 1) / t_new) * (z - z_prev)
        z_prev = z.copy()
        t_fista = t_new

    return np.clip(z, 0, 1)


def temporal_coded_wiener(y, masks, psf, snr=100):
    """Simple Wiener deconvolution per frame."""
    T = len(masks)
    H_fft = sp_fft.fft2(psf, s=(N, N))

    # Deconvolve y first
    x_deconv = wiener_deconv_fft(y, H_fft, snr=snr)

    # Split among frames using mask patterns
    frames = np.zeros((T, N, N))
    mask_sum = np.zeros((N, N))
    for t in range(T):
        mask_sum += masks[t]
    mask_sum = np.clip(mask_sum, 1, None)

    for t in range(T):
        frames[t] = masks[t] * x_deconv / mask_sum * T
    return np.clip(frames, 0, 1)


def temporal_coded_admm(y, masks, psf, n_iter=50, rho=0.3, tv_weight=0.01):
    """ADMM+TV for temporal-coded lensless video."""
    T = len(masks)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    Y_fft = sp_fft.fft2(y)

    # Initialize
    frames = temporal_coded_wiener(y, masks, psf)
    z = frames.copy()
    u = np.zeros_like(frames)

    for _ in range(n_iter):
        # x-update (per-frame FFT-domain)
        for t in range(T):
            rhs_fft = HT_fft * Y_fft / T + rho * sp_fft.fft2(z[t] - u[t])
            denom = np.abs(H_fft) ** 2 / T + rho
            frames[t] = np.real(sp_fft.ifft2(rhs_fft / denom))

        # z-update (TV per frame)
        for t in range(T):
            z[t] = denoise_tv_chambolle(np.clip(frames[t] + u[t], 0, None),
                                        weight=tv_weight / rho)

        # Dual update
        u = u + frames - z

    return np.clip(z, 0, 1)


def temporal_coded_pnp_tv(y, masks, psf, n_iter=40, rho=0.3, tv_weight=0.012):
    """PnP-ADMM+TV for temporal-coded lensless video."""
    return temporal_coded_admm(y, masks, psf, n_iter=n_iter, rho=rho, tv_weight=tv_weight)


def run_temporal_coded_lensless():
    """Run all 5 algorithms on temporal-coded lensless video."""
    print("\n" + "=" * 60)
    print("TEMPORAL-CODED LENSLESS VIDEO (M -> C -> Sigma -> D)")
    print("=" * 60)

    T = 8
    psf = generate_random_psf(N, spread=25)
    masks = [generate_binary_mask(N, fill_factor=0.5) for _ in range(T)]
    videos = generate_video_frames(n_frames=T, size=N, n_samples=N_SAMPLES)

    algorithms = [
        ("Wiener", lambda y: temporal_coded_wiener(y, masks, psf)),
        ("GAP-TV", lambda y: temporal_coded_gap_tv(y, masks, psf)),
        ("FISTA+TV", lambda y: temporal_coded_fista_tv(y, masks, psf)),
        ("ADMM+TV", lambda y: temporal_coded_admm(y, masks, psf)),
        ("PnP+TV", lambda y: temporal_coded_pnp_tv(y, masks, psf)),
    ]

    results = {name: {"psnr": [], "ssim": [], "time": []} for name, _ in algorithms}

    for i, video in enumerate(videos):
        y, _ = temporal_coded_forward(video, masks, psf)
        for name, algo in algorithms:
            t0 = time.time()
            rec_frames = algo(y)
            dt = time.time() - t0
            # Average PSNR/SSIM over frames
            frame_psnr = [psnr(video[t], rec_frames[t], data_range=1.0) for t in range(T)]
            frame_ssim = [ssim(video[t], rec_frames[t], data_range=1.0) for t in range(T)]
            results[name]["psnr"].append(np.mean(frame_psnr))
            results[name]["ssim"].append(np.mean(frame_ssim))
            results[name]["time"].append(dt)

    print(f"\n{'Method':<12} {'PSNR (dB)':<16} {'SSIM':<14} {'Time (s)':<10}")
    print("-" * 52)
    for name, _ in algorithms:
        p_mean = np.mean(results[name]["psnr"])
        p_std = np.std(results[name]["psnr"])
        s_mean = np.mean(results[name]["ssim"])
        t_mean = np.mean(results[name]["time"])
        print(f"{name:<12} {p_mean:.1f} ± {p_std:.1f}    {s_mean:.3f}         {t_mean:.1f}")

    all_psnr = [np.mean(results[n]["psnr"]) for n, _ in algorithms]
    print(f"\nMean PSNR: {np.mean(all_psnr):.1f} ± {np.std(all_psnr):.1f} dB")
    print(f"CoV: {np.std(all_psnr)/np.mean(all_psnr)*100:.1f}%")

    return results


# ============================================================================
# Modality 3: Snapshot Spectral Lensless (M -> W -> C -> Sigma -> D)
# ============================================================================

N_BANDS = 8  # Number of spectral bands


def generate_spectral_cube(n_bands=8, size=256, n_samples=1):
    """Generate hyperspectral data cubes (spatial x spatial x spectral)."""
    cubes = []
    for _ in range(n_samples):
        cube = np.zeros((n_bands, size, size))
        # Base spatial structure shared across bands (slowly varying spectrum)
        base_spatial = np.random.rand(size, size) * 0.2
        base_spatial = ndimage.gaussian_filter(base_spatial, sigma=5)

        # Random spectral objects
        n_objects = np.random.randint(3, 7)
        for j in range(n_objects):
            cx = np.random.randint(30, size - 30)
            cy = np.random.randint(30, size - 30)
            rx = np.random.randint(15, 50)
            ry = np.random.randint(15, 50)
            yy, xx = np.mgrid[0:size, 0:size]
            spatial = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))

            # Random spectral signature (smooth)
            spectrum = np.random.rand(n_bands)
            spectrum = ndimage.gaussian_filter1d(spectrum, sigma=1.5)
            spectrum = spectrum / max(spectrum.max(), 1e-10)

            for b in range(n_bands):
                cube[b] += spatial * spectrum[b] * np.random.uniform(0.3, 0.9)

        for b in range(n_bands):
            cube[b] += base_spatial
            cube[b] = np.clip(cube[b] / max(cube[b].max(), 1e-10), 0, 1)
        cubes.append(cube)
    return cubes


def generate_dispersion_shifts(n_bands=8, max_shift_px=20):
    """Generate wavelength-dependent lateral shifts (prism/grating dispersion).

    Each spectral band gets a different lateral shift, simulating
    a dispersive element (prism or diffraction grating).
    """
    # Linear dispersion: shift proportional to wavelength index
    shifts = []
    for b in range(n_bands):
        dx = int(max_shift_px * (b - n_bands / 2) / n_bands)
        dy = 0  # 1D dispersion along x-axis
        shifts.append((dx, dy))
    return shifts


def apply_dispersion(image, shift):
    """Apply lateral shift (dispersion) to a 2D image."""
    dx, dy = shift
    return ndimage.shift(image, (dy, dx), order=1, mode='constant', cval=0.0)


def apply_dispersion_adjoint(image, shift):
    """Adjoint of dispersion (reverse shift)."""
    dx, dy = shift
    return ndimage.shift(image, (-dy, -dx), order=1, mode='constant', cval=0.0)


def spectral_lensless_forward(cube, mask, psf, shifts):
    """Forward model: y = sum_b H * W_b(M ⊙ x_b) + noise.

    M -> W -> C -> Sigma -> D chain:
      1. M: coded mask modulates each spectral band
      2. W: prism/grating disperses each band differently
      3. C: diffuser PSF convolves each dispersed band
      4. Sigma: all bands accumulate on the detector
      5. D: detector adds noise
    """
    n_bands = cube.shape[0]
    y_clean = np.zeros((N, N))
    for b in range(n_bands):
        coded = mask * cube[b]                          # M: modulate
        dispersed = apply_dispersion(coded, shifts[b])  # W: disperse
        convolved = fft_convolve2d(dispersed, psf)      # C: convolve
        y_clean += convolved                            # Sigma: accumulate
    y_clean /= n_bands  # normalize
    y_noisy = add_noise(y_clean, mean_photons=2000, read_noise=3.0)
    return y_noisy, y_clean


def spectral_lensless_wiener(y, mask, psf, shifts, n_bands=8, snr=80):
    """Simple Wiener deconvolution + dispersion reversal per band."""
    H_fft = sp_fft.fft2(psf, s=(N, N))
    x_deconv = wiener_deconv_fft(y, H_fft, snr=snr)

    # Split among bands using reverse dispersion + mask
    cube_rec = np.zeros((n_bands, N, N))
    for b in range(n_bands):
        undispersed = apply_dispersion_adjoint(x_deconv, shifts[b])
        cube_rec[b] = np.where(mask > 0.5, undispersed / np.clip(mask, 0.5, 1.0), 0.0) / n_bands
    return np.clip(cube_rec, 0, 1)


def spectral_lensless_gap_tv(y, mask, psf, shifts, n_bands=8, n_iter=60, tv_weight=0.01):
    """GAP-TV for snapshot spectral lensless."""
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)

    # Initialize with back-projection
    HT_y = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(y)))
    cube = np.zeros((n_bands, N, N))
    for b in range(n_bands):
        cube[b] = np.where(mask > 0.5,
                           apply_dispersion_adjoint(HT_y, shifts[b]) / np.clip(mask, 0.5, 1.0),
                           0.0) / n_bands

    for _ in range(n_iter):
        # Forward projection
        y_est = np.zeros((N, N))
        for b in range(n_bands):
            coded = mask * cube[b]
            dispersed = apply_dispersion(coded, shifts[b])
            y_est += fft_convolve2d(dispersed, psf)
        y_est /= n_bands

        # GAP update
        residual = y - y_est
        HT_res = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(residual)))
        for b in range(n_bands):
            v = cube[b] + mask * apply_dispersion_adjoint(HT_res, shifts[b]) / n_bands
            cube[b] = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)

    return np.clip(cube, 0, 1)


def spectral_lensless_fista_tv(y, mask, psf, shifts, n_bands=8, n_iter=60, tv_weight=0.006):
    """FISTA+TV for snapshot spectral lensless."""
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    L = np.max(np.abs(H_fft) ** 2) / n_bands + 1e-8
    step = 0.8 / L

    # Initialize
    cube = spectral_lensless_wiener(y, mask, psf, shifts, n_bands)
    z = cube.copy()
    z_prev = cube.copy()
    t_fista = 1.0

    for _ in range(n_iter):
        # Forward
        y_est = np.zeros((N, N))
        for b in range(n_bands):
            coded = mask * cube[b]
            dispersed = apply_dispersion(coded, shifts[b])
            y_est += fft_convolve2d(dispersed, psf)
        y_est /= n_bands
        residual = y_est - y

        # Gradient per band
        HT_res = np.real(sp_fft.ifft2(HT_fft * sp_fft.fft2(residual)))
        grads = np.zeros_like(cube)
        for b in range(n_bands):
            grads[b] = mask * apply_dispersion_adjoint(HT_res, shifts[b]) / n_bands

        # Proximal step (TV per band)
        v = cube - step * grads
        for b in range(n_bands):
            z[b] = denoise_tv_chambolle(np.clip(v[b], 0, None), weight=tv_weight)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t_fista ** 2)) / 2
        cube = z + ((t_fista - 1) / t_new) * (z - z_prev)
        z_prev = z.copy()
        t_fista = t_new

    return np.clip(z, 0, 1)


def spectral_lensless_admm(y, mask, psf, shifts, n_bands=8, n_iter=50, rho=0.3, tv_weight=0.008):
    """ADMM+TV for snapshot spectral lensless."""
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    Y_fft = sp_fft.fft2(y)

    cube = spectral_lensless_wiener(y, mask, psf, shifts, n_bands)
    z = cube.copy()
    u = np.zeros_like(cube)

    for _ in range(n_iter):
        # x-update (per-band FFT domain)
        for b in range(n_bands):
            rhs_fft = HT_fft * Y_fft / n_bands + rho * sp_fft.fft2(z[b] - u[b])
            denom = np.abs(H_fft) ** 2 / n_bands + rho
            cube[b] = np.real(sp_fft.ifft2(rhs_fft / denom))

        # z-update (TV per band)
        for b in range(n_bands):
            z[b] = denoise_tv_chambolle(np.clip(cube[b] + u[b], 0, None),
                                        weight=tv_weight / rho)
        # Dual update
        u = u + cube - z

    return np.clip(z, 0, 1)


def spectral_lensless_twostep(y, mask, psf, shifts, n_bands=8, n_iter_deconv=40, n_iter_demix=30,
                              tv_weight=0.008):
    """Two-step: (1) Wiener deconv, (2) iterative spectral demixing with TV."""
    H_fft = sp_fft.fft2(psf, s=(N, N))
    x_deconv = wiener_deconv_fft(y, H_fft, snr=100)

    # Initialize cube
    cube = np.zeros((n_bands, N, N))
    for b in range(n_bands):
        cube[b] = np.where(mask > 0.5,
                           apply_dispersion_adjoint(x_deconv, shifts[b]) / np.clip(mask, 0.5, 1.0),
                           0.0) / n_bands

    # Iterative demixing: project-and-denoise
    for _ in range(n_iter_demix):
        # Forward: sum dispersed coded bands
        y_est = np.zeros((N, N))
        for b in range(n_bands):
            y_est += apply_dispersion(mask * cube[b], shifts[b])
        y_est /= n_bands

        # Residual
        residual = x_deconv - y_est
        for b in range(n_bands):
            v = cube[b] + mask * apply_dispersion_adjoint(residual, shifts[b]) / n_bands
            cube[b] = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)

    return np.clip(cube, 0, 1)


def run_spectral_lensless():
    """Run all 5 algorithms on snapshot spectral lensless and report results."""
    print("\n" + "=" * 60)
    print("SNAPSHOT SPECTRAL LENSLESS (M -> W -> C -> Sigma -> D)")
    print("=" * 60)

    psf = generate_random_psf(N, spread=25)
    mask = generate_binary_mask(N, fill_factor=0.5)
    shifts = generate_dispersion_shifts(N_BANDS, max_shift_px=20)
    cubes = generate_spectral_cube(n_bands=N_BANDS, size=N, n_samples=N_SAMPLES)

    algorithms = [
        ("Wiener", lambda y: spectral_lensless_wiener(y, mask, psf, shifts, N_BANDS)),
        ("GAP-TV", lambda y: spectral_lensless_gap_tv(y, mask, psf, shifts, N_BANDS)),
        ("FISTA+TV", lambda y: spectral_lensless_fista_tv(y, mask, psf, shifts, N_BANDS)),
        ("ADMM+TV", lambda y: spectral_lensless_admm(y, mask, psf, shifts, N_BANDS)),
        ("TwoStep", lambda y: spectral_lensless_twostep(y, mask, psf, shifts, N_BANDS)),
    ]

    results = {name: {"psnr": [], "ssim": [], "time": []} for name, _ in algorithms}

    for i, cube in enumerate(cubes):
        y, _ = spectral_lensless_forward(cube, mask, psf, shifts)
        for name, algo in algorithms:
            t0 = time.time()
            rec_cube = algo(y)
            dt = time.time() - t0
            # Average PSNR/SSIM over spectral bands
            band_psnr = [psnr(cube[b], rec_cube[b], data_range=1.0) for b in range(N_BANDS)]
            band_ssim = [ssim(cube[b], rec_cube[b], data_range=1.0) for b in range(N_BANDS)]
            results[name]["psnr"].append(np.mean(band_psnr))
            results[name]["ssim"].append(np.mean(band_ssim))
            results[name]["time"].append(dt)

    print(f"\n{'Method':<12} {'PSNR (dB)':<16} {'SSIM':<14} {'Time (s)':<10}")
    print("-" * 52)
    for name, _ in algorithms:
        p_mean = np.mean(results[name]["psnr"])
        p_std = np.std(results[name]["psnr"])
        s_mean = np.mean(results[name]["ssim"])
        t_mean = np.mean(results[name]["time"])
        print(f"{name:<12} {p_mean:.1f} ± {p_std:.1f}    {s_mean:.3f}         {t_mean:.1f}")

    all_psnr = [np.mean(results[n]["psnr"]) for n, _ in algorithms]
    print(f"\nMean PSNR: {np.mean(all_psnr):.1f} ± {np.std(all_psnr):.1f} dB")
    print(f"CoV: {np.std(all_psnr)/np.mean(all_psnr)*100:.1f}%")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Running new system-design modality experiments...")
    print(f"Image size: {N}x{N}, Samples: {N_SAMPLES}\n")

    results_coded = run_coded_lensless()
    results_temporal = run_temporal_coded_lensless()
    results_spectral = run_spectral_lensless()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Three new modalities demonstrate novel primitive compositions:")
    print("  Coded Lensless:          M -> C -> D")
    print("  Temporal-Coded Lensless: M -> C -> Sigma -> D")
    print("  Spectral Lensless:       M -> W -> C -> Sigma -> D")
