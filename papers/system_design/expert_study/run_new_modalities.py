"""Run reconstruction algorithms for new system-design modalities.

New modalities (each recovers a different physical dimension from a single 2D shot):
  1. 3D Lensless (C -> Sigma -> D): depth-dependent PSFs, recovers Nz=8 depth planes (8:1)
  2. Temporal-Coded Lensless Video (M -> C -> Sigma -> D): temporal mask + diffuser (8:1)
  3. Snapshot Spectral Lensless (M -> W -> C -> Sigma -> D): mask + dispersion + diffuser (8:1)

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
# Modality 1: 3D Lensless (C -> Sigma -> D)
#   Depth-dependent PSFs: y = sum_z H_z * x_z + noise
#   Recovers Nz depth planes from a single 2D measurement (Nz:1 compression)
# ============================================================================

N_DEPTHS = 8  # Number of depth planes to recover


def generate_depth_psfs(n_depths=8, size=128, base_spread=15):
    """Generate depth-dependent PSFs for a diffuser camera.

    Each depth plane has a different PSF because the diffuser's caustic
    pattern changes with object distance (defocus + magnification shift).
    """
    psfs = []
    for z in range(n_depths):
        # Spread increases with depth (defocus)
        spread = base_spread + z * 3
        psf = np.random.RandomState(42 + z).rand(size, size) ** 2
        psf = ndimage.gaussian_filter(psf, sigma=spread)
        # Add depth-specific caustic structure
        rng = np.random.RandomState(100 + z)
        for _ in range(4):
            cx, cy = rng.randint(20, size - 20, 2)
            sx, sy = rng.uniform(3, 8 + z, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            caustic = np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
            psf += caustic * rng.uniform(0.3, 1.5)
        # Add lateral shift (magnification changes with depth)
        shift_px = (z - n_depths / 2) * 2.0
        psf = ndimage.shift(psf, (shift_px, shift_px * 0.5), order=1, mode='wrap')
        psf = psf / psf.sum()
        psfs.append(psf)
    return psfs


def generate_3d_volumes(n_depths=8, size=128, n_samples=1):
    """Generate 3D volumes with objects at different depth planes."""
    volumes = []
    for s in range(n_samples):
        vol = np.zeros((n_depths, size, size))
        rng = np.random.RandomState(200 + s)

        # Place objects at random depth planes (sparse in z)
        n_objects = rng.randint(4, 8)
        for j in range(n_objects):
            z = rng.randint(0, n_depths)
            cx = rng.randint(20, size - 20)
            cy = rng.randint(20, size - 20)
            rx = rng.randint(8, 30)
            ry = rng.randint(8, 30)
            yy, xx = np.mgrid[0:size, 0:size]
            obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            vol[z] += obj * rng.uniform(0.3, 0.9)

        # Add faint background across a few planes
        for z in range(n_depths):
            bg = rng.rand(size, size) * 0.05
            vol[z] += ndimage.gaussian_filter(bg, sigma=5)
            vol[z] = np.clip(vol[z] / max(vol[z].max(), 1e-10), 0, 1)
        volumes.append(vol)
    return volumes


def lensless3d_forward(volume, psfs):
    """Forward model: y = sum_z H_z * x_z + noise.

    C -> Sigma -> D chain:
      C: each depth plane convolved with its depth-dependent PSF
      Sigma: all depth planes accumulate on the detector
      D: detector adds noise
    """
    n_depths = volume.shape[0]
    y_clean = np.zeros((N, N))
    for z in range(n_depths):
        y_clean += fft_convolve2d(volume[z], psfs[z])
    y_clean /= n_depths
    y_noisy = add_noise(y_clean, mean_photons=5000, read_noise=3.0)
    return y_noisy, y_clean


def lensless3d_wiener(y, psfs, n_depths=8, snr=80):
    """Per-depth Wiener deconvolution (naive baseline)."""
    vol = np.zeros((n_depths, N, N))
    for z in range(n_depths):
        H_fft = sp_fft.fft2(psfs[z], s=(N, N))
        vol[z] = wiener_deconv_fft(y, H_fft, snr=snr) / n_depths
    return np.clip(vol, 0, 1)


def lensless3d_gap_tv(y, psfs, n_depths=8, n_iter=60, tv_weight=0.01):
    """GAP-TV for 3D lensless depth recovery."""
    H_ffts = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT_ffts = [np.conj(h) for h in H_ffts]

    # Initialize with back-projection
    vol = np.zeros((n_depths, N, N))
    for z in range(n_depths):
        vol[z] = np.real(sp_fft.ifft2(HT_ffts[z] * sp_fft.fft2(y))) / n_depths

    for _ in range(n_iter):
        # Forward projection
        y_est = np.zeros((N, N))
        for z in range(n_depths):
            y_est += np.real(sp_fft.ifft2(H_ffts[z] * sp_fft.fft2(vol[z])))
        y_est /= n_depths

        # GAP update per depth
        residual = y - y_est
        Y_res_fft = sp_fft.fft2(residual)
        for z in range(n_depths):
            v = vol[z] + np.real(sp_fft.ifft2(HT_ffts[z] * Y_res_fft)) / n_depths
            vol[z] = denoise_tv_chambolle(np.clip(v, 0, None), weight=tv_weight)

    return np.clip(vol, 0, 1)


def lensless3d_fista_tv(y, psfs, n_depths=8, n_iter=60, tv_weight=0.006):
    """FISTA+TV for 3D lensless depth recovery."""
    H_ffts = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT_ffts = [np.conj(h) for h in H_ffts]
    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_depths + 1e-8
    step = 0.8 / L

    vol = lensless3d_wiener(y, psfs, n_depths)
    z_vol = vol.copy()
    z_prev = vol.copy()
    t_fista = 1.0

    for _ in range(n_iter):
        # Forward
        y_est = np.zeros((N, N))
        for z in range(n_depths):
            y_est += np.real(sp_fft.ifft2(H_ffts[z] * sp_fft.fft2(vol[z])))
        y_est /= n_depths
        residual = y_est - y
        R_fft = sp_fft.fft2(residual)

        # Gradient per depth
        grads = np.zeros_like(vol)
        for z in range(n_depths):
            grads[z] = np.real(sp_fft.ifft2(HT_ffts[z] * R_fft)) / n_depths

        v = vol - step * grads
        for z in range(n_depths):
            z_vol[z] = denoise_tv_chambolle(np.clip(v[z], 0, None), weight=tv_weight)

        t_new = (1 + np.sqrt(1 + 4 * t_fista ** 2)) / 2
        vol = z_vol + ((t_fista - 1) / t_new) * (z_vol - z_prev)
        z_prev = z_vol.copy()
        t_fista = t_new

    return np.clip(z_vol, 0, 1)


def lensless3d_admm(y, psfs, n_depths=8, n_iter=50, rho=0.3, tv_weight=0.008):
    """ADMM+TV for 3D lensless depth recovery."""
    H_ffts = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT_ffts = [np.conj(h) for h in H_ffts]
    Y_fft = sp_fft.fft2(y)

    vol = lensless3d_wiener(y, psfs, n_depths)
    z_vol = vol.copy()
    u = np.zeros_like(vol)

    for _ in range(n_iter):
        for z in range(n_depths):
            rhs_fft = HT_ffts[z] * Y_fft / n_depths + rho * sp_fft.fft2(z_vol[z] - u[z])
            denom = np.abs(H_ffts[z]) ** 2 / n_depths + rho
            vol[z] = np.real(sp_fft.ifft2(rhs_fft / denom))

        for z in range(n_depths):
            z_vol[z] = denoise_tv_chambolle(np.clip(vol[z] + u[z], 0, None),
                                            weight=tv_weight / rho)
        u = u + vol - z_vol

    return np.clip(z_vol, 0, 1)


def lensless3d_rl(y, psfs, n_depths=8, n_iter=60):
    """Richardson-Lucy for 3D lensless depth recovery."""
    H_ffts = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT_ffts = [np.conj(h) for h in H_ffts]
    eps = 1e-10

    vol = np.ones((n_depths, N, N)) * 0.5 / n_depths

    for _ in range(n_iter):
        # Forward
        y_est = np.zeros((N, N))
        for z in range(n_depths):
            y_est += np.real(sp_fft.ifft2(H_ffts[z] * sp_fft.fft2(vol[z])))
        y_est /= n_depths

        ratio = y / (y_est + eps)
        R_fft = sp_fft.fft2(ratio)

        for z in range(n_depths):
            correction = np.real(sp_fft.ifft2(HT_ffts[z] * R_fft)) / n_depths
            vol[z] = vol[z] * np.clip(correction, 0.1, 10.0)
            vol[z] = np.clip(vol[z], 0, 1)

    return vol


def run_lensless3d():
    """Run all 5 algorithms on 3D lensless and report results."""
    print("=" * 60)
    print("3D LENSLESS (C -> Sigma -> D) — depth recovery")
    print("=" * 60)

    psfs = generate_depth_psfs(N_DEPTHS, N, base_spread=12)
    volumes = generate_3d_volumes(N_DEPTHS, N, N_SAMPLES)

    algorithms = [
        ("Wiener", lambda y: lensless3d_wiener(y, psfs, N_DEPTHS)),
        ("GAP-TV", lambda y: lensless3d_gap_tv(y, psfs, N_DEPTHS)),
        ("FISTA+TV", lambda y: lensless3d_fista_tv(y, psfs, N_DEPTHS)),
        ("ADMM+TV", lambda y: lensless3d_admm(y, psfs, N_DEPTHS)),
        ("R-L", lambda y: lensless3d_rl(y, psfs, N_DEPTHS)),
    ]

    results = {name: {"psnr": [], "ssim": [], "time": []} for name, _ in algorithms}

    for i, vol in enumerate(volumes):
        y, _ = lensless3d_forward(vol, psfs)
        for name, algo in algorithms:
            t0 = time.time()
            rec_vol = algo(y)
            dt = time.time() - t0
            # Average PSNR/SSIM over depth planes
            depth_psnr = [psnr(vol[z], rec_vol[z], data_range=1.0) for z in range(N_DEPTHS)]
            depth_ssim = [ssim(vol[z], rec_vol[z], data_range=1.0) for z in range(N_DEPTHS)]
            results[name]["psnr"].append(np.mean(depth_psnr))
            results[name]["ssim"].append(np.mean(depth_ssim))
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

    results_3d = run_lensless3d()
    results_temporal = run_temporal_coded_lensless()
    results_spectral = run_spectral_lensless()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Three new modalities, each recovering a different physical dimension:")
    print("  3D Lensless:             C -> Sigma -> D          (depth, 8:1)")
    print("  Temporal-Coded Lensless: M -> C -> Sigma -> D     (time, 8:1)")
    print("  Spectral Lensless:       M -> W -> C -> Sigma -> D (wavelength, 8:1)")
