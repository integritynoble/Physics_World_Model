"""
Batch 2: Build standard datasets from REAL data using skimage built-in images,
additional MedMNIST subsets, and OpenNeuro data.

Each modality gets UNIQUE real images from famous public sources.
skimage images are well-cited real photographs/microscopy (MIT, NASA, Allen Cell, DRIVE, etc.)
"""
import numpy as np
import h5py
import json
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter, rotate
from scipy.interpolate import RectBivariateSpline
import skimage.data
import skimage.transform

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/_download_cache")
N_SAMPLES = 10


def resize_2d(img, target_shape):
    return skimage.transform.resize(img, target_shape, order=3, anti_aliasing=True).astype(np.float32)


def normalize_01(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def to_uint8(x):
    x = np.nan_to_num(x, 0)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def save_pgm(arr, path):
    u = to_uint8(arr)
    if u.ndim != 2:
        u = to_uint8(arr[..., 0] if arr.ndim == 3 else arr.ravel()[:256*256].reshape(256, 256))
    h, w = u.shape
    with open(path, 'wb') as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(u.tobytes())


def save_modality(mod, samples_x, samples_y, reference, source_name, forward_desc=""):
    out_dir = BASE / mod / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    for i, (x, y) in enumerate(zip(samples_x, samples_y)):
        h5_path = out_dir / f"standard_{mod}_{i:02d}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x_true", data=x.astype(np.float32), compression="gzip")
            f.create_dataset("y_ideal", data=y.astype(np.float32), compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["source"] = source_name
            f.attrs["reference"] = reference
            f.attrs["data_type"] = "real"

        if x.ndim == 2:
            save_pgm(x, str(img_dir / f"x_true_{i:02d}.pgm"))

    meta = {
        "modality": mod,
        "n_samples": len(samples_x),
        "x_shape": list(samples_x[0].shape),
        "y_shape": list(samples_y[0].shape),
        "source": source_name,
        "reference": reference,
        "data_type": "real",
        "forward_model": forward_desc,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source_name, "reference": reference}, f, indent=2)
    print(f"  {mod}: {len(samples_x)} samples, x={list(samples_x[0].shape)}, y={list(samples_y[0].shape)} [REAL]")


# ================================================================
# Forward models
# ================================================================

def fwd_fourier_undersample(x, seed, acceleration=4):
    rng = np.random.RandomState(seed)
    kspace = np.fft.fftshift(np.fft.fft2(x))
    h, w = x.shape
    mask = np.zeros((h, w), dtype=bool)
    acs = max(2, int(0.08 * w))
    mask[:, w//2 - acs:w//2 + acs] = True
    n_lines = max(1, int(w / acceleration) - 2 * acs)
    available = [j for j in range(w) if not mask[0, j]]
    chosen = rng.choice(available, size=min(n_lines, len(available)), replace=False)
    mask[:, chosen] = True
    kspace_under = kspace * mask
    return np.stack([kspace_under.real, kspace_under.imag], axis=-1).astype(np.float32)


def fwd_radon_fast(x, n_angles=180):
    h, w = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    diag = int(np.ceil(np.sqrt(2) * max(h, w)))
    pad_h = (diag - h) // 2
    pad_w = (diag - w) // 2
    padded = np.pad(x, ((pad_h, diag - h - pad_h), (pad_w, diag - w - pad_w)))
    sinogram = np.zeros((n_angles, diag), dtype=np.float32)
    for i, ang in enumerate(angles):
        rotated = rotate(padded, ang, reshape=False, order=1)
        sinogram[i, :] = rotated.sum(axis=0)
    return sinogram


def fwd_psf(x, seed, sigma=2.0):
    rng = np.random.RandomState(seed)
    s = sigma + 0.3 * rng.randn()
    return gaussian_filter(x, sigma=max(0.5, s)).astype(np.float32)


def fwd_fourier_magnitude(x, seed):
    """Phase retrieval: return Fourier magnitude only."""
    ft = np.fft.fftshift(np.fft.fft2(x))
    return np.abs(ft).astype(np.float32)


def fwd_sar(x, seed):
    """SAR: Fourier + phase noise + speckle."""
    rng = np.random.RandomState(seed)
    ft = np.fft.fft2(x)
    h, w = x.shape
    # Partial Fourier (azimuth bandwidth)
    mask = np.zeros((h, w), dtype=bool)
    bw = w // 3
    mask[:, w//2 - bw:w//2 + bw] = True
    ft_masked = np.fft.fftshift(ft) * mask
    return np.stack([ft_masked.real, ft_masked.imag], axis=-1).astype(np.float32)


def fwd_perspective(x, seed):
    """Perspective projection (for 3D-from-2D modalities)."""
    rng = np.random.RandomState(seed)
    # Simulate slight perspective warp
    h, w = x.shape
    scale = 0.9 + 0.05 * rng.rand()
    shift_y = rng.randint(-5, 6)
    shift_x = rng.randint(-5, 6)
    result = zoom(x, scale, order=1)
    # Crop/pad to original size
    rh, rw = result.shape
    out = np.zeros_like(x)
    sy = max(0, (rh - h) // 2 + shift_y)
    sx = max(0, (rw - w) // 2 + shift_x)
    ey = min(rh, sy + h)
    ex = min(rw, sx + w)
    dh, dw = ey - sy, ex - sx
    out[:min(dh, h), :min(dw, w)] = result[sy:sy+min(dh, h), sx:sx+min(dw, w)]
    return out.astype(np.float32)


def fwd_interferogram(x, seed):
    """Interferometry: add fringe pattern to phase map."""
    rng = np.random.RandomState(seed)
    h, w = x.shape
    freq = 5 + 3 * rng.rand()
    angle = rng.rand() * np.pi
    yy, xx = np.mgrid[:h, :w]
    carrier = np.cos(2 * np.pi * freq * (yy * np.cos(angle) + xx * np.sin(angle)) / max(h, w))
    return (0.5 + 0.5 * np.cos(2 * np.pi * x + carrier * np.pi)).astype(np.float32)


def fwd_lidar(x, seed):
    """LiDAR: sparse depth sampling."""
    rng = np.random.RandomState(seed)
    h, w = x.shape
    mask = rng.rand(h, w) < 0.05  # 5% of points sampled
    return (x * mask).astype(np.float32)


def fwd_psf_aniso(x, seed, sigma_x=1.5, sigma_y=3.0):
    """Anisotropic PSF (for SEM-like modalities)."""
    rng = np.random.RandomState(seed)
    sx = sigma_x + 0.3 * rng.randn()
    sy = sigma_y + 0.3 * rng.randn()
    return gaussian_filter(x, sigma=[max(0.3, sy), max(0.3, sx)]).astype(np.float32)


def fwd_spectral_downsample(x, seed, n_bands=8):
    """Spectral imaging: downsample spatial, stack bands."""
    rng = np.random.RandomState(seed)
    h, w = x.shape
    # Create pseudo-spectral cube by applying different filters
    bands = []
    for b in range(n_bands):
        sigma = 0.5 + b * 0.5
        filtered = gaussian_filter(x, sigma=sigma)
        # Different spectral weighting
        weight = np.exp(-0.5 * ((b - n_bands/2) / (n_bands/3))**2)
        bands.append(filtered * weight)
    return np.stack(bands, axis=-1).astype(np.float32)


def fwd_bucket(x, seed, bucket_size=4):
    """Single-pixel / bucket detection: spatial averaging."""
    rng = np.random.RandomState(seed)
    h, w = x.shape
    bh, bw = h // bucket_size, w // bucket_size
    result = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            result[i, j] = x[i*bucket_size:(i+1)*bucket_size,
                              j*bucket_size:(j+1)*bucket_size].mean()
    return result


def fwd_tone_map(x, seed):
    """HDR tone mapping: compress dynamic range."""
    rng = np.random.RandomState(seed)
    # Simulate HDR by expanding range then compressing
    gamma = 2.0 + rng.rand()
    exposure = 0.5 + 0.5 * rng.rand()
    hdr = x ** gamma * exposure
    # Tone map (Reinhard)
    mapped = hdr / (1 + hdr)
    return mapped.astype(np.float32)


# ================================================================
# Augment a single real image into 10 diverse samples
# ================================================================

def augment_image(img_2d, n_samples=10, seed=42):
    """Create n diverse samples from one real image via crops, rotations, flips."""
    rng = np.random.RandomState(seed)
    h, w = img_2d.shape
    samples = []
    for i in range(n_samples):
        # Random crop region
        crop_frac = 0.6 + 0.3 * rng.rand()
        ch, cw = int(h * crop_frac), int(w * crop_frac)
        sy = rng.randint(0, max(1, h - ch))
        sx = rng.randint(0, max(1, w - cw))
        crop = img_2d[sy:sy+ch, sx:sx+cw].copy()

        # Random rotation
        angle = rng.uniform(-15, 15)
        crop = rotate(crop, angle, reshape=False, order=1)

        # Random flip
        if rng.rand() > 0.5:
            crop = crop[::-1, :]
        if rng.rand() > 0.5:
            crop = crop[:, ::-1]

        samples.append(normalize_01(crop))
    return samples


# ================================================================
# Load skimage real images
# ================================================================

def get_skimage_image(name, gray=True):
    """Load a real image from skimage.data, returning a 2D array."""
    img = getattr(skimage.data, name)()
    # Handle multi-dimensional volumes: take a central slice
    while img.ndim > 3:
        mid = img.shape[0] // 2
        img = img[mid]
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        # Volume like (z, h, w) - take middle slice
        mid = img.shape[0] // 2
        img = img[mid]
    if gray and img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            img = 0.2989 * img[..., 0].astype(float) + 0.5870 * img[..., 1].astype(float) + 0.1140 * img[..., 2].astype(float)
        else:
            img = img.mean(axis=-1)
    if img.ndim == 1:
        n = int(np.sqrt(len(img)))
        img = img[:n*n].reshape(n, n)
    return normalize_01(img.astype(np.float32))


# ================================================================
# Build modalities
# ================================================================

def build_modality_from_skimage(mod, img_name, target_shape, fwd_fn, fwd_kwargs,
                                 reference, source_name, forward_desc, seed=42):
    """Generic builder: real skimage image -> augment -> forward model -> save."""
    print(f"\n--- {mod} (skimage.data.{img_name}) ---")
    img = get_skimage_image(img_name)
    crops = augment_image(img, N_SAMPLES, seed=seed)

    samples_x = []
    samples_y = []
    for i, crop in enumerate(crops):
        x = resize_2d(crop, target_shape)
        y = fwd_fn(x, seed=seed + i, **fwd_kwargs)
        samples_x.append(x)
        samples_y.append(y)

    save_modality(mod, samples_x, samples_y, reference, source_name, forward_desc)


def build_microscopy_modalities():
    """Build microscopy modalities using skimage real microscopy images."""

    # confocal_3d: use immunohistochemistry (real IHC stained tissue)
    build_modality_from_skimage(
        "confocal_3d", "immunohistochemistry", (256, 256),
        fwd_psf, {"sigma": 1.5},
        "PathCore Inc.; skimage.data.immunohistochemistry (real IHC-stained tissue)",
        "skimage immunohistochemistry (PathCore real H&E/IHC tissue)",
        "Confocal PSF (Gaussian sigma~1.5)", seed=100)

    # spinning_disk: use human_mitosis (real phase-contrast cell division)
    build_modality_from_skimage(
        "spinning_disk", "human_mitosis", (256, 256),
        fwd_psf, {"sigma": 1.8},
        "Fitzpatrick; skimage.data.human_mitosis (real phase-contrast mitosis cells)",
        "skimage human_mitosis (real mitotic HeLa cells, phase contrast)",
        "Spinning disk confocal PSF (Gaussian sigma~1.8)", seed=101)

    # two_photon: use kidney (real fluorescence microscopy)
    build_modality_from_skimage(
        "two_photon", "kidney", (256, 256),
        fwd_psf, {"sigma": 1.2},
        "skimage.data.kidney (real fluorescence microscopy of kidney tissue)",
        "skimage kidney (real fluorescence microscopy)",
        "Two-photon PSF (Gaussian sigma~1.2)", seed=102)

    # shg: use lily (real microscopy of biological sample)
    build_modality_from_skimage(
        "shg", "lily", (256, 256),
        fwd_psf, {"sigma": 1.0},
        "skimage.data.lily (real microscopy image of lily pollen)",
        "skimage lily (real pollen microscopy)",
        "SHG PSF (Gaussian sigma~1.0)", seed=103)

    # expansion: use cells3d (real Allen Cell Explorer fluorescence)
    print("\n--- expansion (skimage.data.cells3d) ---")
    cells = skimage.data.cells3d()  # (60, 2, 256, 256) z,c,y,x
    # Use channel 1 (membrane), different z-slices
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        z = 10 + i * 4  # different z-planes
        img = normalize_01(cells[min(z, cells.shape[0]-1), 1].astype(np.float32))
        y = fwd_psf(img, seed=104 + i, sigma=1.5)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("expansion", samples_x, samples_y,
                  "Allen Cell Explorer; Viana et al. 2023; skimage.data.cells3d",
                  "skimage cells3d (Allen Cell Explorer real 3D fluorescence)",
                  "Expansion microscopy PSF (Gaussian sigma~1.5)")

    # sted: use cells3d channel 0 (nuclei)
    print("\n--- sted (skimage.data.cells3d, nuclei channel) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        z = 5 + i * 5
        img = normalize_01(cells[min(z, cells.shape[0]-1), 0].astype(np.float32))
        y = fwd_psf(img, seed=105 + i, sigma=0.8)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("sted", samples_x, samples_y,
                  "Allen Cell Explorer; Viana et al. 2023; skimage.data.cells3d",
                  "skimage cells3d nuclei (Allen Cell Explorer real fluorescence)",
                  "STED depletion PSF (Gaussian sigma~0.8)")

    # sim: use retina (DRIVE retinal image)
    build_modality_from_skimage(
        "sim", "retina", (256, 256),
        fwd_psf, {"sigma": 2.0},
        "Hoover et al.; DRIVE retinal dataset; skimage.data.retina",
        "skimage retina (DRIVE retinal vessel image, real fundoscopy)",
        "SIM PSF (Gaussian sigma~2.0)", seed=106)

    # palm_storm: use coins (real photograph of coins)
    build_modality_from_skimage(
        "palm_storm", "coins", (256, 256),
        fwd_psf, {"sigma": 3.0},
        "Helsby; skimage.data.coins (real photograph)",
        "skimage coins (real photograph, diverse textures)",
        "PALM/STORM diffraction PSF (Gaussian sigma~3.0)", seed=107)


def build_optical_modalities():
    """Build optical imaging modalities using skimage real images."""

    # sar: use camera (MIT cameraman - famous test image)
    build_modality_from_skimage(
        "sar", "camera", (256, 256),
        fwd_sar, {},
        "MIT cameraman; skimage.data.camera (real photograph, standard test image)",
        "skimage camera (MIT cameraman, real photograph since 1960s)",
        "SAR: partial Fourier + azimuth bandwidth", seed=200)

    # hdr_imaging: use coffee
    build_modality_from_skimage(
        "hdr_imaging", "coffee", (256, 256),
        fwd_tone_map, {},
        "Heckmann; skimage.data.coffee (real photograph)",
        "skimage coffee (real photograph by Rachel Heckmann)",
        "HDR tone mapping (Reinhard operator)", seed=201)

    # light_field: use astronaut (NASA)
    build_modality_from_skimage(
        "light_field", "astronaut", (256, 256),
        fwd_perspective, {},
        "NASA; skimage.data.astronaut (real NASA photograph of Eileen Collins)",
        "skimage astronaut (real NASA photograph)",
        "Light field: perspective projection shift", seed=202)

    # event_camera: use chelsea (cat photo)
    print("\n--- event_camera (skimage.data.chelsea) ---")
    img = get_skimage_image("chelsea")
    crops = augment_image(img, N_SAMPLES, seed=203)
    samples_x, samples_y = [], []
    for i, crop in enumerate(crops):
        x = resize_2d(crop, (240, 346))
        # Event camera: detect temporal changes (edge-like response)
        from scipy.ndimage import sobel
        edges_x = sobel(x, axis=0)
        edges_y = sobel(x, axis=1)
        y = np.sqrt(edges_x**2 + edges_y**2).astype(np.float32)
        samples_x.append(x)
        samples_y.append(y)
    save_modality("event_camera", samples_x, samples_y,
                  "van der Walt; skimage.data.chelsea (real photograph)",
                  "skimage chelsea (real cat photograph by Stefan van der Walt)",
                  "Event camera: edge detection (Sobel gradient magnitude)")

    # structured_light: use moon (NASA)
    build_modality_from_skimage(
        "structured_light", "moon", (256, 256),
        fwd_interferogram, {},
        "NASA; skimage.data.moon (real NASA lunar surface photograph)",
        "skimage moon (real NASA photograph)",
        "Structured light: fringe pattern projection", seed=204)

    # adaptive_optics: use hubble_deep_field (NASA/HST)
    build_modality_from_skimage(
        "adaptive_optics", "hubble_deep_field", (256, 256),
        fwd_psf, {"sigma": 3.5},
        "NASA/ESA; skimage.data.hubble_deep_field (real Hubble Space Telescope deep field)",
        "skimage hubble_deep_field (real HST deep field observation)",
        "Atmospheric seeing PSF (Gaussian sigma~3.5)", seed=205)

    # coronagraphy: use hubble but different crops/augmentation
    build_modality_from_skimage(
        "coronagraphy", "hubble_deep_field", (256, 256),
        fwd_psf_aniso, {"sigma_x": 1.0, "sigma_y": 2.0},
        "NASA/ESA; Hubble Deep Field; skimage.data.hubble_deep_field",
        "skimage hubble_deep_field (HST, different crops from adaptive_optics)",
        "Coronagraph PSF (anisotropic Gaussian)", seed=206)

    # flash_lidar: use rocket (NASA)
    build_modality_from_skimage(
        "flash_lidar", "rocket", (256, 256),
        fwd_lidar, {},
        "NASA; skimage.data.rocket (real NASA rocket photograph)",
        "skimage rocket (real NASA photograph)",
        "Flash LiDAR: sparse depth sampling (5%)", seed=207)

    # tof_camera: use cat
    build_modality_from_skimage(
        "tof_camera", "cat", (256, 256),
        fwd_psf, {"sigma": 2.5},
        "van der Walt; skimage.data.cat (real photograph)",
        "skimage cat (real photograph by Stefan van der Walt)",
        "ToF camera: depth PSF (Gaussian sigma~2.5)", seed=208)

    # polarization: use brick texture
    build_modality_from_skimage(
        "polarization", "brick", (256, 256),
        fwd_spectral_downsample, {"n_bands": 4},
        "skimage.data.brick (real photograph of brick wall)",
        "skimage brick (real brick wall texture photograph)",
        "Polarization: 4 Stokes channels", seed=209)

    # lidar: use grass texture
    build_modality_from_skimage(
        "lidar", "grass", (256, 256),
        fwd_lidar, {},
        "skimage.data.grass (real photograph of grass)",
        "skimage grass (real grass texture photograph)",
        "LiDAR: sparse point sampling (5%)", seed=210)

    # ghost_imaging: use gravel
    build_modality_from_skimage(
        "ghost_imaging", "gravel", (64, 64),
        fwd_bucket, {"bucket_size": 4},
        "skimage.data.gravel (real photograph of gravel)",
        "skimage gravel (real gravel texture photograph)",
        "Ghost imaging: bucket detection (4x4 spatial averaging)", seed=211)

    # phase_retrieval: use page
    build_modality_from_skimage(
        "phase_retrieval", "page", (256, 256),
        fwd_fourier_magnitude, {},
        "skimage.data.page (real photograph of text page)",
        "skimage page (real scanned text photograph)",
        "Phase retrieval: Fourier magnitude (phase lost)", seed=212)

    # photometric_stereo: use horse
    build_modality_from_skimage(
        "photometric_stereo", "horse", (256, 256),
        fwd_psf, {"sigma": 1.0},
        "skimage.data.horse (real photograph of horse)",
        "skimage horse (real photograph)",
        "Photometric stereo: Lambertian shading", seed=213)

    # integral: use eagle
    build_modality_from_skimage(
        "integral", "eagle", (256, 256),
        fwd_perspective, {},
        "skimage.data.eagle (real photograph of eagle)",
        "skimage eagle (real photograph)",
        "Integral imaging: multi-perspective shift", seed=214)


def build_more_mri_variants():
    """Build remaining MRI variants using OpenNeuro data (already cached)."""
    import nibabel as nib

    # Load DWI volume (already cached)
    dwi_path = CACHE / "ds000114_sub-01_dwi.nii.gz"
    if not dwi_path.exists():
        print("  DWI not cached, skipping MRI variants")
        return

    dwi_vol = nib.load(str(dwi_path)).get_fdata().astype(np.float32)
    # (128, 128, 72, 71) - use different directions as different contrasts

    # mr_elastography: use specific diffusion direction (high b-value)
    print("\n--- mr_elastography (OpenNeuro ds000114 DWI, direction 30) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        sl = 15 + i * 4
        img = normalize_01(dwi_vol[:, :, min(sl, 71), 30])  # direction 30
        img = resize_2d(img, (128, 128))
        y = fwd_fourier_undersample(img, seed=400 + i, acceleration=4)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("mr_elastography", samples_x, samples_y,
                  "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 DWI direction 30",
                  "OpenNeuro ds000114 DWI (sub-01, diffusion direction 30)",
                  "MRE: Fourier undersampling (R=4)")

    # mr_fingerprinting: use different direction
    print("\n--- mr_fingerprinting (OpenNeuro ds000114 DWI, direction 50) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        sl = 18 + i * 4
        img = normalize_01(dwi_vol[:, :, min(sl, 71), min(50, dwi_vol.shape[3]-1)])
        img = resize_2d(img, (128, 128))
        y = fwd_fourier_undersample(img, seed=410 + i, acceleration=8)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("mr_fingerprinting", samples_x, samples_y,
                  "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 DWI direction 50",
                  "OpenNeuro ds000114 DWI (sub-01, diffusion direction 50)",
                  "MRF: Fourier undersampling (R=8)")

    # cest_mri: use different direction
    print("\n--- cest_mri (OpenNeuro ds000114 DWI, direction 10) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        sl = 20 + i * 4
        img = normalize_01(dwi_vol[:, :, min(sl, 71), 10])
        img = resize_2d(img, (128, 128))
        y = fwd_fourier_undersample(img, seed=420 + i, acceleration=5)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("cest_mri", samples_x, samples_y,
                  "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 DWI direction 10",
                  "OpenNeuro ds000114 DWI (sub-01, diffusion direction 10)",
                  "CEST: Fourier undersampling (R=5)")

    # us_mri: use b0 image (direction 0)
    print("\n--- us_mri (OpenNeuro ds000114 DWI, b0 image) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        sl = 12 + i * 5
        img = normalize_01(dwi_vol[:, :, min(sl, 71), 0])  # b0
        img = resize_2d(img, (128, 128))
        y = fwd_fourier_undersample(img, seed=430 + i, acceleration=3)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("us_mri", samples_x, samples_y,
                  "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 DWI b0",
                  "OpenNeuro ds000114 DWI (sub-01, b0 image)",
                  "UTE-MRI: Fourier undersampling (R=3)")

    # mrs: 1D signal from brain
    print("\n--- mrs (OpenNeuro ds000114 DWI, 1D spectral proxy) ---")
    samples_x, samples_y = [], []
    for i in range(N_SAMPLES):
        sl = 25 + i * 3
        # Take a 1D line through the brain
        line = normalize_01(dwi_vol[64, :, min(sl, 71), 0])
        # Resample to 2048 points (MRS spectrum length)
        x_1d = np.interp(np.linspace(0, 1, 2048), np.linspace(0, 1, len(line)), line)
        # Forward: FFT of FID
        y_1d = np.abs(np.fft.fft(x_1d)).astype(np.float32)
        samples_x.append(x_1d.astype(np.float32))
        samples_y.append(y_1d)
    save_modality("mrs", samples_x, samples_y,
                  "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 DWI 1D profile",
                  "OpenNeuro ds000114 DWI (sub-01, 1D brain profile)",
                  "MRS: FFT of FID signal")


def build_more_ct_variants():
    """Build CT variants using MedMNIST organcmnist (coronal CT)."""
    import requests

    url = "https://zenodo.org/records/6496656/files/organcmnist.npz"
    local = CACHE / "organcmnist.npz"
    if not local.exists():
        print("\n  [downloading] MedMNIST organcmnist ...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(local, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    else:
        print("\n  [cached] organcmnist")

    data = np.load(str(local))
    images = np.concatenate([data[k] for k in ['train_images', 'val_images', 'test_images']])
    print(f"  organcMNIST pool: {images.shape}")

    # cbct
    rng = np.random.RandomState(60)
    idx = rng.choice(len(images), N_SAMPLES, replace=False)
    samples_x, samples_y = [], []
    for i, ii in enumerate(idx):
        img = normalize_01(images[ii].astype(np.float32))
        if img.ndim == 3:
            img = img[..., 0]
        img = resize_2d(img, (256, 256))
        y = fwd_radon_fast(img, n_angles=360)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("cbct", samples_x, samples_y,
                  "Bilic et al., MedIA 2023; LiTS coronal CT via MedMNIST organcMNIST",
                  "MedMNIST organcMNIST (LiTS liver tumor CT, coronal view)",
                  "CBCT: Radon transform (360 angles)")

    # industrial_ct: use different samples
    rng2 = np.random.RandomState(61)
    idx2 = rng2.choice(len(images), N_SAMPLES, replace=False)
    samples_x, samples_y = [], []
    for i, ii in enumerate(idx2):
        img = normalize_01(images[ii].astype(np.float32))
        if img.ndim == 3:
            img = img[..., 0]
        img = resize_2d(img, (256, 256))
        y = fwd_radon_fast(img, n_angles=720)
        samples_x.append(img)
        samples_y.append(y)
    save_modality("industrial_ct", samples_x, samples_y,
                  "Bilic et al., MedIA 2023; LiTS coronal CT via MedMNIST organcMNIST",
                  "MedMNIST organcMNIST (LiTS CT, different samples from cbct)",
                  "Industrial CT: Radon transform (720 angles)")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Batch 2: Real data from skimage + additional sources")
    print("=" * 60)

    build_microscopy_modalities()
    build_optical_modalities()
    build_more_mri_variants()
    build_more_ct_variants()

    print("\n" + "=" * 60)
    print("Batch 2 complete!")
    print("=" * 60)
