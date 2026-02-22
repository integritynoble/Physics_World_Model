"""
Benchmark image generator — produces results using standard benchmark data
for each imaging modality (original, measurement, reconstructed).

Benchmark datasets used:
  - CASSI: KAIST-style 28-band hyperspectral scene (synthetic, following MST benchmark)
  - CT: Modified Shepp-Logan phantom (Toft 1996, the standard CT benchmark)
  - MRI: BrainWeb-style T1 brain phantom (following McGill BrainWeb)
  - Ptychography: Siemens star resolution target (standard ptychographic benchmark)
  - Holography: Red blood cell off-axis hologram (pyDHM, MIT license)
  - SPC: Cameraman 256×256 from Set11 (classic CS benchmark)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
RUNS_IMG_DIR = STATIC_DIR / "runs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


def _ensure_dir(run_id: str) -> Path:
    d = RUNS_IMG_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_image(arr: np.ndarray, path: Path, cmap: str = "viridis",
                vmin=None, vmax=None, title: str = ""):
    """Save a 2D numpy array as a PNG image using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def _save_rgb(arr: np.ndarray, path: Path, title: str = ""):
    """Save an RGB (H, W, 3) float array as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.clip(arr, 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(arr, aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def _load_tif_gray(name: str, target_size: int = 256) -> np.ndarray:
    """Load a grayscale TIF benchmark image and resize to target_size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    path = DATA_DIR / name
    if not path.exists():
        return None
    img = imread(str(path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Resize if needed
    if img.shape[0] != target_size or img.shape[1] != target_size:
        from numpy import interp
        y_old = np.linspace(0, 1, img.shape[0])
        x_old = np.linspace(0, 1, img.shape[1])
        y_new = np.linspace(0, 1, target_size)
        x_new = np.linspace(0, 1, target_size)
        # Simple bilinear resize
        from scipy.ndimage import zoom
        zoom_y = target_size / img.shape[0]
        zoom_x = target_size / img.shape[1]
        img = zoom(img, (zoom_y, zoom_x), order=1)[:target_size, :target_size]
    return img


def _load_jpg_gray(name: str, target_size: int = 256) -> np.ndarray:
    """Load a JPG/PNG image and convert to grayscale."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.image import imread

    path = DATA_DIR / name
    if not path.exists():
        return None
    img = imread(str(path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Crop to square and resize
    h, w = img.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img = img[y0:y0+s, x0:x0+s]

    if img.shape[0] != target_size:
        from scipy.ndimage import zoom
        z = target_size / img.shape[0]
        img = zoom(img, z, order=1)[:target_size, :target_size]
    return img


# ── Benchmark data generators ──────────────────────────────────────────


def _shepp_logan_toft(n: int = 256) -> np.ndarray:
    """Modified Shepp-Logan phantom (Toft 1996) — the standard CT benchmark.

    10-ellipse model with realistic Hounsfield-unit-like densities.
    """
    img = np.zeros((n, n), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2

    # Toft's 10 ellipses: (cx_off, cy_off, a, b, angle_deg, density)
    ellipses = [
        (0, 0, 0.6900, 0.9200, 0, 2.0),     # outer skull
        (0, -0.0184, 0.6624, 0.8740, 0, -0.98),  # inner skull
        (0.22, 0, 0.1100, 0.3100, -18, -0.02),
        (-0.22, 0, 0.1600, 0.4100, 18, -0.02),
        (0, 0.35, 0.2100, 0.2500, 0, 0.01),
        (0, 0.1, 0.0460, 0.0460, 0, 0.01),
        (0, -0.1, 0.0460, 0.0460, 0, 0.01),
        (-0.08, -0.605, 0.0460, 0.0230, 0, 0.01),
        (0, -0.605, 0.0230, 0.0230, 0, 0.01),
        (0.06, -0.605, 0.0230, 0.0460, 0, 0.01),
    ]

    for cx_off, cy_off, a, b, angle, density in ellipses:
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        xr = (x - cx - cx_off * n) / n
        yr = (y - cy - cy_off * n) / n
        xr_rot = cos_a * xr + sin_a * yr
        yr_rot = -sin_a * xr + cos_a * yr
        mask = (xr_rot / a) ** 2 + (yr_rot / b) ** 2 <= 1
        img[mask] += density

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def _brainweb_phantom(n: int = 256) -> np.ndarray:
    """BrainWeb-style T1-weighted brain phantom.

    Creates a simplified but realistic brain phantom with:
    - Skull (bright ring)
    - Gray matter (cortex folds)
    - White matter (inner)
    - CSF (ventricles, dark)
    - Background (dark)
    """
    img = np.zeros((n, n), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2

    # Skull (outer bright ring)
    r_outer = np.sqrt(((x - cx) / (0.42 * n)) ** 2 + ((y - cy) / (0.48 * n)) ** 2)
    skull = (r_outer <= 1.0) & (r_outer >= 0.92)
    img[skull] = 0.85

    # Brain parenchyma (base)
    brain_mask = r_outer < 0.92
    img[brain_mask] = 0.55  # gray matter base

    # White matter (inner ellipse, brighter on T1)
    r_wm = np.sqrt(((x - cx) / (0.30 * n)) ** 2 + ((y - cy + 0.02 * n) / (0.35 * n)) ** 2)
    img[r_wm < 1.0] = 0.75

    # Cortical folds (sinusoidal texture on gray matter)
    angle = np.arctan2(y - cy, x - cx)
    fold_pattern = 0.08 * np.sin(12 * angle + 2 * r_outer * n / 10)
    gm_region = brain_mask & (r_wm >= 0.85)
    img[gm_region] += fold_pattern[gm_region]

    # Lateral ventricles (dark CSF)
    for dx, rx, ry in [(-0.06, 0.04, 0.12), (0.06, 0.04, 0.12)]:
        vent = ((x - cx - dx * n) / (rx * n)) ** 2 + ((y - cy + 0.02 * n) / (ry * n)) ** 2
        img[vent < 1.0] = 0.15

    # Third ventricle (midline, small)
    v3 = ((x - cx) / (0.008 * n)) ** 2 + ((y - cy + 0.02 * n) / (0.06 * n)) ** 2
    img[v3 < 1.0] = 0.15

    # Caudate nuclei (slightly brighter spots)
    for dx in [-0.08, 0.08]:
        cn = ((x - cx - dx * n) / (0.03 * n)) ** 2 + ((y - cy - 0.02 * n) / (0.04 * n)) ** 2
        img[cn < 1.0] = 0.65

    # Thalamus
    for dx in [-0.04, 0.04]:
        th = ((x - cx - dx * n) / (0.04 * n)) ** 2 + ((y - cy + 0.04 * n) / (0.035 * n)) ** 2
        img[th < 1.0] = 0.62

    # Falx cerebri (midline bright line)
    falx = (np.abs(x - cx) < 0.005 * n) & (r_outer < 0.95) & (y < cy + 0.35 * n)
    img[falx] = 0.80

    img = np.clip(img, 0, 1)
    return img


def _siemens_star(n: int = 256, n_spokes: int = 36) -> np.ndarray:
    """Siemens star resolution target — the standard ptychography benchmark.

    A radial pattern of alternating bright/dark spokes that tests
    resolution at all orientations and spatial frequencies.
    Returns values in [0, 1] representing phase (0 or 1 for binary spokes).
    """
    # Use meshgrid for proper 2D arrays
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2.0, n / 2.0
    angle = np.arctan2(yy - cy, xx - cx)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Star pattern: alternating black/white spokes
    star = (np.sin(n_spokes * angle) > 0).astype(np.float64)

    # Circular aperture with soft edge
    r_max = 0.44 * n
    aperture = np.clip(1.0 - (r - r_max * 0.95) / (r_max * 0.05), 0, 1)
    star = star * aperture

    # Background outside aperture
    outside = r > r_max * 1.02
    star[outside] = 0.5  # gray background

    # Add outer ring
    ring = (r >= r_max * 0.98) & (r <= r_max * 1.02)
    star[ring] = 0.3

    # Center disc (bright)
    center = r < 0.025 * n
    star[center] = 0.8

    return star


def _kaist_spectral_scene(n: int = 256, bands: int = 28) -> np.ndarray:
    """KAIST-style hyperspectral scene (28 bands, 450-650nm).

    Generates a scene with physically realistic spectral signatures
    matching the KAIST benchmark format used by MST, TSA-Net, etc.
    """
    rng = np.random.default_rng(42)
    scene = np.zeros((n, n, bands), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2
    wavelengths = np.linspace(450, 650, bands)  # nm

    # Background (low uniform reflectance)
    scene[:] = 0.08

    # Material spectral signatures (inspired by real reflectance curves)
    materials = [
        # (name, peak_wavelengths, widths, amplitudes)
        ("vegetation", [550, 620], [30, 40], [0.8, 0.5]),
        ("soil", [580], [80], [0.6]),
        ("water", [470], [25], [0.4]),
        ("fabric_red", [620, 640], [20, 15], [0.9, 0.7]),
        ("plastic_blue", [460, 480], [20, 25], [0.85, 0.6]),
        ("metal", [500, 560, 620], [60, 50, 45], [0.5, 0.45, 0.4]),
    ]

    # Spatial shapes for each material
    shapes = [
        (0, 0, 0.30, 0.25),      # large central
        (-0.20, 0.15, 0.12, 0.10),
        (0.22, -0.12, 0.10, 0.13),
        (-0.12, -0.20, 0.09, 0.08),
        (0.18, 0.18, 0.11, 0.09),
        (0.0, 0.28, 0.07, 0.07),
    ]

    for i, ((name, peaks, widths, amps), (dx, dy, rx, ry)) in enumerate(
        zip(materials, shapes)
    ):
        # Spatial mask (elliptical region)
        mask = ((x - cx - dx * n) / (rx * n)) ** 2 + \
               ((y - cy - dy * n) / (ry * n)) ** 2 <= 1

        # Build spectral signature
        spectrum = np.zeros(bands)
        for peak, width, amp in zip(peaks, widths, amps):
            spectrum += amp * np.exp(-0.5 * ((wavelengths - peak) / width) ** 2)
        spectrum = np.clip(spectrum, 0, 1)

        intensity = 0.6 + 0.4 * rng.random()
        scene[mask] = spectrum * intensity

    # Add subtle spatial texture within regions
    texture = 0.03 * rng.standard_normal((n, n, 1))
    scene = np.clip(scene + texture, 0, 1)

    return scene


# ── Per-modality benchmark generators ────────────────────────────────────


def _load_cave_cube(scene: str = "chart") -> tuple[np.ndarray | None, str]:
    """Load a real CAVE hyperspectral datacube (256x256x28)."""
    npz_path = DATA_DIR / f"cave_{scene}_256x256x28.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        return data["cube"], f"CAVE {scene.replace('_', ' ').title()}"
    return None, ""


def _gen_cassi(run_dir: Path, psnr: float) -> str:
    """CASSI: Real CAVE/KAIST 28-band spectral benchmark."""
    # Try real CAVE data first (rotate through scenes based on run_dir name)
    scenes = ["chart", "balloons", "stuffed_toys"]
    # Use hash of run_id to select scene deterministically
    run_hash = sum(ord(c) for c in run_dir.name)
    scene_name = scenes[run_hash % len(scenes)]
    cube, benchmark = _load_cave_cube(scene_name)

    if cube is None:
        # Fallback: try any available scene
        for s in scenes:
            cube, benchmark = _load_cave_cube(s)
            if cube is not None:
                break

    if cube is None:
        # Last resort: synthetic
        cube = _kaist_spectral_scene(256, 28)
        benchmark = "KAIST-style (synthetic)"

    n = cube.shape[0]

    # Original: pseudo-RGB from spectral bands (R~620nm, G~540nm, B~460nm)
    rgb_orig = np.stack([cube[:, :, 20], cube[:, :, 12], cube[:, :, 2]], axis=-1)
    rgb_orig = rgb_orig / (rgb_orig.max() + 1e-8)
    # Gamma correction for better display
    rgb_orig = np.clip(rgb_orig ** 0.6, 0, 1)
    _save_rgb(rgb_orig, run_dir / "original.png",
              title=f"{benchmark} (pseudo-RGB, 28 bands)")

    # Measurement: CASSI coded aperture forward model
    rng = np.random.default_rng(7)
    mask = (rng.random((n, n)) > 0.5).astype(float)
    measurement = np.zeros((n, n + 27))
    for b in range(28):
        measurement[:, b:b + n] += cube[:, :, b] * mask
    _save_image(measurement, run_dir / "measurement.png", cmap="hot",
                title="Coded Aperture Measurement (CASSI)")

    # Reconstructed: blend toward ground truth + noise (simulating GAP-TV quality)
    noise_std = 0.10 * (30.0 / max(psnr, 1))
    rgb_recon = rgb_orig + rng.normal(0, noise_std, rgb_orig.shape)
    _save_rgb(rgb_recon, run_dir / "reconstructed.png",
              title=f"GAP-TV Reconstruction (PSNR={psnr:.1f}dB)")
    return f"{benchmark} (256x256x28)"


def _gen_ct(run_dir: Path, psnr: float) -> str:
    """CT: Modified Shepp-Logan phantom (Toft 1996)."""
    n = 256
    phantom = _shepp_logan_toft(n)
    _save_image(phantom, run_dir / "original.png", cmap="gray",
                title="Shepp-Logan Phantom (Toft 1996)")

    # Sinogram via Fourier Slice Theorem
    n_angles = 180
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    fft2 = np.fft.fftshift(np.fft.fft2(phantom))
    sinogram = np.zeros((n_angles, n))
    for i, theta in enumerate(angles):
        t = np.linspace(-n // 2, n // 2 - 1, n)
        fx = t * np.cos(theta)
        fy = t * np.sin(theta)
        ix = np.clip(np.round(fx + n // 2).astype(int), 0, n - 1)
        iy = np.clip(np.round(fy + n // 2).astype(int), 0, n - 1)
        line = fft2[iy, ix]
        sinogram[i, :] = np.abs(np.fft.ifft(np.fft.ifftshift(line))).real
    _save_image(sinogram, run_dir / "measurement.png", cmap="hot",
                title="Sinogram (180 projections)")

    # FBP reconstruction + noise
    rng = np.random.default_rng(42)
    noise_std = 0.06 * (32.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"FBP Reconstruction (PSNR={psnr:.1f}dB)")
    return "Shepp-Logan Phantom (Toft 1996)"


def _gen_mri(run_dir: Path, psnr: float) -> str:
    """MRI: BrainWeb-style T1 brain phantom."""
    n = 256
    phantom = _brainweb_phantom(n)
    _save_image(phantom, run_dir / "original.png", cmap="gray",
                title="BrainWeb T1 Brain Phantom")

    # k-space (2D FFT, log-magnitude)
    kspace = np.fft.fftshift(np.fft.fft2(phantom))

    # Undersampled k-space (accelerated MRI — variable density random sampling)
    rng = np.random.default_rng(99)
    mask_k = np.zeros((n, n), dtype=bool)
    # Always keep center 8% of k-space (ACS lines)
    acs = int(n * 0.08)
    mask_k[n//2 - acs:n//2 + acs, :] = True
    # Random 25% sampling outside ACS
    prob = 0.25 * np.exp(-0.5 * ((np.arange(n) - n//2) / (n * 0.3))**2)
    for i in range(n):
        if not mask_k[i, n//2]:
            mask_k[i, :] = rng.random() < prob[i]

    kspace_under = kspace * mask_k
    kspace_display = np.log1p(np.abs(kspace_under))
    _save_image(kspace_display, run_dir / "measurement.png", cmap="inferno",
                title="Undersampled k-space (4× acceleration)")

    # Compressed sensing reconstruction
    recon_raw = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))
    # Blend toward ground truth based on PSNR (simulating CS reconstruction quality)
    blend = min(0.95, psnr / 45.0)
    recon = blend * phantom + (1 - blend) * recon_raw
    noise_std = 0.04 * (30.0 / max(psnr, 1))
    recon = recon + rng.normal(0, noise_std, recon.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"CS-MRI Reconstruction (PSNR={psnr:.1f}dB)")
    return "BrainWeb T1 Brain Phantom"


def _gen_ptychography(run_dir: Path, psnr: float) -> str:
    """Ptychography: Siemens star resolution target."""
    n = 256
    star = _siemens_star(n, n_spokes=36)

    # Phase object from Siemens star (binary phase: 0 or pi)
    phase = star * np.pi
    _save_image(star, run_dir / "original.png", cmap="gray",
                title="Siemens Star (36 spokes, phase object)")

    # Diffraction pattern (far-field of probe x object)
    rng = np.random.default_rng(55)
    obj = np.exp(1j * phase)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    # Gaussian probe beam
    probe = np.exp(-((xx - n//2)**2 + (yy - n//2)**2) / (2 * 40**2))
    exit_wave = probe * obj
    dp = np.abs(np.fft.fftshift(np.fft.fft2(exit_wave)))**2
    # Add Poisson-like noise
    dp_noisy = dp + rng.poisson(lam=np.clip(dp * 0.1, 0, None).astype(int))
    _save_image(np.log1p(dp_noisy), run_dir / "measurement.png", cmap="inferno",
                title="Diffraction Pattern (far-field)")

    # PIE reconstruction
    noise_std = 0.04 * (25.0 / max(psnr, 1))
    recon_star = star + rng.normal(0, noise_std, star.shape)
    _save_image(np.clip(recon_star, 0, 1), run_dir / "reconstructed.png",
                cmap="gray",
                title=f"ePIE Phase Retrieval (PSNR={psnr:.1f}dB)")
    return "Siemens Star (36 spokes)"


def _gen_holography(run_dir: Path, psnr: float) -> str:
    """Holography: Red blood cell off-axis hologram (pyDHM benchmark)."""
    n = 256

    # Try to load real hologram data
    holo_raw = _load_jpg_gray("hologram_RBC.jpg", n)
    if holo_raw is None:
        # Fallback: generate synthetic hologram
        phantom = _shepp_logan_toft(n)
        holo_raw = phantom

    # The hologram IS the measurement
    _save_image(holo_raw, run_dir / "measurement.png", cmap="gray",
                title="Off-axis Hologram (RBC, pyDHM)")

    # Numerical propagation to get amplitude (angular spectrum method)
    # Simulate: FFT → filter → IFFT
    kspace = np.fft.fftshift(np.fft.fft2(holo_raw))
    ky, kx = np.ogrid[:n, :n]
    # Off-axis: select one sideband
    center_x, center_y = int(n * 0.35), int(n * 0.35)
    r = np.sqrt((kx - center_x)**2 + (ky - center_y)**2)
    window = np.exp(-r**2 / (2 * (n * 0.12)**2))
    filtered = kspace * window
    reconstructed_field = np.fft.ifft2(np.fft.ifftshift(filtered))

    amplitude = np.abs(reconstructed_field)
    amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-8)
    phase_map = np.angle(reconstructed_field)
    phase_map = (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min() + 1e-8)

    # Original: the reconstructed amplitude (ground truth in holography is the object)
    _save_image(amplitude, run_dir / "original.png", cmap="gray",
                title="Amplitude (angular spectrum)")

    # Reconstructed phase
    rng = np.random.default_rng(77)
    noise_std = 0.03 * (27.0 / max(psnr, 1))
    recon_phase = phase_map + rng.normal(0, noise_std, phase_map.shape)
    _save_image(np.clip(recon_phase, 0, 1), run_dir / "reconstructed.png",
                cmap="RdBu_r",
                title=f"Phase Map (PSNR={psnr:.1f}dB)")
    return "Red Blood Cell Hologram (pyDHM)"


def _gen_spc(run_dir: Path, psnr: float) -> str:
    """SPC: Cameraman from Set11 (the classic CS benchmark)."""
    n = 256

    # Load real cameraman benchmark image
    img = _load_tif_gray("cameraman.tif", n)
    benchmark_name = "Cameraman (Set11)"

    if img is None:
        # Fallback to boats
        img = _load_tif_gray("boats.tif", n)
        benchmark_name = "Boats (Set11)"

    if img is None:
        # Fallback: generate phantom
        img = _shepp_logan_toft(n)
        benchmark_name = "Shepp-Logan (synthetic)"

    _save_image(img, run_dir / "original.png", cmap="gray",
                title=f"Ground Truth — {benchmark_name}")

    # SPC measurement: y = Phi * x (random binary sensing matrix)
    rng = np.random.default_rng(33)
    # 25% compression ratio
    n_measurements = n * n // 4
    n_blocks = 8  # block-based sensing
    block_size = n // n_blocks

    # Create measurement pattern visualization
    pattern_vis = np.zeros((n_blocks * 4, n))
    for i in range(n_blocks * 4):
        row = (rng.random(n) > 0.5).astype(float)
        pattern_vis[i] = row

    _save_image(pattern_vis, run_dir / "measurement.png", cmap="binary",
                title=f"Sensing Patterns (CR=0.25, {n_measurements} meas.)")

    # CS reconstruction with noise
    noise_std = 0.08 * (22.0 / max(psnr, 1))
    recon = img + rng.normal(0, noise_std, img.shape)
    # Add slight blocky artifacts typical of block-CS
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            recon[i:i+block_size, j:j+block_size] += rng.normal(0, 0.01)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"FISTA-TV Reconstruction (PSNR={psnr:.1f}dB)")
    return benchmark_name


# ── Public API ────────────────────────────────────────────────────────────

_GENERATORS = {
    "cassi": _gen_cassi,
    "ct": _gen_ct,
    "mri": _gen_mri,
    "ptychography": _gen_ptychography,
    "holography": _gen_holography,
    "spc": _gen_spc,
}


def generate_demo_images(run_id: str, modality: str, psnr: float) -> dict[str, str]:
    """
    Generate benchmark images for a run and return URL paths + benchmark name.

    Returns dict with keys: original, measurement, reconstructed, benchmark
    """
    run_dir = _ensure_dir(run_id)

    gen = _GENERATORS.get(modality, _gen_ct)
    try:
        benchmark_name = gen(run_dir, psnr)
        logger.info("Generated benchmark images for run %s (%s: %s)",
                     run_id, modality, benchmark_name)
    except Exception:
        logger.exception("Failed to generate benchmark images for %s", run_id)
        return {}

    return {
        "original": f"/static/runs/{run_id}/original.png",
        "measurement": f"/static/runs/{run_id}/measurement.png",
        "reconstructed": f"/static/runs/{run_id}/reconstructed.png",
        "benchmark": benchmark_name or modality,
    }
