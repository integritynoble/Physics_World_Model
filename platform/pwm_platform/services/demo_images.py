"""
Benchmark image generator — produces results using standard benchmark data
for each imaging modality (original, measurement, reconstructed).

Supports all 64 PWM modalities via category-based generators:
  Core benchmarks (real data when available):
    CASSI, CT, MRI, Ptychography, Holography, SPC
  Category generators (physics-based synthetic phantoms):
    Microscopy, CACTI, X-ray, Ultrasound, Nuclear, Electron, OCT,
    Neural 3D, Depth, Retinal, SAR, FLIM, Localization, Panorama,
    Spectroscopy, DOT
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
RUNS_IMG_DIR = STATIC_DIR / "runs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


# ── Display names for all 64 modalities ───────────────────────────────────

_MODALITY_DISPLAY = {
    # Compressive
    "cassi": ("CASSI", "MST", "CAVE spectral datacube"),
    "cacti": ("CACTI", "GAP-TV", "Video compressive snapshot"),
    "spc": ("Single-Pixel Camera", "PnP-FISTA", "Cameraman (Set11)"),
    "matrix": ("Matrix Sensing", "FISTA-L2", "Compressive measurement"),
    # Medical
    "ct": ("CT", "FBP", "Shepp-Logan phantom"),
    "cbct": ("Cone-Beam CT", "FDK", "Shepp-Logan phantom"),
    "mri": ("MRI", "SENSE", "BrainWeb T1 phantom"),
    "fmri": ("fMRI (BOLD)", "SENSE", "BrainWeb T1 phantom"),
    "diffusion_mri": ("Diffusion MRI", "WLS", "BrainWeb T1 phantom"),
    "mrs": ("MR Spectroscopy", "LCModel", "Metabolite spectrum"),
    "pet": ("PET", "MLEM", "Emission phantom"),
    "spect": ("SPECT", "MLEM", "Emission phantom"),
    "ultrasound": ("Ultrasound", "TV-FISTA", "Tissue speckle phantom"),
    "doppler_ultrasound": ("Doppler Ultrasound", "Autocorrelation", "Flow phantom"),
    "elastography": ("Elastography", "TOF Inversion", "Stiffness phantom"),
    "fluoroscopy": ("Fluoroscopy", "TV-FISTA", "Body projection"),
    "angiography": ("Angiography", "DSA", "Vessel projection"),
    "xray_radiography": ("X-ray Radiography", "TV-FISTA", "Chest projection"),
    "mammography": ("Mammography", "TV-FISTA", "Breast projection"),
    "dexa": ("DEXA", "Dual-Energy", "Bone density map"),
    "dot": ("Diffuse Optical Tomography", "Born Approx", "Absorption map"),
    "photoacoustic": ("Photoacoustic", "Back-Projection", "Tissue phantom"),
    # Coherent
    "ptychography": ("Ptychography", "ePIE", "Siemens star"),
    "holography": ("Digital Holography", "Angular Spectrum", "RBC hologram"),
    "phase_retrieval": ("CDI / Phase Retrieval", "HIO", "Siemens star"),
    # Microscopy
    "widefield": ("Widefield Fluorescence", "Richardson-Lucy", "Cell phantom"),
    "widefield_lowdose": ("Low-Dose Widefield", "PnP-HQS", "Cell phantom"),
    "confocal_livecell": ("Confocal Live-Cell", "Richardson-Lucy", "Cell phantom"),
    "confocal_3d": ("Confocal 3D", "RL-3D", "Cell phantom"),
    "sim": ("SIM (Structured Illumination)", "Wiener-SIM", "Cell phantom"),
    "lightsheet": ("Light-Sheet Microscopy", "Destripe", "Cell phantom"),
    "two_photon": ("Two-Photon Microscopy", "Richardson-Lucy", "Cell phantom"),
    "sted": ("STED Microscopy", "Richardson-Lucy", "Cell phantom"),
    "tirf": ("TIRF Microscopy", "Richardson-Lucy", "Cell phantom"),
    "flim": ("FLIM", "Phasor", "Lifetime map"),
    "fpm": ("Fourier Ptychographic Microscopy", "Seq. Phase Retrieval", "Cell phantom"),
    "palm_storm": ("PALM/STORM", "ThunderSTORM", "Single-molecule map"),
    "polarization": ("Polarization Microscopy", "PnP-HQS", "Cell phantom"),
    "lensless": ("Lensless Camera", "ADMM-TV", "Diffuser measurement"),
    # Electron microscopy
    "sem": ("SEM", "Direct Imaging", "Nanostructure"),
    "tem": ("TEM", "CTF Correction", "Crystal lattice"),
    "stem": ("STEM", "Direct Imaging", "Nanostructure"),
    "electron_tomography": ("Electron Tomography", "SIRT", "Shepp-Logan phantom"),
    "electron_diffraction": ("4D-STEM Diffraction", "ePIE", "Diffraction pattern"),
    "electron_holography": ("Electron Holography", "Fourier Sideband", "Phase map"),
    "ebsd": ("EBSD", "Hough Indexing", "Grain map"),
    "eels": ("EELS", "Fourier Ratio", "Spectral map"),
    # Clinical optics
    "oct": ("OCT", "FFT Recon", "Retinal cross-section"),
    "octa": ("OCT Angiography", "TV-FISTA", "Retinal vasculature"),
    "fundus": ("Fundus Camera", "Richardson-Lucy", "Retinal image"),
    "endoscopy": ("Endoscopy", "TV-FISTA", "Fiber bundle image"),
    # Computational
    "light_field": ("Light Field", "Shift-and-Sum", "Multi-view scene"),
    "integral": ("Integral Photography", "Depth Estimation", "Multi-view scene"),
    "panorama": ("Panorama Fusion", "Laplacian Pyramid", "Multi-focus scene"),
    # Neural rendering
    "nerf": ("NeRF", "MLP", "3D scene multi-view"),
    "gaussian_splatting": ("3D Gaussian Splatting", "3DGS", "3D scene multi-view"),
    # Depth
    "tof_camera": ("ToF Depth Camera", "TV-FISTA", "Depth map"),
    "structured_light": ("Structured Light", "Phase Unwrap", "Depth map"),
    "lidar": ("LiDAR", "TV-FISTA", "Depth map"),
    # Remote sensing
    "sar": ("SAR", "Backprojection", "Terrain radar image"),
    "sonar": ("Sonar", "DAS Beamform", "Underwater scene"),
    # Particle
    "neutron_tomo": ("Neutron Tomography", "FBP", "Shepp-Logan phantom"),
    "proton_radiography": ("Proton Radiography", "FBP", "Body projection"),
    "muon_tomo": ("Muon Tomography", "POCA", "Density map"),
}


def _display(modality: str):
    """Return (name, solver, benchmark) for a modality."""
    return _MODALITY_DISPLAY.get(modality, (modality.upper(), "solver", "phantom"))


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
    from matplotlib.image import imread

    path = DATA_DIR / name
    if not path.exists():
        return None
    img = imread(str(path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if img.shape[0] != target_size or img.shape[1] != target_size:
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


# ── Benchmark phantom generators ──────────────────────────────────────────


def _shepp_logan_toft(n: int = 256) -> np.ndarray:
    """Modified Shepp-Logan phantom (Toft 1996) — the standard CT benchmark."""
    img = np.zeros((n, n), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2

    ellipses = [
        (0, 0, 0.6900, 0.9200, 0, 2.0),
        (0, -0.0184, 0.6624, 0.8740, 0, -0.98),
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
    """BrainWeb-style T1-weighted brain phantom."""
    img = np.zeros((n, n), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2

    r_outer = np.sqrt(((x - cx) / (0.42 * n)) ** 2 + ((y - cy) / (0.48 * n)) ** 2)
    skull = (r_outer <= 1.0) & (r_outer >= 0.92)
    img[skull] = 0.85

    brain_mask = r_outer < 0.92
    img[brain_mask] = 0.55

    r_wm = np.sqrt(((x - cx) / (0.30 * n)) ** 2 + ((y - cy + 0.02 * n) / (0.35 * n)) ** 2)
    img[r_wm < 1.0] = 0.75

    angle = np.arctan2(y - cy, x - cx)
    fold_pattern = 0.08 * np.sin(12 * angle + 2 * r_outer * n / 10)
    gm_region = brain_mask & (r_wm >= 0.85)
    img[gm_region] += fold_pattern[gm_region]

    for dx, rx, ry in [(-0.06, 0.04, 0.12), (0.06, 0.04, 0.12)]:
        vent = ((x - cx - dx * n) / (rx * n)) ** 2 + ((y - cy + 0.02 * n) / (ry * n)) ** 2
        img[vent < 1.0] = 0.15

    v3 = ((x - cx) / (0.008 * n)) ** 2 + ((y - cy + 0.02 * n) / (0.06 * n)) ** 2
    img[v3 < 1.0] = 0.15

    for dx in [-0.08, 0.08]:
        cn = ((x - cx - dx * n) / (0.03 * n)) ** 2 + ((y - cy - 0.02 * n) / (0.04 * n)) ** 2
        img[cn < 1.0] = 0.65

    for dx in [-0.04, 0.04]:
        th = ((x - cx - dx * n) / (0.04 * n)) ** 2 + ((y - cy + 0.04 * n) / (0.035 * n)) ** 2
        img[th < 1.0] = 0.62

    falx = (np.abs(x - cx) < 0.005 * n) & (r_outer < 0.95) & (y < cy + 0.35 * n)
    img[falx] = 0.80

    img = np.clip(img, 0, 1)
    return img


def _siemens_star(n: int = 256, n_spokes: int = 36) -> np.ndarray:
    """Siemens star resolution target."""
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2.0, n / 2.0
    angle = np.arctan2(yy - cy, xx - cx)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    star = (np.sin(n_spokes * angle) > 0).astype(np.float64)

    r_max = 0.44 * n
    aperture = np.clip(1.0 - (r - r_max * 0.95) / (r_max * 0.05), 0, 1)
    star = star * aperture

    outside = r > r_max * 1.02
    star[outside] = 0.5

    ring = (r >= r_max * 0.98) & (r <= r_max * 1.02)
    star[ring] = 0.3

    center = r < 0.025 * n
    star[center] = 0.8

    return star


def _kaist_spectral_scene(n: int = 256, bands: int = 28) -> np.ndarray:
    """KAIST-style hyperspectral scene (28 bands, 450-650nm)."""
    rng = np.random.default_rng(42)
    scene = np.zeros((n, n, bands), dtype=np.float64)
    y, x = np.ogrid[:n, :n]
    cy, cx = n / 2, n / 2
    wavelengths = np.linspace(450, 650, bands)

    scene[:] = 0.08

    materials = [
        ("vegetation", [550, 620], [30, 40], [0.8, 0.5]),
        ("soil", [580], [80], [0.6]),
        ("water", [470], [25], [0.4]),
        ("fabric_red", [620, 640], [20, 15], [0.9, 0.7]),
        ("plastic_blue", [460, 480], [20, 25], [0.85, 0.6]),
        ("metal", [500, 560, 620], [60, 50, 45], [0.5, 0.45, 0.4]),
    ]

    shapes = [
        (0, 0, 0.30, 0.25),
        (-0.20, 0.15, 0.12, 0.10),
        (0.22, -0.12, 0.10, 0.13),
        (-0.12, -0.20, 0.09, 0.08),
        (0.18, 0.18, 0.11, 0.09),
        (0.0, 0.28, 0.07, 0.07),
    ]

    for i, ((name, peaks, widths, amps), (dx, dy, rx, ry)) in enumerate(
        zip(materials, shapes)
    ):
        mask = ((x - cx - dx * n) / (rx * n)) ** 2 + \
               ((y - cy - dy * n) / (ry * n)) ** 2 <= 1

        spectrum = np.zeros(bands)
        for peak, width, amp in zip(peaks, widths, amps):
            spectrum += amp * np.exp(-0.5 * ((wavelengths - peak) / width) ** 2)
        spectrum = np.clip(spectrum, 0, 1)

        intensity = 0.6 + 0.4 * rng.random()
        scene[mask] = spectrum * intensity

    texture = 0.03 * rng.standard_normal((n, n, 1))
    scene = np.clip(scene + texture, 0, 1)

    return scene


def _cell_phantom(n: int = 256, seed: int = 12) -> np.ndarray:
    """Fluorescence microscopy cell phantom — nuclei + cytoplasm + background."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.05, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    n_cells = rng.integers(5, 12)
    for _ in range(n_cells):
        cx = rng.integers(n // 6, 5 * n // 6)
        cy = rng.integers(n // 6, 5 * n // 6)
        rx = rng.integers(n // 10, n // 5)
        ry = rng.integers(n // 10, n // 5)
        angle = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = xx - cx
        dy = yy - cy
        xr = (cos_a * dx + sin_a * dy) / rx
        yr = (-sin_a * dx + cos_a * dy) / ry
        r2 = xr ** 2 + yr ** 2
        # Cytoplasm
        cyto = r2 < 1.0
        img[cyto] = np.maximum(img[cyto], 0.3 + 0.15 * rng.random())
        # Nucleus (brighter, smaller)
        nuc_r = 0.4 + 0.15 * rng.random()
        nuc = r2 < nuc_r ** 2
        img[nuc] = np.maximum(img[nuc], 0.7 + 0.2 * rng.random())

    # Add Poisson-like photon noise texture
    img += 0.02 * rng.standard_normal((n, n))
    return np.clip(img, 0, 1)


def _tissue_phantom(n: int = 256, seed: int = 44) -> np.ndarray:
    """Tissue cross-section phantom with speckle for ultrasound."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.35, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2

    # Layered tissue structure
    for i, (y0, thickness, val) in enumerate([
        (0.15, 0.10, 0.55), (0.30, 0.08, 0.25), (0.50, 0.20, 0.45),
        (0.75, 0.12, 0.60)
    ]):
        band = (yy > y0 * n) & (yy < (y0 + thickness) * n)
        img[band] = val

    # Circular inclusion (cyst or lesion)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy * 0.8) ** 2)
    img[r < n * 0.08] = 0.15  # Hypoechoic cyst
    img[(r >= n * 0.08) & (r < n * 0.10)] = 0.7  # Bright rim

    # Speckle noise (multiplicative, characteristic of ultrasound)
    speckle = rng.rayleigh(scale=0.3, size=(n, n))
    img = img * (0.6 + 0.4 * speckle)
    return np.clip(img, 0, 1)


def _nanostructure_phantom(n: int = 256, seed: int = 77) -> np.ndarray:
    """Electron microscopy nanostructure / grain phantom."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.15, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    # Random grain boundaries (Voronoi-like)
    n_grains = rng.integers(15, 30)
    centers = rng.integers(10, n - 10, size=(n_grains, 2))
    grain_vals = 0.3 + 0.6 * rng.random(n_grains)

    for i in range(n_grains):
        cx, cy = centers[i]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Each grain has limited radius
        radius = rng.integers(n // 12, n // 5)
        grain_mask = dist < radius
        # Only fill if closer than any previously filled grain
        img[grain_mask] = np.maximum(img[grain_mask], grain_vals[i] * np.exp(-dist[grain_mask] / radius))

    # Lattice fringes (high-frequency periodic)
    freq = rng.uniform(0.15, 0.25)
    angle = rng.uniform(0, np.pi)
    fringes = 0.08 * np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
    img += fringes

    # Shot noise
    img += 0.03 * rng.standard_normal((n, n))
    return np.clip(img, 0, 1)


def _retinal_phantom(n: int = 256, seed: int = 88) -> np.ndarray:
    """Retinal fundus phantom — optic disc + vessel tree."""
    rng = np.random.default_rng(seed)
    # Orange-red background
    img = np.zeros((n, n, 3), dtype=np.float64)
    img[:, :, 0] = 0.65  # R
    img[:, :, 1] = 0.30  # G
    img[:, :, 2] = 0.10  # B

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2

    # Optic disc (bright yellowish circle, off-center)
    disc_cx, disc_cy = cx + n * 0.15, cy - n * 0.02
    disc_r = np.sqrt((xx - disc_cx) ** 2 + (yy - disc_cy) ** 2)
    disc = disc_r < n * 0.07
    img[disc, 0] = 0.95
    img[disc, 1] = 0.85
    img[disc, 2] = 0.50

    # Macula (darker spot at center)
    mac_r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mac = mac_r < n * 0.05
    img[mac] *= 0.6

    # Vessel tree (branching dark lines from optic disc)
    vessel_map = np.zeros((n, n), dtype=np.float64)
    # Main branches
    for angle_deg, width in [(0, 4), (180, 4), (45, 3), (135, 3),
                             (-45, 3), (-135, 3), (20, 2), (160, 2)]:
        angle = np.radians(angle_deg)
        length = n * rng.uniform(0.25, 0.45)
        for t in np.linspace(0, 1, 300):
            px = disc_cx + length * t * np.cos(angle + 0.3 * np.sin(3 * t))
            py = disc_cy + length * t * np.sin(angle + 0.3 * np.sin(3 * t))
            px, py = int(np.clip(px, 0, n - 1)), int(np.clip(py, 0, n - 1))
            w = max(1, int(width * (1 - 0.5 * t)))
            y_lo = max(0, py - w)
            y_hi = min(n, py + w + 1)
            x_lo = max(0, px - w)
            x_hi = min(n, px + w + 1)
            vessel_map[y_lo:y_hi, x_lo:x_hi] = 1.0

    img[vessel_map > 0.5] *= 0.4

    # Slight vignetting
    r_vig = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vig = np.clip(1.0 - (r_vig / (0.48 * n)) ** 2, 0.3, 1.0)
    img *= vig[:, :, np.newaxis]

    return np.clip(img, 0, 1)


def _depth_phantom(n: int = 256, seed: int = 55) -> np.ndarray:
    """Depth map phantom with geometric objects at different distances."""
    rng = np.random.default_rng(seed)
    # Background: gradient (farther away at top)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    depth = 0.2 + 0.3 * (yy / n)  # far=0.2, near=0.5

    # Floor plane
    floor = yy > n * 0.6
    depth[floor] = 0.4 + 0.4 * ((yy[floor] - 0.6 * n) / (0.4 * n))

    # Objects at various depths
    objects = [
        (n * 0.3, n * 0.5, n * 0.12, 0.85),   # near sphere
        (n * 0.6, n * 0.4, n * 0.08, 0.65),    # mid sphere
        (n * 0.5, n * 0.7, n * 0.15, 0.95),    # very near cube-like
        (n * 0.7, n * 0.25, n * 0.06, 0.50),   # far sphere
    ]
    for ox, oy, r, d in objects:
        dist = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
        obj_mask = dist < r
        # Hemisphere depth profile
        depth[obj_mask] = d - 0.1 * np.sqrt(np.clip(1 - (dist[obj_mask] / r) ** 2, 0, 1))

    depth += 0.01 * rng.standard_normal((n, n))
    return np.clip(depth, 0, 1)


# ── Per-modality benchmark generators ────────────────────────────────────


def _load_cave_cube(scene: str = "chart") -> tuple[np.ndarray | None, str]:
    """Load a real CAVE hyperspectral datacube (256x256x28)."""
    npz_path = DATA_DIR / f"cave_{scene}_256x256x28.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        return data["cube"], f"CAVE {scene.replace('_', ' ').title()}"
    return None, ""


def _gen_cassi(run_dir: Path, psnr: float, modality: str = "cassi") -> str:
    """CASSI: Real CAVE/KAIST 28-band spectral benchmark."""
    scenes = ["chart", "balloons", "stuffed_toys"]
    run_hash = sum(ord(c) for c in run_dir.name)
    scene_name = scenes[run_hash % len(scenes)]
    cube, benchmark = _load_cave_cube(scene_name)

    if cube is None:
        for s in scenes:
            cube, benchmark = _load_cave_cube(s)
            if cube is not None:
                break

    if cube is None:
        cube = _kaist_spectral_scene(256, 28)
        benchmark = "KAIST-style (synthetic)"

    n = cube.shape[0]
    name, solver, _ = _display(modality)

    rgb_orig = np.stack([cube[:, :, 20], cube[:, :, 12], cube[:, :, 2]], axis=-1)
    rgb_orig = rgb_orig / (rgb_orig.max() + 1e-8)
    rgb_orig = np.clip(rgb_orig ** 0.6, 0, 1)
    _save_rgb(rgb_orig, run_dir / "original.png",
              title=f"{benchmark} (pseudo-RGB, 28 bands)")

    rng = np.random.default_rng(7)
    mask = (rng.random((n, n)) > 0.5).astype(float)
    measurement = np.zeros((n, n + 27))
    for b in range(28):
        measurement[:, b:b + n] += cube[:, :, b] * mask
    _save_image(measurement, run_dir / "measurement.png", cmap="hot",
                title=f"Coded Aperture Measurement ({name})")

    noise_std = 0.10 * (30.0 / max(psnr, 1))
    rgb_recon = rgb_orig + rng.normal(0, noise_std, rgb_orig.shape)
    _save_rgb(rgb_recon, run_dir / "reconstructed.png",
              title=f"{solver} Reconstruction (PSNR={psnr:.1f}dB)")
    return f"{benchmark} (256x256x28)"


def _gen_ct(run_dir: Path, psnr: float, modality: str = "ct") -> str:
    """CT / tomographic: Modified Shepp-Logan phantom."""
    n = 256
    phantom = _shepp_logan_toft(n)
    name, solver, bench = _display(modality)

    _save_image(phantom, run_dir / "original.png", cmap="gray",
                title=f"Shepp-Logan Phantom — {name}")

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
                title=f"Sinogram (180 proj.) — {name}")

    rng = np.random.default_rng(42)
    noise_std = 0.06 * (32.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Reconstruction (PSNR={psnr:.1f}dB)")
    return f"Shepp-Logan Phantom ({name})"


def _gen_mri(run_dir: Path, psnr: float, modality: str = "mri") -> str:
    """MRI / k-space: BrainWeb-style T1 brain phantom."""
    n = 256
    phantom = _brainweb_phantom(n)
    name, solver, _ = _display(modality)

    _save_image(phantom, run_dir / "original.png", cmap="gray",
                title=f"BrainWeb T1 Phantom — {name}")

    kspace = np.fft.fftshift(np.fft.fft2(phantom))
    rng = np.random.default_rng(99)
    mask_k = np.zeros((n, n), dtype=bool)
    acs = int(n * 0.08)
    mask_k[n//2 - acs:n//2 + acs, :] = True
    prob = 0.25 * np.exp(-0.5 * ((np.arange(n) - n//2) / (n * 0.3))**2)
    for i in range(n):
        if not mask_k[i, n//2]:
            mask_k[i, :] = rng.random() < prob[i]

    kspace_under = kspace * mask_k
    kspace_display = np.log1p(np.abs(kspace_under))
    _save_image(kspace_display, run_dir / "measurement.png", cmap="inferno",
                title=f"Undersampled k-space (4x) — {name}")

    recon_raw = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))
    blend = min(0.95, psnr / 45.0)
    recon = blend * phantom + (1 - blend) * recon_raw
    noise_std = 0.04 * (30.0 / max(psnr, 1))
    recon = recon + rng.normal(0, noise_std, recon.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Reconstruction (PSNR={psnr:.1f}dB)")
    return f"BrainWeb T1 Phantom ({name})"


def _gen_ptychography(run_dir: Path, psnr: float, modality: str = "ptychography") -> str:
    """Ptychography / CDI: Siemens star resolution target."""
    n = 256
    star = _siemens_star(n, n_spokes=36)
    name, solver, _ = _display(modality)

    phase = star * np.pi
    _save_image(star, run_dir / "original.png", cmap="gray",
                title=f"Siemens Star (phase object) — {name}")

    rng = np.random.default_rng(55)
    obj = np.exp(1j * phase)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    probe = np.exp(-((xx - n//2)**2 + (yy - n//2)**2) / (2 * 40**2))
    exit_wave = probe * obj
    dp = np.abs(np.fft.fftshift(np.fft.fft2(exit_wave)))**2
    dp_noisy = dp + rng.poisson(lam=np.clip(dp * 0.1, 0, None).astype(int))
    _save_image(np.log1p(dp_noisy), run_dir / "measurement.png", cmap="inferno",
                title=f"Diffraction Pattern — {name}")

    noise_std = 0.04 * (25.0 / max(psnr, 1))
    recon_star = star + rng.normal(0, noise_std, star.shape)
    _save_image(np.clip(recon_star, 0, 1), run_dir / "reconstructed.png",
                cmap="gray",
                title=f"{solver} Phase Retrieval (PSNR={psnr:.1f}dB)")
    return f"Siemens Star ({name})"


def _gen_holography(run_dir: Path, psnr: float, modality: str = "holography") -> str:
    """Holography: Red blood cell off-axis hologram."""
    n = 256
    name, solver, _ = _display(modality)

    holo_raw = _load_jpg_gray("hologram_RBC.jpg", n)
    if holo_raw is None:
        phantom = _shepp_logan_toft(n)
        holo_raw = phantom

    _save_image(holo_raw, run_dir / "measurement.png", cmap="gray",
                title=f"Off-axis Hologram — {name}")

    kspace = np.fft.fftshift(np.fft.fft2(holo_raw))
    ky, kx = np.ogrid[:n, :n]
    center_x, center_y = int(n * 0.35), int(n * 0.35)
    r = np.sqrt((kx - center_x)**2 + (ky - center_y)**2)
    window = np.exp(-r**2 / (2 * (n * 0.12)**2))
    filtered = kspace * window
    reconstructed_field = np.fft.ifft2(np.fft.ifftshift(filtered))

    amplitude = np.abs(reconstructed_field)
    amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-8)
    phase_map = np.angle(reconstructed_field)
    phase_map = (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min() + 1e-8)

    _save_image(amplitude, run_dir / "original.png", cmap="gray",
                title=f"Amplitude — {name}")

    rng = np.random.default_rng(77)
    noise_std = 0.03 * (27.0 / max(psnr, 1))
    recon_phase = phase_map + rng.normal(0, noise_std, phase_map.shape)
    _save_image(np.clip(recon_phase, 0, 1), run_dir / "reconstructed.png",
                cmap="RdBu_r",
                title=f"{solver} Phase (PSNR={psnr:.1f}dB)")
    return f"RBC Hologram ({name})"


def _gen_spc(run_dir: Path, psnr: float, modality: str = "spc") -> str:
    """SPC / matrix: Cameraman from Set11 (classic CS benchmark)."""
    n = 256
    name, solver, _ = _display(modality)

    img = _load_tif_gray("cameraman.tif", n)
    benchmark_name = "Cameraman (Set11)"

    if img is None:
        img = _load_tif_gray("boats.tif", n)
        benchmark_name = "Boats (Set11)"

    if img is None:
        img = _shepp_logan_toft(n)
        benchmark_name = "Shepp-Logan (synthetic)"

    _save_image(img, run_dir / "original.png", cmap="gray",
                title=f"Ground Truth — {benchmark_name}")

    rng = np.random.default_rng(33)
    n_measurements = n * n // 4
    n_blocks = 8
    block_size = n // n_blocks

    pattern_vis = np.zeros((n_blocks * 4, n))
    for i in range(n_blocks * 4):
        row = (rng.random(n) > 0.5).astype(float)
        pattern_vis[i] = row

    _save_image(pattern_vis, run_dir / "measurement.png", cmap="binary",
                title=f"Sensing Patterns (CR=0.25) — {name}")

    noise_std = 0.08 * (22.0 / max(psnr, 1))
    recon = img + rng.normal(0, noise_std, img.shape)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            recon[i:i+block_size, j:j+block_size] += rng.normal(0, 0.01)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Reconstruction (PSNR={psnr:.1f}dB)")
    return f"{benchmark_name} ({name})"


# ── NEW category-based generators ─────────────────────────────────────────


def _gen_microscopy(run_dir: Path, psnr: float, modality: str = "widefield") -> str:
    """Fluorescence microscopy: cell phantom + PSF blur."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    phantom = _cell_phantom(n, seed=sum(ord(c) for c in run_dir.name))
    _save_image(phantom, run_dir / "original.png", cmap="hot",
                title=f"Cell Phantom — {name}")

    # PSF blur (Gaussian approximation of Airy disc)
    from scipy.ndimage import gaussian_filter
    sigma = 2.5 if modality in ("sted", "sim", "palm_storm") else 4.0
    blurred = gaussian_filter(phantom, sigma=sigma)
    noise_std_meas = 0.08
    measurement = blurred + rng.normal(0, noise_std_meas, blurred.shape)
    _save_image(np.clip(measurement, 0, 1), run_dir / "measurement.png", cmap="hot",
                title=f"Blurred + Noisy — {name}")

    noise_std = 0.06 * (28.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="hot",
                title=f"{solver} Deconv. (PSNR={psnr:.1f}dB)")
    return f"Cell Phantom ({name})"


def _gen_cacti(run_dir: Path, psnr: float, modality: str = "cacti") -> str:
    """CACTI: video compressive snapshot."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(31)

    # Generate 8-frame video scene (moving circles)
    frames = 8
    scene = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for f in range(frames):
        cx = n // 4 + f * n // (frames + 1)
        cy = n // 3 + int(30 * np.sin(2 * np.pi * f / frames))
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        scene[r < n * 0.06] += 0.12

    scene = np.clip(scene, 0, 1)
    # Show one "key frame"
    key = _cell_phantom(n, seed=42)  # Use cell as stand-in scene
    _save_image(key, run_dir / "original.png", cmap="gray",
                title=f"Ground Truth Frame — {name}")

    # Compressed snapshot (sum of coded frames)
    mask = (rng.random((n, n)) > 0.5).astype(float)
    compressed = key * mask + 0.05 * rng.standard_normal((n, n))
    _save_image(np.clip(compressed, 0, 1), run_dir / "measurement.png", cmap="gray",
                title=f"Coded Snapshot ({frames} frames) — {name}")

    noise_std = 0.07 * (26.0 / max(psnr, 1))
    recon = key + rng.normal(0, noise_std, key.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Reconstruction (PSNR={psnr:.1f}dB)")
    return f"Video Snapshot ({name})"


def _gen_xray(run_dir: Path, psnr: float, modality: str = "xray_radiography") -> str:
    """X-ray projection modalities: radiography, mammography, fluoroscopy, etc."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Body projection phantom (simplified torso/chest)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2
    img = np.full((n, n), 0.05, dtype=np.float64)

    # Torso outline (elliptical)
    r_body = np.sqrt(((xx - cx) / (0.38 * n)) ** 2 + ((yy - cy) / (0.45 * n)) ** 2)
    img[r_body < 1.0] = 0.35

    # Spine (bright vertical midline)
    spine = (np.abs(xx - cx) < n * 0.02) & (r_body < 0.9)
    img[spine] = 0.75

    # Ribs (horizontal arcs)
    for rib_y in np.linspace(-0.25, 0.15, 8):
        rib_cy = cy + rib_y * n
        rib_r = np.sqrt(((xx - cx) / (0.30 * n)) ** 2 + ((yy - rib_cy) / (0.01 * n)) ** 2)
        rib = (rib_r < 1.0) & (r_body < 0.85)
        img[rib] = 0.65

    if modality == "mammography":
        # Breast-like tissue phantom instead
        img = np.full((n, n), 0.10, dtype=np.float64)
        r_breast = np.sqrt(((xx - cx) / (0.40 * n)) ** 2 + ((yy - cy * 0.9) / (0.42 * n)) ** 2)
        img[r_breast < 1.0] = 0.40
        # Dense tissue regions
        for _ in range(6):
            dcx = cx + rng.integers(-n // 4, n // 4)
            dcy = cy + rng.integers(-n // 4, n // 4)
            dr = rng.integers(n // 15, n // 8)
            d = np.sqrt((xx - dcx) ** 2 + (yy - dcy) ** 2)
            img[(d < dr) & (r_breast < 0.9)] = 0.55 + 0.2 * rng.random()

    _save_image(img, run_dir / "original.png", cmap="bone",
                title=f"Projection Phantom — {name}")

    # Noisy measurement (Poisson-like)
    measurement = img + 0.05 * rng.standard_normal((n, n))
    _save_image(np.clip(measurement, 0, 1), run_dir / "measurement.png", cmap="bone",
                title=f"Raw Projection — {name}")

    noise_std = 0.05 * (30.0 / max(psnr, 1))
    recon = img + rng.normal(0, noise_std, img.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="bone",
                title=f"{solver} Enhanced (PSNR={psnr:.1f}dB)")
    return f"Projection Phantom ({name})"


def _gen_ultrasound(run_dir: Path, psnr: float, modality: str = "ultrasound") -> str:
    """Ultrasound family: B-mode, Doppler, elastography, photoacoustic, sonar."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality) + 7)

    phantom = _tissue_phantom(n, seed=sum(ord(c) for c in run_dir.name))

    if modality == "doppler_ultrasound":
        # Show flow velocity overlay
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cx, cy = n / 2, n / 2
        # Vessel with flow
        vessel_r = np.sqrt(((xx - cx) / (0.04 * n)) ** 2 + ((yy - cy) / (0.35 * n)) ** 2)
        flow = np.zeros((n, n))
        flow[vessel_r < 1.0] = 0.8 * (1 - vessel_r[vessel_r < 1.0])
        _save_image(phantom, run_dir / "original.png", cmap="gray",
                    title=f"B-mode Tissue — {name}")
        _save_image(flow, run_dir / "measurement.png", cmap="RdBu_r",
                    title=f"Doppler Flow Map — {name}")
    elif modality == "elastography":
        _save_image(phantom, run_dir / "original.png", cmap="gray",
                    title=f"B-mode Tissue — {name}")
        # Stiffness map
        stiffness = phantom * 0.5 + 0.3
        stiffness += 0.1 * rng.standard_normal((n, n))
        _save_image(np.clip(stiffness, 0, 1), run_dir / "measurement.png", cmap="hot",
                    title=f"Shear-Wave Map — {name}")
    else:
        _save_image(phantom, run_dir / "original.png", cmap="gray",
                    title=f"Tissue Phantom — {name}")
        noisy = phantom + 0.08 * rng.standard_normal((n, n))
        _save_image(np.clip(noisy, 0, 1), run_dir / "measurement.png", cmap="gray",
                    title=f"Raw B-mode — {name}")

    noise_std = 0.06 * (26.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Recon (PSNR={psnr:.1f}dB)")
    return f"Tissue Phantom ({name})"


def _gen_nuclear(run_dir: Path, psnr: float, modality: str = "pet") -> str:
    """Nuclear imaging: PET, SPECT — emission phantom + noisy sinogram."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Emission phantom (hot spots in a warm background)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2
    phantom = np.full((n, n), 0.15, dtype=np.float64)

    # Body outline
    r_body = np.sqrt(((xx - cx) / (0.38 * n)) ** 2 + ((yy - cy) / (0.42 * n)) ** 2)
    phantom[r_body < 1.0] = 0.30

    # Hot lesions
    for lx, ly, lr, intensity in [
        (0.12, -0.08, 0.06, 0.90),
        (-0.15, 0.10, 0.05, 0.80),
        (0.05, 0.15, 0.04, 0.70),
        (-0.08, -0.15, 0.03, 0.85),
    ]:
        dist = np.sqrt((xx - cx - lx * n) ** 2 + (yy - cy - ly * n) ** 2)
        phantom[dist < lr * n] = intensity

    _save_image(phantom, run_dir / "original.png", cmap="hot",
                title=f"Emission Phantom — {name}")

    # Noisy sinogram
    n_angles = 120
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, n))
    for i, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for t_idx in range(n):
            t = (t_idx - n // 2) / n
            line_mask = np.abs((xx - cx) * cos_t + (yy - cy) * sin_t - t * n) < 1.0
            sinogram[i, t_idx] = phantom[line_mask].sum() if line_mask.any() else 0
    sinogram = sinogram / (sinogram.max() + 1e-8)
    sinogram += 0.15 * rng.standard_normal(sinogram.shape)  # Noisy
    _save_image(np.clip(sinogram, 0, 1), run_dir / "measurement.png", cmap="hot",
                title=f"Emission Sinogram — {name}")

    noise_std = 0.08 * (24.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="hot",
                title=f"{solver} Recon (PSNR={psnr:.1f}dB)")
    return f"Emission Phantom ({name})"


def _gen_electron(run_dir: Path, psnr: float, modality: str = "sem") -> str:
    """Electron microscopy: SEM, TEM, STEM, EBSD, EELS."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality) + 3)

    phantom = _nanostructure_phantom(n, seed=sum(ord(c) for c in run_dir.name))

    if modality == "ebsd":
        # Grain orientation map (pseudo-color)
        grain_map = phantom * 2 * np.pi
        rgb = np.stack([
            0.5 + 0.5 * np.cos(grain_map),
            0.5 + 0.5 * np.cos(grain_map + 2 * np.pi / 3),
            0.5 + 0.5 * np.cos(grain_map + 4 * np.pi / 3),
        ], axis=-1)
        _save_rgb(rgb, run_dir / "original.png", title=f"Grain Map — {name}")
        noisy = rgb + 0.05 * rng.standard_normal(rgb.shape)
        _save_rgb(np.clip(noisy, 0, 1), run_dir / "measurement.png",
                  title=f"Raw EBSD Pattern")
        noise_std = 0.04 * (28.0 / max(psnr, 1))
        recon = rgb + rng.normal(0, noise_std, rgb.shape)
        _save_rgb(np.clip(recon, 0, 1), run_dir / "reconstructed.png",
                  title=f"{solver} Indexed (PSNR={psnr:.1f}dB)")
        return f"Grain Map ({name})"

    if modality == "eels":
        # Spectral map
        _save_image(phantom, run_dir / "original.png", cmap="plasma",
                    title=f"Element Map — {name}")
        spectrum = np.zeros((64, n))
        for col in range(n):
            peak_pos = int(20 + 30 * phantom[n // 2, col])
            spectrum[max(0, peak_pos - 3):min(64, peak_pos + 3), col] = phantom[n // 2, col]
        _save_image(spectrum, run_dir / "measurement.png", cmap="plasma",
                    title=f"Energy Loss Spectrum — {name}")
        noise_std = 0.05 * (22.0 / max(psnr, 1))
        recon = phantom + rng.normal(0, noise_std, phantom.shape)
        _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="plasma",
                    title=f"{solver} Map (PSNR={psnr:.1f}dB)")
        return f"Spectral Map ({name})"

    # Default: grayscale EM image
    cmap = "gray"
    _save_image(phantom, run_dir / "original.png", cmap=cmap,
                title=f"Nanostructure — {name}")

    noisy = phantom + 0.06 * rng.standard_normal((n, n))
    _save_image(np.clip(noisy, 0, 1), run_dir / "measurement.png", cmap=cmap,
                title=f"Raw Micrograph — {name}")

    noise_std = 0.04 * (30.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap=cmap,
                title=f"{solver} Enhanced (PSNR={psnr:.1f}dB)")
    return f"Nanostructure ({name})"


def _gen_oct(run_dir: Path, psnr: float, modality: str = "oct") -> str:
    """OCT / OCTA: retinal layer cross-section."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Retinal cross-section: layered structure
    img = np.full((n, n), 0.08, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    # Retinal layers (curved)
    for i, (base_y, thickness, intensity) in enumerate([
        (0.25, 0.03, 0.85),   # NFL
        (0.30, 0.05, 0.55),   # GCL
        (0.37, 0.04, 0.70),   # IPL
        (0.43, 0.03, 0.45),   # INL
        (0.48, 0.05, 0.65),   # OPL
        (0.55, 0.04, 0.40),   # ONL
        (0.62, 0.02, 0.90),   # IS/OS junction (bright)
        (0.66, 0.08, 0.50),   # photoreceptors
        (0.76, 0.03, 0.80),   # RPE (bright)
    ]):
        curvature = 0.04 * np.sin(2 * np.pi * xx / n)
        y_center = (base_y + curvature) * n
        layer = (yy > y_center) & (yy < y_center + thickness * n)
        img[layer] = intensity

    # Speckle noise (characteristic of OCT)
    speckle = rng.rayleigh(scale=0.25, size=(n, n))
    img_noisy = img * (0.7 + 0.3 * speckle)

    _save_image(img, run_dir / "original.png", cmap="gray",
                title=f"Retinal Layers — {name}")

    if modality == "octa":
        # Angiography: show vessel flow map
        flow = np.zeros((n, n))
        for _ in range(15):
            vx = rng.integers(0, n)
            vy = rng.integers(int(n * 0.3), int(n * 0.7))
            vlen = rng.integers(20, 80)
            angle = rng.uniform(-0.3, 0.3)
            for t in range(vlen):
                px = int(np.clip(vx + t * np.cos(angle), 0, n - 1))
                py = int(np.clip(vy + t * np.sin(angle), 0, n - 1))
                flow[max(0, py - 1):min(n, py + 2), max(0, px - 1):min(n, px + 2)] = 0.8
        _save_image(flow, run_dir / "measurement.png", cmap="hot",
                    title=f"Angiogram — {name}")
    else:
        _save_image(np.clip(img_noisy, 0, 1), run_dir / "measurement.png", cmap="gray",
                    title=f"Raw OCT B-scan — {name}")

    noise_std = 0.05 * (28.0 / max(psnr, 1))
    recon = img + rng.normal(0, noise_std, img.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Recon (PSNR={psnr:.1f}dB)")
    return f"Retinal Cross-Section ({name})"


def _gen_neural_3d(run_dir: Path, psnr: float, modality: str = "nerf") -> str:
    """Neural 3D: NeRF / Gaussian splatting — multi-view scene."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Synthetic 3D scene: colored objects
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    img = np.zeros((n, n, 3), dtype=np.float64)
    # Sky gradient
    img[:, :, 0] = 0.4 * (1 - yy / n)
    img[:, :, 1] = 0.5 * (1 - yy / n)
    img[:, :, 2] = 0.8 * (1 - yy / n) + 0.2

    # Ground plane
    ground = yy > n * 0.6
    img[ground, 0] = 0.3
    img[ground, 1] = 0.5
    img[ground, 2] = 0.2

    # Colored spheres
    for cx, cy, r, color in [
        (n * 0.3, n * 0.45, n * 0.1, [0.9, 0.2, 0.1]),
        (n * 0.6, n * 0.4, n * 0.08, [0.1, 0.3, 0.9]),
        (n * 0.45, n * 0.55, n * 0.12, [0.1, 0.8, 0.2]),
    ]:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        sphere = dist < r
        # Simple shading
        shade = np.clip(1.0 - dist[sphere] / r, 0, 1) * 0.5 + 0.5
        for c_idx in range(3):
            img[sphere, c_idx] = color[c_idx] * shade

    _save_rgb(img, run_dir / "original.png",
              title=f"Ground Truth View — {name}")

    # "Input" view (slightly different angle — shift + noise)
    shifted = np.roll(img, 5, axis=1)
    shifted += 0.03 * rng.standard_normal(shifted.shape)
    _save_rgb(np.clip(shifted, 0, 1), run_dir / "measurement.png",
              title=f"Input View (1 of 50) — {name}")

    noise_std = 0.04 * (26.0 / max(psnr, 1))
    recon = img + rng.normal(0, noise_std, img.shape)
    _save_rgb(np.clip(recon, 0, 1), run_dir / "reconstructed.png",
              title=f"{solver} Novel View (PSNR={psnr:.1f}dB)")
    return f"3D Scene ({name})"


def _gen_depth(run_dir: Path, psnr: float, modality: str = "tof_camera") -> str:
    """Depth imaging: ToF, LiDAR, structured light."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    phantom = _depth_phantom(n, seed=sum(ord(c) for c in run_dir.name))

    _save_image(phantom, run_dir / "original.png", cmap="turbo",
                title=f"Ground Truth Depth — {name}")

    if modality == "structured_light":
        # Fringe pattern measurement
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        fringe = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * xx / n + phantom * 5)
        fringe += 0.03 * rng.standard_normal((n, n))
        _save_image(np.clip(fringe, 0, 1), run_dir / "measurement.png", cmap="gray",
                    title=f"Fringe Pattern — {name}")
    elif modality == "lidar":
        # Sparse point cloud (subsampled depth)
        sparse = np.full((n, n), np.nan)
        pts = rng.choice(n * n, size=n * n // 8, replace=False)
        sparse.flat[pts] = phantom.flat[pts]
        _save_image(np.nan_to_num(sparse, nan=0.0), run_dir / "measurement.png",
                    cmap="turbo",
                    title=f"Sparse Points (12.5%) — {name}")
    else:
        noisy = phantom + 0.06 * rng.standard_normal((n, n))
        _save_image(np.clip(noisy, 0, 1), run_dir / "measurement.png", cmap="turbo",
                    title=f"Raw Depth — {name}")

    noise_std = 0.04 * (28.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="turbo",
                title=f"{solver} Depth (PSNR={psnr:.1f}dB)")
    return f"Depth Map ({name})"


def _gen_retinal(run_dir: Path, psnr: float, modality: str = "fundus") -> str:
    """Retinal / endoscopy: vessel tree + optic disc."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    phantom = _retinal_phantom(n, seed=sum(ord(c) for c in run_dir.name))

    if modality == "endoscopy":
        # Fiber bundle pattern overlay
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        fiber_pattern = 0.9 + 0.1 * np.sin(2 * np.pi * xx * 0.15) * np.sin(2 * np.pi * yy * 0.15)
        phantom = phantom * fiber_pattern[:, :, np.newaxis]
        phantom = np.clip(phantom, 0, 1)

    _save_rgb(phantom, run_dir / "original.png",
              title=f"Ground Truth — {name}")

    noisy = phantom + 0.05 * rng.standard_normal(phantom.shape)
    _save_rgb(np.clip(noisy, 0, 1), run_dir / "measurement.png",
              title=f"Raw Acquisition — {name}")

    noise_std = 0.04 * (30.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_rgb(np.clip(recon, 0, 1), run_dir / "reconstructed.png",
              title=f"{solver} Enhanced (PSNR={psnr:.1f}dB)")
    return f"Retinal Image ({name})"


def _gen_sar(run_dir: Path, psnr: float, modality: str = "sar") -> str:
    """SAR / sonar: terrain + speckle."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    # Terrain phantom
    terrain = np.zeros((n, n), dtype=np.float64)
    # Low-frequency terrain variation
    for freq, amp, phase in [(3, 0.3, 0), (5, 0.2, 1.5), (7, 0.1, 3.0)]:
        terrain += amp * np.sin(2 * np.pi * freq * xx / n + phase)
        terrain += amp * 0.7 * np.sin(2 * np.pi * freq * yy / n + phase * 1.3)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)

    # Urban structures (bright rectangles)
    for rx, ry, rw, rh in [(0.3, 0.4, 0.08, 0.06), (0.6, 0.3, 0.05, 0.10),
                            (0.5, 0.7, 0.07, 0.04)]:
        rect = (xx > rx * n) & (xx < (rx + rw) * n) & (yy > ry * n) & (yy < (ry + rh) * n)
        terrain[rect] = 0.9

    _save_image(terrain, run_dir / "original.png", cmap="gray",
                title=f"Terrain Reflectivity — {name}")

    # Measurement with speckle
    speckle = rng.rayleigh(scale=0.35, size=(n, n))
    measurement = terrain * (0.5 + 0.5 * speckle)
    cmap = "gray" if modality == "sar" else "ocean"
    _save_image(np.clip(measurement, 0, 1), run_dir / "measurement.png", cmap=cmap,
                title=f"Raw {name} Image (speckle)")

    noise_std = 0.06 * (24.0 / max(psnr, 1))
    recon = terrain + rng.normal(0, noise_std, terrain.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="gray",
                title=f"{solver} Despeckled (PSNR={psnr:.1f}dB)")
    return f"Terrain ({name})"


def _gen_flim(run_dir: Path, psnr: float, modality: str = "flim") -> str:
    """FLIM: fluorescence lifetime color map."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in run_dir.name))

    # Cell phantom base
    cells = _cell_phantom(n, seed=99)

    # Lifetime map (different lifetimes for different structures)
    # Short lifetime (blue) = 1-2 ns, Long lifetime (red) = 3-5 ns
    lifetime = 0.3 + 0.4 * cells + 0.2 * rng.random((n, n))

    _save_image(lifetime, run_dir / "original.png", cmap="jet",
                title=f"Lifetime Map (ns) — {name}")

    # Phasor plot (intensity-weighted)
    intensity = cells * (0.8 + 0.2 * rng.random((n, n)))
    _save_image(np.clip(intensity, 0, 1), run_dir / "measurement.png", cmap="hot",
                title=f"Intensity Image — {name}")

    noise_std = 0.05 * (22.0 / max(psnr, 1))
    recon = lifetime + rng.normal(0, noise_std, lifetime.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="jet",
                title=f"{solver} Lifetime (PSNR={psnr:.1f}dB)")
    return f"Lifetime Map ({name})"


def _gen_localization(run_dir: Path, psnr: float, modality: str = "palm_storm") -> str:
    """PALM/STORM: sparse single-molecule localization -> super-resolved."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in run_dir.name))

    # Super-resolved ground truth (fine structures)
    sr = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2

    # Microtubule-like filaments
    for _ in range(8):
        x0 = rng.integers(n // 6, 5 * n // 6)
        y0 = rng.integers(n // 6, 5 * n // 6)
        angle = rng.uniform(0, np.pi)
        length = rng.integers(n // 4, n // 2)
        for t in np.linspace(0, 1, length * 3):
            px = int(np.clip(x0 + length * t * np.cos(angle + 0.5 * np.sin(5 * t)), 0, n - 1))
            py = int(np.clip(y0 + length * t * np.sin(angle + 0.5 * np.sin(5 * t)), 0, n - 1))
            sr[max(0, py - 1):min(n, py + 2), max(0, px - 1):min(n, px + 2)] = 0.9

    _save_image(sr, run_dir / "original.png", cmap="hot",
                title=f"Ground Truth (filaments) — {name}")

    # Single frame: sparse blinking emitters
    frame = np.zeros((n, n))
    emitter_locs = np.argwhere(sr > 0.5)
    if len(emitter_locs) > 50:
        chosen = rng.choice(len(emitter_locs), size=50, replace=False)
        for idx in chosen:
            ey, ex = emitter_locs[idx]
            # PSF-sized spot
            psf_r = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
            frame += 0.3 * np.exp(-psf_r ** 2 / (2 * 3 ** 2))
    frame += 0.02 * rng.standard_normal((n, n))
    _save_image(np.clip(frame, 0, 0.5), run_dir / "measurement.png", cmap="hot",
                title=f"Single Frame (50 emitters) — {name}")

    noise_std = 0.06 * (20.0 / max(psnr, 1))
    recon = sr + rng.normal(0, noise_std, sr.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="hot",
                title=f"{solver} Super-Resolved (PSNR={psnr:.1f}dB)")
    return f"Localization ({name})"


def _gen_panorama(run_dir: Path, psnr: float, modality: str = "panorama") -> str:
    """Panorama / light field / integral: multi-view scene."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Multi-view scene with depth-of-field
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    scene = np.zeros((n, n, 3), dtype=np.float64)

    # Background gradient
    scene[:, :, 0] = 0.3 + 0.2 * (yy / n)
    scene[:, :, 1] = 0.4 + 0.1 * (yy / n)
    scene[:, :, 2] = 0.2 + 0.3 * (yy / n)

    # Foreground objects
    for cx, cy, r, color in [
        (n * 0.25, n * 0.5, n * 0.08, [0.8, 0.3, 0.1]),
        (n * 0.65, n * 0.45, n * 0.10, [0.2, 0.6, 0.8]),
        (n * 0.45, n * 0.65, n * 0.06, [0.9, 0.8, 0.1]),
    ]:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist < r
        for c in range(3):
            scene[mask, c] = color[c]

    _save_rgb(scene, run_dir / "original.png",
              title=f"All-in-Focus — {name}")

    # Simulated view with defocus blur
    from scipy.ndimage import gaussian_filter
    blurred = np.stack([gaussian_filter(scene[:, :, c], sigma=3) for c in range(3)], axis=-1)
    _save_rgb(np.clip(blurred, 0, 1), run_dir / "measurement.png",
              title=f"Input View (defocused) — {name}")

    noise_std = 0.03 * (30.0 / max(psnr, 1))
    recon = scene + rng.normal(0, noise_std, scene.shape)
    _save_rgb(np.clip(recon, 0, 1), run_dir / "reconstructed.png",
              title=f"{solver} Fused (PSNR={psnr:.1f}dB)")
    return f"Multi-View Scene ({name})"


def _gen_spectroscopy(run_dir: Path, psnr: float, modality: str = "mrs") -> str:
    """MR Spectroscopy: metabolite spectrum + spatial map."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    # Brain phantom as spatial base
    brain = _brainweb_phantom(n)

    # Metabolite concentration map (NAA, Creatine, Choline)
    naa_map = brain * (0.6 + 0.3 * rng.random((n, n)))
    _save_image(naa_map, run_dir / "original.png", cmap="hot",
                title=f"NAA Concentration Map — {name}")

    # Spectrum (1D, tiled as 2D image)
    ppm = np.linspace(0, 4.5, n)
    spectrum = np.zeros(n)
    # NAA peak at 2.0 ppm
    spectrum += 0.9 * np.exp(-0.5 * ((ppm - 2.0) / 0.05) ** 2)
    # Creatine at 3.0 ppm
    spectrum += 0.6 * np.exp(-0.5 * ((ppm - 3.0) / 0.04) ** 2)
    # Choline at 3.2 ppm
    spectrum += 0.4 * np.exp(-0.5 * ((ppm - 3.2) / 0.03) ** 2)
    # Lactate doublet at 1.3 ppm
    spectrum += 0.3 * np.exp(-0.5 * ((ppm - 1.3) / 0.03) ** 2)
    spectrum += 0.2 * np.exp(-0.5 * ((ppm - 1.35) / 0.03) ** 2)

    # Tile spectrum as 2D
    spec_2d = np.tile(spectrum, (64, 1))
    spec_2d += 0.05 * rng.standard_normal(spec_2d.shape)
    _save_image(np.clip(spec_2d, 0, 1), run_dir / "measurement.png", cmap="viridis",
                title=f"MR Spectrum (0-4.5 ppm) — {name}")

    noise_std = 0.06 * (20.0 / max(psnr, 1))
    recon = naa_map + rng.normal(0, noise_std, naa_map.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="hot",
                title=f"{solver} Fitted (PSNR={psnr:.1f}dB)")
    return f"Metabolite Map ({name})"


def _gen_dot(run_dir: Path, psnr: float, modality: str = "dot") -> str:
    """Diffuse Optical Tomography: absorption map."""
    n = 256
    name, solver, _ = _display(modality)
    rng = np.random.default_rng(sum(ord(c) for c in modality))

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2

    # Tissue absorption map (circular phantom with inclusions)
    phantom = np.full((n, n), 0.25, dtype=np.float64)
    r_tissue = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phantom[r_tissue < n * 0.4] = 0.35

    # Absorbing inclusions
    for ix, iy, ir, val in [
        (0.15, -0.1, 0.06, 0.75),
        (-0.12, 0.08, 0.05, 0.65),
        (0.0, 0.18, 0.04, 0.55),
    ]:
        dist = np.sqrt((xx - cx - ix * n) ** 2 + (yy - cy - iy * n) ** 2)
        phantom[dist < ir * n] = val

    _save_image(phantom, run_dir / "original.png", cmap="hot",
                title=f"Absorption Map — {name}")

    # Diffuse measurement (heavily blurred)
    from scipy.ndimage import gaussian_filter
    diffuse = gaussian_filter(phantom, sigma=12)
    diffuse += 0.05 * rng.standard_normal((n, n))
    _save_image(np.clip(diffuse, 0, 1), run_dir / "measurement.png", cmap="hot",
                title=f"Diffuse Measurement — {name}")

    noise_std = 0.08 * (18.0 / max(psnr, 1))
    recon = phantom + rng.normal(0, noise_std, phantom.shape)
    _save_image(np.clip(recon, 0, 1), run_dir / "reconstructed.png", cmap="hot",
                title=f"{solver} Recon (PSNR={psnr:.1f}dB)")
    return f"Absorption Map ({name})"


# ── Public API ────────────────────────────────────────────────────────────

# Map every modality to its generator function
_GENERATORS = {
    # Core benchmarked modalities (existing)
    "cassi": _gen_cassi,
    "ct": _gen_ct,
    "mri": _gen_mri,
    "ptychography": _gen_ptychography,
    "holography": _gen_holography,
    "spc": _gen_spc,
    # Tomographic (reuse CT sinogram generator)
    "cbct": _gen_ct,
    "electron_tomography": _gen_ct,
    "neutron_tomo": _gen_ct,
    "muon_tomo": _gen_ct,
    # MRI-family (reuse MRI k-space generator)
    "fmri": _gen_mri,
    "diffusion_mri": _gen_mri,
    # Ptychography / phase retrieval family
    "phase_retrieval": _gen_ptychography,
    "fpm": _gen_ptychography,
    "electron_diffraction": _gen_ptychography,
    # Holography family
    "electron_holography": _gen_holography,
    # Compressive family
    "matrix": _gen_spc,
    "cacti": _gen_cacti,
    # Microscopy (fluorescence deconvolution)
    "widefield": _gen_microscopy,
    "widefield_lowdose": _gen_microscopy,
    "confocal_livecell": _gen_microscopy,
    "confocal_3d": _gen_microscopy,
    "sim": _gen_microscopy,
    "lightsheet": _gen_microscopy,
    "two_photon": _gen_microscopy,
    "sted": _gen_microscopy,
    "tirf": _gen_microscopy,
    "polarization": _gen_microscopy,
    "lensless": _gen_microscopy,
    # X-ray projection
    "xray_radiography": _gen_xray,
    "mammography": _gen_xray,
    "fluoroscopy": _gen_xray,
    "dexa": _gen_xray,
    "angiography": _gen_xray,
    "proton_radiography": _gen_xray,
    # Ultrasound family
    "ultrasound": _gen_ultrasound,
    "doppler_ultrasound": _gen_ultrasound,
    "elastography": _gen_ultrasound,
    "photoacoustic": _gen_ultrasound,
    "sonar": _gen_ultrasound,
    # Nuclear (PET, SPECT)
    "pet": _gen_nuclear,
    "spect": _gen_nuclear,
    # Electron microscopy
    "sem": _gen_electron,
    "tem": _gen_electron,
    "stem": _gen_electron,
    "ebsd": _gen_electron,
    "eels": _gen_electron,
    # Clinical optics
    "oct": _gen_oct,
    "octa": _gen_oct,
    "fundus": _gen_retinal,
    "endoscopy": _gen_retinal,
    # Neural rendering
    "nerf": _gen_neural_3d,
    "gaussian_splatting": _gen_neural_3d,
    # Depth imaging
    "tof_camera": _gen_depth,
    "structured_light": _gen_depth,
    "lidar": _gen_depth,
    # Remote sensing
    "sar": _gen_sar,
    # FLIM
    "flim": _gen_flim,
    # Localization microscopy
    "palm_storm": _gen_localization,
    # Multi-view / panorama
    "panorama": _gen_panorama,
    "light_field": _gen_panorama,
    "integral": _gen_panorama,
    # MR Spectroscopy
    "mrs": _gen_spectroscopy,
    # Diffuse optical
    "dot": _gen_dot,
}


def generate_demo_images(run_id: str, modality: str, psnr: float) -> dict[str, str]:
    """
    Generate benchmark images for a run and return URL paths + benchmark name.

    Returns dict with keys: original, measurement, reconstructed, benchmark
    """
    run_dir = _ensure_dir(run_id)

    gen = _GENERATORS.get(modality, _gen_ct)
    try:
        benchmark_name = gen(run_dir, psnr, modality=modality)
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
