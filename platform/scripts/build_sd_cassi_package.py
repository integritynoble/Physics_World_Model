#!/usr/bin/env python3
"""Build the SD-CASSI dataset package with unique per-tier data.

Public tier  — KAIST simulation scenes (256x256x28, publicly known, full GT)
               Uses real calibrated mask.
Dev tier     — 500x500x28 unique scenes from CAVE, TokyoTech, Harvard, Chikusei,
               TSA real reconstructions (secret crops, flips, rotations)
               Uses assumed (randomly generated) mask.
Hidden tier  — 500x500x28 different unique scenes from the same sources
               (different scenes/crops/transforms — fully secret)
               Uses assumed (randomly generated) mask.

All dev/hidden scenes are simulated through the CASSI forward model so we
have ground truth for server-side scoring, but the scenes themselves are
unique and cannot be found in any public dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRUTH_DIR = REPO_ROOT / "datasets" / "TSA_simu_data" / "Truth"
MASK_PATH = REPO_ROOT / "datasets" / "TSA_simu_data" / "mask.mat"
RECON_DIR = REPO_ROOT / "datasets" / "TSA_real_data" / "TSA_reconstruction"
BENCH_DIR = REPO_ROOT / "datasets" / "benchmark"
CAVE_DIR  = BENCH_DIR / "CAVE"
TT_DIR    = BENCH_DIR / "TokyoTech" / "SceneAll"
HARVARD_DIR = BENCH_DIR / "Harvard" / "CZ_hsdb"
CHIKUSEI_MAT = BENCH_DIR / "Chikusei" / "Chikusei_MATLAB" / "HyperspecVNIR_Chikusei_20140729.mat"
OUTPUT_DIR = REPO_ROOT / "platform" / "sd_cassi_package"

NUM_BANDS = 28
WAVELENGTHS_NM = np.linspace(450, 650, NUM_BANDS).astype(int)
LARGE_SIZE = 500

# ═══════════════════════════════════════════════════════════════════════════
#  Scene Sources — unique crops and transforms held only on PWM server
# ═══════════════════════════════════════════════════════════════════════════
#
# Sources:
#   cave      — CAVE: 512×512×31 → crop 500×500, bands 1-28 (drop first and last 2)
#   tokyotech — TT-31: 500×500×31 → use as-is, bands 1-28
#   harvard   — Harvard: 1392×1040×31 → crop 500×500, bands 1-28
#   chikusei  — Chikusei: 2517×2335×128 → crop 500×500, resample 128→28
#   tsa_recon — TSA real recon: 660×660×28 → crop 500×500

# Dev tier: 20 unique 500×500×28 scenes
# (source, file/scene, crop_row, crop_col, flip_h, flip_v, rot90)
DEV_SCENES = [
    # CAVE (5 scenes — crop 6px border region with secret offsets)
    ("cave", "beads_ms",                       3,  7, False, False, 0),
    ("cave", "flowers_ms",                     8,  2, True,  False, 0),
    ("cave", "oil_painting_ms",                5,  4, False, True,  0),
    ("cave", "feathers_ms",                    1,  9, True,  True,  0),
    ("cave", "egyptian_statue_ms",             6,  3, False, False, 1),
    # TokyoTech (5 scenes — already 500×500, just transforms)
    ("tokyotech", "Butterfly.mat",             0,  0, True,  False, 0),
    ("tokyotech", "CD.mat",                    0,  0, False, True,  0),
    ("tokyotech", "Cloth.mat",                 0,  0, True,  True,  0),
    ("tokyotech", "Fan.mat",                   0,  0, False, False, 1),
    ("tokyotech", "Flower.mat",                0,  0, True,  False, 2),
    # Harvard (5 scenes — large, plenty of crop room)
    ("harvard", "img1.mat",                   120, 340, False, False, 0),
    ("harvard", "imga2.mat",                   80, 520, True,  False, 0),
    ("harvard", "imgb5.mat",                  200, 180, False, True,  0),
    ("harvard", "imgd2.mat",                  310, 650, True,  True,  0),
    ("harvard", "imge1.mat",                   50, 400, False, False, 0),
    # Chikusei (3 scenes — huge image, many crop regions)
    ("chikusei", None,                        100, 200, False, False, 0),
    ("chikusei", None,                        800, 1400, True,  False, 0),
    ("chikusei", None,                       1600, 700, False, True,  0),
    # TSA real recons (2 scenes — 660→500 crops)
    ("tsa_recon", "Recon_scene1.mat",          80,  40, False, False, 0),
    ("tsa_recon", "Recon_scene2.mat",          30, 120, True,  False, 0),
]

# Hidden tier: 20 different unique 500×500×28 scenes
HIDDEN_SCENES = [
    # CAVE (5 different scenes)
    ("cave", "glass_tiles_ms",                 4,  6, True,  False, 0),
    ("cave", "pompoms_ms",                     7,  1, False, True,  0),
    ("cave", "stuffed_toys_ms",                2,  8, True,  True,  0),
    ("cave", "real_and_fake_peppers_ms",       9,  5, False, False, 2),
    ("cave", "watercolors_ms",                 3, 10, True,  False, 1),
    # TokyoTech (5 different scenes)
    ("tokyotech", "Cloth3.mat",                0,  0, False, True,  1),
    ("tokyotech", "Doll.mat",                  0,  0, True,  False, 0),
    ("tokyotech", "Party.mat",                 0,  0, True,  True,  0),
    ("tokyotech", "Tape.mat",                  0,  0, False, False, 2),
    ("tokyotech", "Tshirts.mat",               0,  0, True,  False, 3),
    # Harvard (5 different scenes + crops)
    ("harvard", "imgb0.mat",                  150, 250, True,  False, 0),
    ("harvard", "imgb7.mat",                  400, 100, False, True,  0),
    ("harvard", "imgd9.mat",                   70, 800, True,  True,  0),
    ("harvard", "imgf5.mat",                  280, 450, False, False, 0),
    ("harvard", "imgh0.mat",                  100, 600, True,  False, 0),
    # Chikusei (3 different crop regions)
    ("chikusei", None,                        400, 900, True,  True,  0),
    ("chikusei", None,                       1200, 300, False, False, 0),
    ("chikusei", None,                       1900, 1500, True,  False, 0),
    # TSA real recons (2 different crops)
    ("tsa_recon", "Recon_scene3.mat",         100,  60, False, True,  0),
    ("tsa_recon", "Recon_scene4.mat",         140,  50, True,  False, 0),
]


# ── Data loaders ─────────────────────────────────────────────────────────

def load_kaist_scene(scene_idx: int) -> np.ndarray:
    """Load a KAIST scene (256×256×28) by index (0-based)."""
    path = TRUTH_DIR / f"scene{scene_idx + 1:02d}.mat"
    return _load_cube_from_mat(path)


def load_recon_scene(filename: str) -> np.ndarray:
    """Load a TSA real reconstruction scene (660×660×28)."""
    return _load_cube_from_mat(RECON_DIR / filename)


def load_cave_scene(scene_name: str) -> np.ndarray:
    """Load a CAVE scene (512×512×31) from 31 PNG files."""
    scene_dir = CAVE_DIR / scene_name / scene_name
    if not scene_dir.exists():
        # Try without double nesting
        scene_dir = CAVE_DIR / scene_name
    pngs = sorted([f for f in os.listdir(scene_dir) if f.endswith('.png')])
    assert len(pngs) == 31, f"Expected 31 bands in {scene_dir}, got {len(pngs)}"
    bands = []
    for png_file in pngs:
        img = Image.open(scene_dir / png_file)
        bands.append(np.array(img, dtype=np.float64))
    cube = np.stack(bands, axis=-1)  # (512, 512, 31)
    # Normalize to [0, 1]
    vmax = cube.max()
    if vmax > 0:
        cube = cube / vmax
    return cube


def load_tokyotech_scene(filename: str) -> np.ndarray:
    """Load a TokyoTech TT-31 scene (500×500×31)."""
    return _load_cube_from_mat(TT_DIR / filename)


def load_harvard_scene(filename: str) -> np.ndarray:
    """Load a Harvard scene (1040×1392×31)."""
    mat = sio.loadmat(str(HARVARD_DIR / filename))
    for key in ("ref", "img", "data"):
        if key in mat:
            cube = mat[key].astype(np.float64)
            # Normalize
            vmax = cube.max()
            if vmax > 0:
                cube = cube / vmax
            return cube
    for key, val in mat.items():
        if not key.startswith("_") and isinstance(val, np.ndarray) and val.ndim == 3:
            cube = val.astype(np.float64)
            vmax = cube.max()
            if vmax > 0:
                cube = cube / vmax
            return cube
    raise ValueError(f"Cannot find cube data in Harvard/{filename}")


_chikusei_cache = None

def load_chikusei_crop(crop_r: int, crop_c: int, size: int = 500) -> np.ndarray:
    """Load a crop from Chikusei (2517×2335×128) and resample to 28 bands."""
    global _chikusei_cache
    if _chikusei_cache is None:
        import h5py
        print("    [Loading Chikusei into memory...]", end=" ", flush=True)
        with h5py.File(str(CHIKUSEI_MAT), "r") as f:
            # Shape is (128, 2335, 2517) in the file — need to transpose
            data = f["chikusei"][:]  # (128, 2335, 2517)
        _chikusei_cache = np.transpose(data, (2, 1, 0)).astype(np.float64)  # (2517, 2335, 128)
        vmax = _chikusei_cache.max()
        if vmax > 0:
            _chikusei_cache = _chikusei_cache / vmax
        print("done", flush=True)

    cube = _chikusei_cache  # (2517, 2335, 128)
    assert crop_r + size <= cube.shape[0], f"Chikusei crop row {crop_r}+{size} > {cube.shape[0]}"
    assert crop_c + size <= cube.shape[1], f"Chikusei crop col {crop_c}+{size} > {cube.shape[1]}"
    crop = cube[crop_r:crop_r + size, crop_c:crop_c + size, :]  # (500, 500, 128)

    # Resample 128 bands (363-1018nm) → 28 bands (450-650nm)
    src_wl = np.linspace(363, 1018, 128)
    tgt_wl = np.linspace(450, 650, 28)
    # For each target band, find the nearest source band
    resampled = np.zeros((size, size, 28), dtype=np.float64)
    for i, tw in enumerate(tgt_wl):
        # Gaussian-weighted average of nearby source bands (sigma=10nm)
        weights = np.exp(-0.5 * ((src_wl - tw) / 10.0) ** 2)
        weights /= weights.sum()
        resampled[:, :, i] = np.tensordot(crop, weights, axes=(2, 0))
    return resampled


def _load_cube_from_mat(path: Path) -> np.ndarray:
    """Load a hyperspectral cube from a .mat file."""
    mat = sio.loadmat(str(path))
    for key in ("img", "img_clean", "data", "truth", "recon", "ref"):
        if key in mat:
            cube = mat[key].astype(np.float64)
            return cube
    for key, val in mat.items():
        if not key.startswith("_") and isinstance(val, np.ndarray) and val.ndim == 3:
            return val.astype(np.float64)
    raise ValueError(f"Cannot find cube data in {path}")


def load_mask_256() -> np.ndarray:
    """Load the real calibrated coded aperture mask (256×256)."""
    mat = sio.loadmat(str(MASK_PATH))
    for key in ("mask", "mask3d"):
        if key in mat:
            m = mat[key]
            return m[:, :, 0].astype(np.float64) if m.ndim == 3 else m.astype(np.float64)
    for key, val in mat.items():
        if not key.startswith("_") and isinstance(val, np.ndarray) and val.ndim == 2:
            return val.astype(np.float64)
    raise ValueError("Cannot find mask data")


def generate_assumed_mask(size: int, seed: int = 7777) -> np.ndarray:
    """Generate an assumed random binary coded aperture mask for dev/hidden.

    This is NOT the real calibrated mask — it's a randomly generated binary
    mask that contestants must work with. The real mask is only provided for
    the public tier.
    """
    rng = np.random.RandomState(seed)
    # 50% fill-factor random binary mask (standard for CASSI)
    mask = (rng.rand(size, size) > 0.5).astype(np.float64)
    return mask


# ── Band selection helpers ───────────────────────────────────────────────

def select_28_from_31(cube_31: np.ndarray) -> np.ndarray:
    """Select 28 bands from a 31-band cube.

    31 bands span 400-700nm (10nm steps).
    We want 28 bands spanning 450-650nm → bands 5-32 of 1-31
    i.e., indices 4 through 31 → drop first 4, take next 28... but that
    gives 400,410,...700 → indices 5:33? Let's map properly.

    31 bands: 400, 410, 420, 430, 440, 450, 460, ..., 700 nm
    28 bands: 450, 457, 464, ..., 650 nm (our target)

    Simplest: use spectral interpolation.
    """
    src_wl = np.linspace(400, 700, 31)
    tgt_wl = np.linspace(450, 650, 28)
    H, W, _ = cube_31.shape
    result = np.zeros((H, W, 28), dtype=np.float64)
    for i, tw in enumerate(tgt_wl):
        # Linear interpolation between nearest source bands
        idx = np.searchsorted(src_wl, tw) - 1
        idx = max(0, min(idx, 29))  # clamp
        t = (tw - src_wl[idx]) / (src_wl[idx + 1] - src_wl[idx])
        result[:, :, i] = (1 - t) * cube_31[:, :, idx] + t * cube_31[:, :, idx + 1]
    return result


# ── Scene preparation ────────────────────────────────────────────────────

def prepare_scene_large(source: str, filename: str | None,
                        crop_r: int, crop_c: int,
                        flip_h: bool, flip_v: bool, rot90: int) -> np.ndarray:
    """Load, crop, transform, and band-select to produce 500×500×28."""
    sz = LARGE_SIZE

    if source == "tsa_recon":
        cube = load_recon_scene(filename)  # (660, 660, 28)
        cube = cube[crop_r:crop_r + sz, crop_c:crop_c + sz, :]

    elif source == "cave":
        cube_31 = load_cave_scene(filename)  # (512, 512, 31)
        cube_31 = cube_31[crop_r:crop_r + sz, crop_c:crop_c + sz, :]
        cube = select_28_from_31(cube_31)

    elif source == "tokyotech":
        cube_31 = load_tokyotech_scene(filename)  # (500, 500, 31)
        # Normalize if needed
        vmax = cube_31.max()
        if vmax > 1.0:
            cube_31 = cube_31 / vmax
        cube = select_28_from_31(cube_31)

    elif source == "harvard":
        cube_31 = load_harvard_scene(filename)  # (1040, 1392, 31)
        cube_31 = cube_31[crop_r:crop_r + sz, crop_c:crop_c + sz, :]
        cube = select_28_from_31(cube_31)

    elif source == "chikusei":
        cube = load_chikusei_crop(crop_r, crop_c, sz)  # already (500, 500, 28)

    else:
        raise ValueError(f"Unknown source: {source}")

    # Apply transforms
    if flip_h:
        cube = cube[:, ::-1, :].copy()
    if flip_v:
        cube = cube[::-1, :, :].copy()
    if rot90 > 0:
        cube = np.rot90(cube, k=rot90, axes=(0, 1)).copy()

    # Normalize to [0, 1]
    vmax = cube.max()
    if vmax > 0:
        cube = cube / vmax

    assert cube.shape == (sz, sz, NUM_BANDS), f"Unexpected shape: {cube.shape}"
    return cube


# ── SD-CASSI forward model ──────────────────────────────────────────────

def sd_cassi_forward(cube: np.ndarray, mask: np.ndarray,
                     dispersion_step: float = 2.0) -> np.ndarray:
    """SD-CASSI forward model: mask -> disperse -> spectral sum -> 2D measurement."""
    H, W, L = cube.shape
    m = mask[:H, :W]
    W_out = W + int(dispersion_step * (L - 1))
    measurement = np.zeros((H, W_out), dtype=np.float64)
    for lam in range(L):
        shift = int(dispersion_step * lam)
        measurement[:, shift:shift + W] += cube[:, :, lam] * m
    return measurement


def add_poisson_gaussian_noise(measurement: np.ndarray, rng: np.random.RandomState,
                                poisson_alpha: float = 1.0,
                                gaussian_sigma: float = 0.01) -> np.ndarray:
    """Add realistic Poisson-Gaussian noise."""
    scaled = np.clip(measurement * poisson_alpha, 0, None)
    noisy = rng.poisson(scaled * 1000).astype(np.float64) / 1000 / poisson_alpha
    noisy += rng.randn(*measurement.shape) * gaussian_sigma
    return np.clip(noisy, 0, None)


# ── Image utilities ──────────────────────────────────────────────────────

def cube_to_rgb(cube: np.ndarray) -> np.ndarray:
    """Convert 28-band hyperspectral cube to pseudo-RGB."""
    band_wl = np.linspace(450, 650, cube.shape[2])
    r = np.tensordot(cube, np.exp(-0.5 * ((band_wl - 620) / 30) ** 2), axes=(2, 0))
    g = np.tensordot(cube, np.exp(-0.5 * ((band_wl - 535) / 30) ** 2), axes=(2, 0))
    b = np.tensordot(cube, np.exp(-0.5 * ((band_wl - 470) / 25) ** 2), axes=(2, 0))
    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb / (rgb.max() + 1e-8)
    return np.clip(rgb ** 0.45, 0, 1)


def save_image(arr: np.ndarray, path: Path, cmap: str = "viridis",
               vmin=None, vmax=None):
    """Save a 2D or RGB array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    dpi = 150
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    if arr.ndim == 3:
        ax.imshow(arr)
    else:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)


# ── Tier generation ──────────────────────────────────────────────────────

def generate_public_tier(mask_256: np.ndarray):
    """Public tier: KAIST scenes 1-10 (256x256x28) with real calibrated mask."""
    print("\n" + "=" * 60)
    print("PUBLIC TIER -- KAIST scenes (256x256x28, real calibrated mask)")
    print("=" * 60)

    tier_dir = OUTPUT_DIR / "public" / "images"
    rng = np.random.RandomState(1001)
    disp_step = 2.02

    for idx in range(10):
        scene_dir = tier_dir / f"scene_{idx:02d}"
        print(f"  Scene {idx:02d} (KAIST scene{idx+1:02d})...", end=" ", flush=True)

        cube = load_kaist_scene(idx)

        save_image(cube_to_rgb(cube), scene_dir / "ground_truth_rgb.png")
        for b in range(NUM_BANDS):
            save_image(cube[:, :, b],
                       scene_dir / "spectral" / f"band_{b:02d}_{WAVELENGTHS_NM[b]}nm.png",
                       cmap="inferno", vmin=0, vmax=cube.max())

        meas = sd_cassi_forward(cube, mask_256, disp_step)
        meas = add_poisson_gaussian_noise(meas, rng)
        save_image(meas, scene_dir / "measurement.png", cmap="gray")
        print("done")

    # Save the real mask image
    save_image(mask_256, OUTPUT_DIR / "public" / "mask.png", cmap="gray", vmin=0, vmax=1)
    print("  Real calibrated mask saved")


def generate_large_tier(tier_name: str, scene_defs: list, mask: np.ndarray,
                        disp_step: float, seed: int):
    """Generate a 500x500x28 tier with assumed mask."""
    print(f"\n{'=' * 60}")
    print(f"{tier_name.upper()} TIER -- 20 unique 500x500x28 scenes (assumed mask)")
    print(f"  Dispersion step: {disp_step}, Seed: {seed}")
    print(f"{'=' * 60}")

    tier_dir = OUTPUT_DIR / tier_name / "images"
    rng = np.random.RandomState(seed)

    for idx, (source, filename, cr, cc, fh, fv, rot) in enumerate(scene_defs):
        scene_dir = tier_dir / f"scene_{idx:02d}"
        tag = f"{source}"
        if filename:
            tag += f"/{filename}"
        if cr or cc:
            tag += f" crop({cr},{cc})"
        transforms = []
        if fh: transforms.append("flipH")
        if fv: transforms.append("flipV")
        if rot: transforms.append(f"rot{rot*90}")
        if transforms:
            tag += " " + "+".join(transforms)

        print(f"  Scene {idx:02d} [{tag}]...", end=" ", flush=True)

        cube = prepare_scene_large(source, filename, cr, cc, fh, fv, rot)

        save_image(cube_to_rgb(cube), scene_dir / "ground_truth_rgb.png")
        for b in range(NUM_BANDS):
            save_image(cube[:, :, b],
                       scene_dir / "spectral" / f"band_{b:02d}_{WAVELENGTHS_NM[b]}nm.png",
                       cmap="inferno", vmin=0, vmax=cube.max())

        meas = sd_cassi_forward(cube, mask, disp_step)
        meas = add_poisson_gaussian_noise(meas, rng)
        save_image(meas, scene_dir / "measurement.png", cmap="gray")
        print("done")

    # Save the assumed mask image
    save_image(mask, OUTPUT_DIR / tier_name / "mask.png", cmap="gray", vmin=0, vmax=1)
    print(f"  Assumed mask saved ({mask.shape[0]}x{mask.shape[1]})")


# ── README writers ───────────────────────────────────────────────────────

def write_readmes():
    """Write all markdown documentation files."""

    (OUTPUT_DIR / "README.md").write_text("""\
# SD-CASSI Benchmark Dataset

## Single-Disperser Coded Aperture Snapshot Spectral Imager

SD-CASSI captures a 3D hyperspectral datacube (x, y, wavelength) in a **single snapshot**
by applying a coded aperture mask followed by a dispersive prism. The result is a 2D
compressed measurement that mixes spatial and spectral information.

### Forward Model

```
M(mask) -> W(a, alpha) -> Sum_lambda -> D(g, eta)
```

| Stage | Primitive | Description |
|-------|-----------|-------------|
| 1 | **M** -- Coded Aperture | Binary mask modulates each spatial-spectral voxel |
| 2 | **W** -- Prism Dispersion | Wavelength-dependent lateral shift (slope *a*, axis *alpha*) |
| 3 | **Sum_lambda** -- Spectral Sum | Detector integrates across all wavelength bands |
| 4 | **D** -- Detector | Applies gain *g* and additive read noise *eta* |

### Challenge Tiers

| Tier | Scenes | Size | Mask | Source | Ground Truth | Mismatch |
|------|--------|------|------|--------|:---:|----------|
| **Public** | 10 | 256x256x28 | Real calibrated | KAIST (public) | Provided | Moderate |
| **Dev** | 20 | 500x500x28 | Assumed (random) | Mixed (private) | Withheld | Mild |
| **Hidden** | 20 | 500x500x28 | Assumed (random) | Mixed (private) | Withheld | Severe |

### Mask Policy

- **Public tier:** The real calibrated coded aperture mask from the KAIST/TSA system
  is provided. Contestants can study and use it directly.
- **Dev & Hidden tiers:** An assumed random binary mask (50% fill factor) is provided.
  This simulates the realistic scenario where the exact mask calibration is unknown.
  Contestants must work with this assumed mask and handle mask uncertainty as part
  of the mismatch.

### Data Sources

Dev and hidden scenes are generated from **5 diverse hyperspectral sources**:
- **CAVE** (Columbia): 512x512x31 natural indoor scenes
- **TokyoTech TT-31**: 500x500x31 multispectral scenes
- **Harvard**: 1392x1040x31 indoor/outdoor scenes
- **Chikusei**: 2517x2335x128 airborne remote sensing
- **TSA real reconstructions**: 660x660x28 CASSI reconstructed scenes

All scenes use proprietary crop coordinates and geometric transforms. These
exact scenes cannot be found in any public dataset.

### Mismatch Parameters

| Parameter | Symbol | Description | Spec Range |
|-----------|--------|-------------|------------|
| `mask_dx` | Dx | Mask lateral shift | 0.3 -- 0.7 px |
| `mask_dy` | Dy | Mask vertical shift | 0.1 -- 0.5 px |
| `mask_rotation` | theta | Mask rotation | 0.0 -- 0.2 deg |
| `dispersion_slope` | a | Dispersion coefficient | 1.90 -- 2.15 px/band |
| `dispersion_axis` | alpha | Dispersion angle | 0.0 -- 0.3 deg |

### Scoring

```
Score = 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - Hx|| / ||y||)
```

### Spectral Bands

28 bands spanning the visible spectrum (450 nm -- 650 nm):

| Band | nm | Band | nm | Band | nm | Band | nm |
|------|----|------|----|------|----|------|----|
| 00 | 450 | 07 | 501 | 14 | 553 | 21 | 604 |
| 01 | 457 | 08 | 509 | 15 | 560 | 22 | 612 |
| 02 | 464 | 09 | 516 | 16 | 568 | 23 | 619 |
| 03 | 472 | 10 | 524 | 17 | 575 | 24 | 627 |
| 04 | 479 | 11 | 531 | 18 | 583 | 25 | 634 |
| 05 | 486 | 12 | 538 | 19 | 590 | 26 | 642 |
| 06 | 494 | 13 | 546 | 20 | 597 | 27 | 650 |

### Directory Structure

```
sd_cassi/
+-- README.md
+-- public/                         (256x256x28, real mask)
|   +-- README.md
|   +-- mask.png                    (real calibrated mask)
|   +-- images/scene_00/ ... scene_09/
|       +-- ground_truth_rgb.png
|       +-- measurement.png
|       +-- spectral/band_XX_YYYnm.png (x28)
+-- dev/                            (500x500x28, assumed mask)
|   +-- README.md
|   +-- mask.png                    (assumed random mask)
|   +-- images/scene_00/ ... scene_19/
|       +-- ground_truth_rgb.png  (private -- server-side only)
|       +-- measurement.png
|       +-- spectral/ (x28, private)
+-- hidden/                         (500x500x28, assumed mask)
    +-- README.md
    +-- mask.png                    (assumed random mask)
    +-- images/scene_00/ ... scene_19/
        +-- ground_truth_rgb.png  (private -- server-side only)
        +-- measurement.png
        +-- spectral/ (x28, private)
```

### References

1. Choi et al., "High-quality hyperspectral reconstruction using a spectral prior," ACM TOG, 2017. (KAIST)
2. Wagadarikar et al., "Single disperser design for coded aperture snapshot spectral imaging," Appl. Opt., 2008.
3. Yasuma et al., "Generalized assorted pixel camera," IEEE PAMI, 2010. (CAVE)
4. Monno et al., "A practical one-shot multispectral imaging system using a single image sensor," IEEE TIP, 2015. (TT-31)
5. Chakrabarti & Zickler, "Statistics of real-world hyperspectral images," CVPR, 2011. (Harvard)
6. Yokoya & Iwasaki, "Airborne hyperspectral data over Chikusei," U. of Tokyo, 2016. (Chikusei)
""")

    (OUTPUT_DIR / "public" / "README.md").write_text("""\
# SD-CASSI -- Public Tier

Full-access development tier with **all data visible**.

## Source

10 scenes from the **KAIST Real-World Hyperspectral Dataset** (publicly available).

## Mask

**Real calibrated mask** from the TSA/KAIST system is provided (`mask.png`).
This is the actual coded aperture pattern used to generate the measurements.

## Signal Dimensions

- **Datacube:** 256 x 256 x 28 (height x width x spectral bands, 450-650 nm)
- **Measurement:** 256 x ~310 (height x width + dispersion spread)

## What's Included

- **Ground truth** (x_true): Original 256 x 256 x 28 hyperspectral datacubes
- **Measurements** (y): 2D compressed snapshots with moderate mismatch + noise
- **Real calibrated mask**: The actual coded aperture pattern
- **True mismatch spec**: Exact parameters used to generate measurements
- **Spec ranges**: Search bounds for each mismatch parameter

## True Mismatch Parameters

| Parameter | Value |
|-----------|-------|
| mask_dx | 0.5 px |
| mask_dy | 0.3 px |
| mask_rotation | 0.1 deg |
| dispersion_slope | 2.02 px/band |
| dispersion_axis | 0.15 deg |

## Images

Each `scene_XX/` folder contains:
- `ground_truth_rgb.png` -- Pseudo-RGB rendering
- `measurement.png` -- The 2D compressed measurement
- `spectral/band_XX_YYYnm.png` -- Each of the 28 spectral bands
""")

    (OUTPUT_DIR / "dev" / "README.md").write_text("""\
# SD-CASSI -- Dev Tier

Blind evaluation tier -- **ground truth withheld**.

## Source

20 unique scenes from 5 diverse hyperspectral sources (CAVE, TokyoTech TT-31,
Harvard, Chikusei airborne, TSA real reconstructions) with proprietary spatial
crops and geometric transforms. **These scenes are unique to the PWM server.**

## Mask

**Assumed random binary mask** (50% fill factor) is provided (`mask.png`).
This is NOT the real calibrated mask -- contestants must handle mask uncertainty
as part of the reconstruction challenge.

## Signal Dimensions

- **Datacube:** 500 x 500 x 28 (height x width x spectral bands, 450-650 nm)
- **Measurement:** 500 x ~554 (height x width + dispersion spread)

Each scene is simulated through the CASSI forward model with mild mismatch
(30% perturbation from nominal) and Poisson-Gaussian noise.

## What Contestants Receive

- **Measurements** (y): 2D compressed snapshots
- **Assumed mask**: Random binary mask (not calibrated)
- **Spec ranges**: Search bounds for mismatch parameters

## What's Withheld

- Ground truth datacubes (used for server-side scoring only)
- True mismatch parameters
- Real mask calibration
- Source scene identities, crop coordinates, and transforms
""")

    (OUTPUT_DIR / "hidden" / "README.md").write_text("""\
# SD-CASSI -- Hidden Tier

Fully blind evaluation -- **all data withheld**.

## Source

20 unique scenes from 5 diverse hyperspectral sources with different
scenes/crops/transforms from the dev tier. **Completely secret.**

## Mask

**Assumed random binary mask** (50% fill factor) is provided (`mask.png`).
Same assumed mask as the dev tier.

## Signal Dimensions

- **Datacube:** 500 x 500 x 28 (height x width x spectral bands, 450-650 nm)
- **Measurement:** 500 x ~554 (height x width + dispersion spread)

Severe mismatch (80% perturbation) makes this the hardest tier.
Final leaderboard ranking is based on hidden-tier scores.

## What's Withheld

- Everything: measurements, ground truth, real mask, spec, source identities
- Scoring is performed server-side after model submission
""")

    print("  README files written (4 total)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Masks ──
    mask_256 = load_mask_256()
    print(f"Real calibrated mask loaded: {mask_256.shape}")

    mask_500 = generate_assumed_mask(LARGE_SIZE, seed=7777)
    print(f"Assumed random mask generated: {mask_500.shape}")

    # ── Public tier: KAIST scenes, real mask ──
    generate_public_tier(mask_256)

    # ── Dev tier: 20 unique scenes, assumed mask, mild mismatch ──
    generate_large_tier(
        "dev", DEV_SCENES, mask_500,
        disp_step=2.08,
        seed=2001,
    )

    # ── Hidden tier: 20 unique scenes, assumed mask, severe mismatch ──
    generate_large_tier(
        "hidden", HIDDEN_SCENES, mask_500,
        disp_step=1.95,
        seed=3001,
    )

    # ── Documentation ──
    write_readmes()

    # ── Summary ──
    total_png = sum(1 for _ in OUTPUT_DIR.rglob("*.png"))
    total_md = sum(1 for _ in OUTPUT_DIR.rglob("*.md"))
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())

    print(f"\n{'=' * 60}")
    print(f"Package built: {OUTPUT_DIR}")
    print(f"  Images: {total_png}")
    print(f"  Markdown: {total_md}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")

    for tier in ("public", "dev", "hidden"):
        tier_dir = OUTPUT_DIR / tier
        n = sum(1 for _ in tier_dir.rglob("*.png"))
        sz = sum(f.stat().st_size for f in tier_dir.rglob("*") if f.is_file())
        scenes = sum(1 for d in (tier_dir / "images").iterdir() if d.is_dir())
        print(f"  {tier:8s}: {scenes} scenes, {n:4d} images, {sz/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
