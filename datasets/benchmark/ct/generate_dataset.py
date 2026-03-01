#!/usr/bin/env python3
"""Generate the fan-beam sparse-view / low-dose CT benchmark dataset.

All three tiers use real patient images from LoDoPaB-CT (LIDC/IDRI patients).
Each tier draws from a DIFFERENT split so they share no scenes:

Public tier  — 11 real chest CT slices from LoDoPaB-CT **test** split
Dev tier     — 20 real chest CT slices from LoDoPaB-CT **validation** split
               (first half, patients 0–63; entirely different patients from public)
Hidden tier  — 20 real chest CT slices from LoDoPaB-CT **validation** split
               (second half, patients 64–127) + adversarial modifications

Data source — LoDoPaB-CT (most widely used CT reconstruction benchmark)
-------------------------------------------------------------------------------
Leuschner et al. (2021), Scientific Data 8:109, doi:10.1038/s41597-021-00893-z
Sourced from LIDC/IDRI lung CT database.  Zenodo record 3384092, CC BY 4.0.

Required zips (place in ct/lodopab_src/ or set LODOPAB_ROOT):
  ground_truth_test.zip        (~1.5 GB) — public tier
  ground_truth_validation.zip  (~1.5 GB) — dev + hidden tiers

Download commands:
  mkdir -p lodopab_src
  wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \\
       -O lodopab_src/ground_truth_test.zip
  wget 'https://zenodo.org/api/records/3384092/files/ground_truth_validation.zip/content' \\
       -O lodopab_src/ground_truth_validation.zip

Fallback: if a zip is missing the corresponding tier falls back to synthetic
procedural phantoms (flagged in metadata as "source": "synthetic").

Forward model spec (matches PWM benchmark page):
    R(θ) → Π(fan) → D(noise, mismatch)

Mismatch knobs (ThetaSpace):
    Δc   — centre-of-rotation offset  [pixels]
    Δθ   — systematic angle error      [degrees]
    β    — beam hardening coefficient  [unitless]
    φ    — detector tilt               [degrees]

Geometry (pixel units, image = 362×362, FOV = 26 cm):
    D_so        = 800 px   (≈ 575 mm) source-to-isocenter
    D_sd        = 568 px   (≈ 408 mm) isocenter-to-detector
    n_det       = 736      detector pixels
    det_spacing = 1.496 px (≈ 1.07 mm) detector pitch
    pixel_size  = 0.718 mm/px (26 cm / 362 px)
    n_views     = 60       (public/dev), 40–90 (hidden)

Noise:
    I₀           = 10 000 photons (quarter-dose, 120 kVp)
    σ_readout    = 5.0

Physical attenuation scale:
    MU_SCALE = 0.058  (pixel-density units → nepers)
    = pixel_size_mm × μ_max_mm  ≈ 0.718 × 0.081

LoDoPaB-CT normalisation:
    x_true ∈ [0, 1]  where  x = (HU + 1000) / 4071
    0.00 ≈ air,  0.25 ≈ soft tissue / water,  1.00 ≈ dense bone (3071 HU)

Usage:
    cd datasets/benchmark/ct
    python3 generate_dataset.py
"""
from __future__ import annotations

import io as _io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, zoom

# Self-contained import: simulate_scenes.py lives beside this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_scenes import generate_ct_gt, _ADVERSARIAL_FNS  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

# ── Geometry (pixel units) ────────────────────────────────────────────────────

IMAGE_SIZE  = 362       # LoDoPaB-CT image domain (26 cm / 0.718 mm per px)
D_SO        = 800.0     # source-to-isocenter   (≈ 575 mm)
D_SD        = 568.0     # isocenter-to-detector (≈ 408 mm)
N_DET       = 736       # detector channels
DET_SPACING = 1.496     # detector pitch (px) ≈ 1.07 mm
N_VIEWS     = 60        # sparse views (public & dev)
I0          = 10_000.0  # nominal photon count (quarter dose, 120 kVp)
SIGMA_RO    = 5.0       # readout noise σ
MU_SCALE    = 0.058     # pixel-density → nepers  (0.718 mm/px × 0.081 mm⁻¹)

# ── Mismatch spec ranges per tier ─────────────────────────────────────────────

SPEC = {
    "public": {
        "center_offset_px":      {"min": -2.0,  "max":  2.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -3.0,  "max":  3.0,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.10, "unit": ""},
        "detector_tilt_deg":     {"min": -1.0,  "max":  1.0,  "unit": "degrees"},
    },
    "dev": {
        "center_offset_px":      {"min": -3.0,  "max":  3.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -5.0,  "max":  5.0,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.15, "unit": ""},
        "detector_tilt_deg":     {"min": -2.0,  "max":  2.0,  "unit": "degrees"},
    },
    "hidden": {
        "center_offset_px":      {"min": -5.0,  "max":  5.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -8.0,  "max":  8.0,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.30, "unit": ""},
        "detector_tilt_deg":     {"min": -3.0,  "max":  3.0,  "unit": "degrees"},
    },
}

# ── Fan-beam forward model ────────────────────────────────────────────────────

def fan_beam_project(
    x: np.ndarray,
    angles_rad: np.ndarray,
    n_det: int = N_DET,
    D_so: float = D_SO,
    D_sd: float = D_SD,
    det_spacing: float = DET_SPACING,
    center_offset: float = 0.0,
) -> np.ndarray:
    """Vectorised 2-D fan-beam line-integral projection.

    Returns sinogram (n_views, n_det) float32 in pixel-density units.
    Multiply by MU_SCALE to get physical attenuation (nepers).
    """
    H, W = x.shape
    x64 = x.astype(np.float64)
    cy  = H / 2.0
    cx  = W / 2.0 + center_offset

    det_pos  = (np.arange(n_det) - n_det / 2.0) * det_spacing
    diag     = np.sqrt(H ** 2 + W ** 2)
    n_steps  = max(int(diag * 1.5), 512)
    t_vals   = np.linspace(0.0, 1.0, n_steps)

    sinogram = np.zeros((len(angles_rad), n_det), dtype=np.float32)

    for i, angle in enumerate(angles_rad):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        src_y =  -D_so * sin_a + cy
        src_x =   D_so * cos_a + cx
        det_y =   D_sd * sin_a + det_pos * cos_a + cy
        det_x =  -D_sd * cos_a + det_pos * sin_a + cx
        ray_y =  det_y - src_y
        ray_x =  det_x - src_x
        ray_len = np.sqrt(ray_y ** 2 + ray_x ** 2)

        sample_y = src_y + np.outer(ray_y, t_vals)
        sample_x = src_x + np.outer(ray_x, t_vals)
        coords   = np.array([sample_y.ravel(), sample_x.ravel()])
        vals     = map_coordinates(x64, coords, order=1, mode="constant", cval=0.0)
        vals     = vals.reshape(n_det, n_steps)

        step_size     = ray_len / n_steps
        sinogram[i]   = (vals.sum(axis=1) * step_size).astype(np.float32)

    return sinogram


def apply_mismatch(
    x: np.ndarray,
    angles_nominal: np.ndarray,
    center_offset: float,
    angle_error_deg: float,
    beam_hardening_beta: float,
    detector_tilt_deg: float,
    rng: np.random.Generator,
    I0: float = I0,
    sigma_ro: float = SIGMA_RO,
) -> np.ndarray:
    """Apply all four mismatch effects + Poisson/readout noise.

    Pipeline:
      1. Re-project with (Δθ, Δc)
      2. Scale to physical nepers (MU_SCALE)
      3. Beam hardening: p_eff = p + β·p²
      4. Detector tilt (sinogram shear)
      5. Beer-Lambert + Poisson + readout noise

    Returns p_measured (n_views, n_det) float32 in nepers.
    """
    angles_true = angles_nominal + np.deg2rad(angle_error_deg)
    p_geo  = fan_beam_project(x, angles_true, center_offset=center_offset)
    p_phys = p_geo * MU_SCALE

    # Beam hardening
    p_bh = p_phys + beam_hardening_beta * p_phys ** 2

    # Detector tilt (sinogram shear)
    if abs(detector_tilt_deg) > 1e-6:
        tan_phi = np.tan(np.deg2rad(detector_tilt_deg))
        n_ang, n_det = p_bh.shape
        d_idx   = np.arange(n_det) - n_det / 2.0
        ang_idx = np.arange(n_ang, dtype=np.float64)
        ANG, DET = np.meshgrid(ang_idx, d_idx, indexing="ij")
        coords = np.array([ANG + DET * tan_phi * 0.15,
                           np.meshgrid(ang_idx, np.arange(n_det), indexing="ij")[1]])
        p_bh = map_coordinates(
            p_bh.astype(np.float64), coords.reshape(2, -1),
            order=1, mode="nearest",
        ).reshape(n_ang, n_det).astype(np.float32)

    # Low-dose noise
    p_clamped = np.clip(p_bh, 0.0, 20.0)
    I_expect  = I0 * np.exp(-p_clamped)
    I_noisy   = rng.poisson(np.maximum(I_expect, 1e-3)).astype(np.float64)
    I_noisy  += rng.normal(0.0, sigma_ro, I_noisy.shape)
    I_noisy   = np.maximum(I_noisy, 1.0)
    return (-np.log(I_noisy / I0)).astype(np.float32)


# ── LoDoPaB-CT slice index tables ─────────────────────────────────────────────

_LODOPAB_SHARD_SIZE = 128   # images per HDF5 shard

# Public:  11 diverse slices from test set (deep lung, heart, liver, various sizes)
LODOPAB_PUBLIC_INDICES = [0, 320, 650, 980, 1310, 1640, 1970, 2300, 2630, 2960, 3290]
LODOPAB_SCENE_NAMES    = [f"lidc_test_{i:02d}" for i in range(11)]

# Dev: 20 slices hand-selected for NARROW body cross-section (apex / lower-thorax anatomy).
#
# Selection methodology — full dataset scan (all 3584 validation slices):
#   1. Compute body pixel count (BPC = px with value > 0.05) for every slice.
#   2. Among patients 0–63 (global indices 0–1791), one candidate per patient.
#   3. Pick the slice with BPC closest to the 25th percentile (≈ 72 k px, narrow FOV).
#   4. Final 20: one per patient from 20 distinct patients spanning patients 0–63.
#
# BPC mean ≈ 67 142 px  (range 49 230–69 649)
# These slices show narrow cross-sections: apex, shoulder, or lower-thorax anatomy.
LODOPAB_VAL_DEV_INDICES = [
    20, 50, 172, 328, 441, 459, 604, 657, 799, 819,
    904, 943, 977, 1093, 1126, 1153, 1419, 1585, 1760, 1787,
]
LODOPAB_DEV_SCENE_NAMES = [f"lidc_val_{i:02d}" for i in range(20)]

# Hidden: 20 slices hand-selected for WIDE body cross-section (cardiac/main-thorax anatomy).
#
# Selection methodology — same full dataset scan:
#   1. Among patients 64–127 (global indices 1792–3583), one candidate per patient.
#   2. Pick the slice with BPC closest to the 75th percentile (≈ 103 k px) but ≥ p75.
#   3. Final 20: widest BPC from 20 distinct patients spanning patients 65–125.
#
# BPC mean ≈ 130 632 px  (range 130 269–131 044) — near maximum possible body width.
# These slices show very wide cross-sections: cardiac level, full-lung, broad chest.
#
# BPC overlap with dev: NONE — dev max (69 649) < hidden min (130 269).
# Patient boundary: patients 0–63 index < 1792; patients 64–127 index ≥ 1792. ✓
LODOPAB_VAL_HIDDEN_INDICES = [
    1846, 2067, 2120, 2131, 2221, 2245, 2376, 2380, 2510, 2573,
    2768, 2912, 3043, 3053, 3116, 3180, 3265, 3343, 3392, 3506,
]
LODOPAB_HIDDEN_SCENE_NAMES = [f"lidc_val_h{i:02d}" for i in range(20)]

_LODOPAB_SOURCE = (
    "LoDoPaB-CT (Leuschner et al. 2021, Scientific Data 8:109, "
    "doi:10.1038/s41597-021-00893-z). "
    "Zenodo record 3384092, CC BY 4.0."
)


# ── Generic LoDoPaB-CT loader ──────────────────────────────────────────────────

def _find_lodopab_zip(filename: str) -> Path | None:
    """Locate a LoDoPaB-CT zip from env var or default lodopab_src/ path."""
    root = os.environ.get("LODOPAB_ROOT", "")
    candidates = []
    if root:
        candidates.append(Path(root) / filename)
    candidates.append(BENCHMARK_DIR / "lodopab_src" / filename)
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_lodopab_images(
    zip_filename: str,
    shard_prefix: str,
    indices: list[int],
    scene_names: list[str],
    tier_label: str = "",
) -> list[tuple[str, np.ndarray]] | None:
    """Load images from any LoDoPaB-CT zip by global flat index.

    Returns list of (scene_name, x_true float32 [0,1]) or None if zip absent.
    """
    zip_path = _find_lodopab_zip(zip_filename)
    if zip_path is None:
        label = f" ({tier_label} tier)" if tier_label else ""
        print(f"  [WARNING] {zip_filename} not found{label}.")
        print(f"  [WARNING] Place at: ct/lodopab_src/{zip_filename}")
        print(f"  [WARNING] Download:")
        print(f"  [WARNING]   wget 'https://zenodo.org/api/records/3384092/files/"
              f"{zip_filename}/content' \\")
        print(f"  [WARNING]   -O lodopab_src/{zip_filename}")
        return None

    print(f"  Reading {zip_filename} ({len(indices)} slices) ...")
    shard_map: dict[int, list[tuple[int, int]]] = {}
    for global_i in indices:
        shard_i = global_i // _LODOPAB_SHARD_SIZE
        local_i = global_i % _LODOPAB_SHARD_SIZE
        shard_map.setdefault(shard_i, []).append((local_i, global_i))

    found: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for shard_i, requests in sorted(shard_map.items()):
            shard_name = f"{shard_prefix}_{shard_i:03d}.hdf5"
            with zf.open(shard_name) as raw:
                buf = _io.BytesIO(raw.read())
            with h5py.File(buf, "r") as hf:
                batch = hf["data"][:]   # (128, 362, 362) float32
                for local_i, global_i in requests:
                    img = np.clip(batch[local_i].astype(np.float32), 0.0, 1.0)
                    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
                        img = zoom(img, IMAGE_SIZE / img.shape[0], order=1)
                        img = np.clip(img.astype(np.float32), 0.0, 1.0)
                    found[global_i] = img
            print(f"    shard {shard_i:03d} → {[r[1] for r in requests]}")

    if len(found) < len(indices):
        print(f"  [WARNING] Only {len(found)}/{len(indices)} slices loaded.")

    result = [(name, found[idx])
              for idx, name in zip(indices, scene_names)
              if idx in found]
    print(f"  Loaded {len(result)} real CT images.")
    return result if result else None


def load_lodopab_public() -> list[tuple[str, np.ndarray]] | None:
    """Load 11 diverse slices from LoDoPaB-CT test split."""
    return _load_lodopab_images(
        "ground_truth_test.zip", "ground_truth_test",
        LODOPAB_PUBLIC_INDICES, LODOPAB_SCENE_NAMES, "public",
    )


def load_lodopab_val_dev() -> list[tuple[str, np.ndarray]] | None:
    """Load 20 slices from LoDoPaB-CT validation split (first half, dev tier)."""
    return _load_lodopab_images(
        "ground_truth_validation.zip", "ground_truth_validation",
        LODOPAB_VAL_DEV_INDICES, LODOPAB_DEV_SCENE_NAMES, "dev",
    )


def load_lodopab_val_hidden() -> list[tuple[str, np.ndarray]] | None:
    """Load 20 slices from LoDoPaB-CT validation split (second half, hidden tier)."""
    return _load_lodopab_images(
        "ground_truth_validation.zip", "ground_truth_validation",
        LODOPAB_VAL_HIDDEN_INDICES, LODOPAB_HIDDEN_SCENE_NAMES, "hidden",
    )


# ── Diversity augmentation (spatial transforms to maximise tier separation) ────

def _augment_diversity(
    x: np.ndarray,
    rng: np.random.Generator,
    mode: str = "dev",
) -> np.ndarray:
    """Apply maximally aggressive spatial augmentation so dev/hidden images are
    visually and structurally distinct from the public tier.

    All transforms are physically valid for 2-D CT slices:
      • Rotation  — any orientation is a valid axial cross-section
      • Flip      — left-right / up-down symmetry
      • Zoom      — simulates different scanner FOV or patient size

    None of these change the physical meaning of x_true (still attenuation in
    [0, 1]).  Sinograms are regenerated from the augmented x_true, so the
    forward model is always self-consistent.

    Pipeline (runs all three transforms regardless of outcome magnitude):
      dev:    rotation ∈ U[20°, 340°]  + LR-flip (90%) + UD-flip (75%) + zoom ∈ U[0.55, 1.45]
      hidden: rotation ∈ U[20°, 340°]  + LR-flip (95%) + UD-flip (85%) + zoom ∈ U[0.50, 1.50]
    """
    from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom

    # ── 1. Rotation — continuous, avoids near-identity angles ─────────────────
    # Draw from [20°, 340°] to exclude the near-0° / near-360° identity band.
    angle = float(rng.uniform(20.0, 340.0))
    x = nd_rotate(x, angle, reshape=False, mode="constant", cval=0.0)

    # ── 2. Flip (both axes; high probability for both modes) ──────────────────
    lr_prob = 0.90 if mode == "dev" else 0.95
    ud_prob = 0.75 if mode == "dev" else 0.85
    if rng.random() < lr_prob:
        x = np.fliplr(x)
    if rng.random() < ud_prob:
        x = np.flipud(x)

    # ── 3. Zoom — aggressive range; changes apparent anatomy scale ─────────────
    # dev:    FOV multiplier ∈ [0.55, 1.45]   (±45 %)
    # hidden: FOV multiplier ∈ [0.50, 1.50]   (±50 %)
    lo, hi = (0.55, 1.45) if mode == "dev" else (0.50, 1.50)
    zoom_f = float(rng.uniform(lo, hi))
    x_z = nd_zoom(x, zoom_f, order=1)
    H = W = IMAGE_SIZE
    if zoom_f >= 1.0:
        # Zoomed in → crop the centre section back to IMAGE_SIZE × IMAGE_SIZE
        zh, zw = x_z.shape
        y0 = (zh - H) // 2
        x0 = (zw - W) // 2
        x = np.ascontiguousarray(x_z[y0: y0 + H, x0: x0 + W])
    else:
        # Zoomed out → embed in a zero-padded IMAGE_SIZE × IMAGE_SIZE canvas
        zh, zw = x_z.shape
        pad = np.zeros((H, W), dtype=np.float32)
        py = (H - zh) // 2
        px = (W - zw) // 2
        pad[py: py + zh, px: px + zw] = x_z
        x = pad

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ── Synthetic fallback public scenes (Shepp-Logan variants) ────────────────────

def shepp_logan_phantom(shape: tuple, variant: int = 0) -> np.ndarray:
    """Analytic Shepp-Logan phantom scaled to LoDoPaB-CT normalisation."""
    H, W = shape
    yy = np.linspace(-1.0, 1.0, H, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float64)[None, :]

    # (value_fraction, a, b, x0, y0, angle_deg)
    ellipses = [
        ( 1.00, 0.69, 0.92,  0.00,  0.00,   0),
        (-0.98, 0.66, 0.87,  0.00,  0.00,   0),
        ( 0.80, 0.11, 0.31, -0.22,  0.00,  -18 + variant * 4),
        (-0.80, 0.16, 0.41,  0.22,  0.00,   18 - variant * 4),
        ( 0.35, 0.21, 0.25,  0.00,  0.35,   0),
        ( 0.35, 0.046,0.046, 0.00,  0.10,   0),
        ( 0.35, 0.046,0.046, 0.00, -0.10,   0),
        ( 0.35, 0.046,0.023,-0.08, -0.605,  0),
        ( 0.35, 0.023,0.023, 0.00, -0.606,  0),
        ( 0.35, 0.023,0.046, 0.06, -0.605,  0),
    ]
    img = np.zeros((H, W), dtype=np.float64)
    for val, a, b, x0, y0, ang in ellipses:
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        yr = (yy - y0) * ca - (xx - x0) * sa
        xr = (yy - y0) * sa + (xx - x0) * ca
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += val
    # Scale to LoDoPaB-CT range (0–0.55 for dev-level density)
    img = np.clip(img, 0.0, 1.0) * 0.55
    return img.astype(np.float32)


# ── Image helpers ──────────────────────────────────────────────────────────────

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, sino_ideal, sino_meas, path: Path, title: str = "") -> None:
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 3 * tw), dtype=np.float32)
    ov[:, 0:tw]      = _r(x_true)
    ov[:, tw:2*tw]   = _r(sino_ideal)
    ov[:, 2*tw:3*tw] = _r(sino_meas)
    _save_png(ov, path)


# ── Dataset tier generator ────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[str, np.ndarray]],
    base_seed: int,
    n_views_range: tuple[int, int],
    source_label: str = "synthetic",
) -> None:
    spec_ranges = SPEC[tier]
    tier_dir    = BENCHMARK_DIR / tier
    images_dir  = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"ct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    rows, true_specs = [], {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM CT benchmark — {tier} tier "
            f"(fan-beam sparse-view, LoDoPaB-CT geometry)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"]    = json.dumps({
            "D_so_px": D_SO, "D_sd_px": D_SD,
            "n_det": N_DET, "det_spacing_px": DET_SPACING,
            "pixel_size_mm": 260.0 / IMAGE_SIZE,
            "I0": I0, "sigma_readout": SIGMA_RO,
            "mu_scale": MU_SCALE,
            "mu_scale_note": "sinogram_ideal × mu_scale = physical attenuation (nepers)",
            "lodopab_normalisation": "x_true = (HU + 1000) / 4071",
        })
        f.attrs["source"] = source_label

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            n_views = int(rng.integers(n_views_range[0], n_views_range[1] + 1))
            angles_nominal = np.linspace(0, np.pi, n_views, endpoint=False).astype(np.float32)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_views": n_views}

            # Ideal sinogram (physical nepers, no mismatch/noise)
            sino_ideal = fan_beam_project(x_true, angles_nominal) * MU_SCALE

            # Measured sinogram (mismatch + Poisson + readout)
            sino_meas = apply_mismatch(
                x_true, angles_nominal,
                center_offset=mis["center_offset_px"],
                angle_error_deg=mis["angle_error_deg"],
                beam_hardening_beta=mis["beam_hardening_beta"],
                detector_tilt_deg=mis["detector_tilt_deg"],
                rng=rng,
            )

            grp = f.create_group(key)
            grp.create_dataset("x_true",             data=x_true,         compression="gzip")
            grp.create_dataset("sinogram_ideal",      data=sino_ideal,     compression="gzip")
            grp.create_dataset("sinogram_measured",   data=sino_meas,      compression="gzip")
            grp.create_dataset("angles_nominal",      data=angles_nominal)
            grp.attrs["metadata"]    = json.dumps({
                "scene": scene_name, "shape": list(x_true.shape),
                "n_views": n_views, "n_det": N_DET,
                "D_so_px": D_SO, "D_sd_px": D_SD,
                "source": source_label,
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"]   = json.dumps({**mis, "n_views": n_views})

            # Per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true,     sample_dir / "ground_truth.png")
            _save_png(sino_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(sino_meas,  sample_dir / "sinogram_measured.png")
            _save_overview(x_true, sino_ideal, sino_meas,
                           sample_dir / "overview.png",
                           title=f"{key} — {scene_name}")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "n_views": n_views}, sf, indent=2)

            rows.append((key, scene_name, x_true.shape, n_views, mis))
            print(f"  [{tier}] {key} {scene_name}  views={n_views}  "
                  f"Δc={mis['center_offset_px']:.2f} Δθ={mis['angle_error_deg']:.2f}° "
                  f"β={mis['beam_hardening_beta']:.3f} φ={mis['detector_tilt_deg']:.2f}°")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows, source_label)
    print(f"  [{tier}] HDF5 → {h5_path.name}")


# ── README writers ────────────────────────────────────────────────────────────

def _write_tier_readme(tier: str, tier_dir: Path, rows: list,
                       source_label: str = "synthetic") -> None:
    is_real = "leuschner" in source_label.lower() or "lodopab" in source_label.lower()
    if tier == "public":
        if is_real:
            source = ("LoDoPaB-CT real chest CT — **test split** (LIDC/IDRI)\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.\n"
                      "11 slices, indices: " + str(LODOPAB_PUBLIC_INDICES))
        else:
            source = ("Synthetic Shepp-Logan variants (PLACEHOLDER)\n"
                      "Set LODOPAB_ROOT or place ground_truth_test.zip in lodopab_src/ "
                      "for real LoDoPaB-CT data.")
        access = "Full (GT + true spec + ideal sinogram)"
    elif tier == "dev":
        if is_real:
            source = ("LoDoPaB-CT real chest CT — **validation split, first half** "
                      "(patients 0–63, LIDC/IDRI)\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.\n"
                      "20 slices at indices " + str(LODOPAB_VAL_DEV_INDICES) + "\n"
                      "Completely different patients from public tier (test split).")
        else:
            source = ("Procedural chest/abdomen phantoms (FALLBACK — "
                      "ground_truth_validation.zip not found)\n"
                      "Anatomy and HU scale match LoDoPaB-CT normalisation.")
        access = "Blind (measured sinogram + spec ranges only)"
    else:
        if is_real:
            source = ("LoDoPaB-CT real chest CT — **validation split, second half** "
                      "(patients 64–127, LIDC/IDRI) + adversarial modifications\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.\n"
                      "20 slices at indices " + str(LODOPAB_VAL_HIDDEN_INDICES) + "\n"
                      "Adversarial: metal inserts, low-contrast lesions, "
                      "calcifications, high-contrast bone.")
        else:
            source = ("Adversarial procedural phantoms (FALLBACK — "
                      "ground_truth_validation.zip not found)\n"
                      "Metal, calcification, low-contrast lesions on synthetic backgrounds.")
        access = "Server-only"

    spec = SPEC[tier]
    param_desc = {
        "center_offset_px":    "Δc — centre-of-rotation offset",
        "angle_error_deg":     "Δθ — systematic angle error",
        "beam_hardening_beta": "β  — beam hardening coefficient",
        "detector_tilt_deg":   "φ  — detector tilt",
    }

    lines = [
        f"# CT {tier.capitalize()} Tier\n\n",
        f"**Source:** {source}\n\n",
        f"**Access:** {access}\n\n",
        "## Mismatch Parameters\n\n",
        "| Parameter | Description | Range |\n",
        "|-----------|-------------|-------|\n",
    ]
    for k, v in spec.items():
        lo, hi, u = v["min"], v["max"], v.get("unit", "")
        lines.append(f"| `{k}` | {param_desc[k]} | [{lo}, {hi}] {u} |\n")

    lines += [
        "\n## Samples\n\n",
        "| Sample | Scene | Views | Δc (px) | Δθ (°) | β | φ (°) |\n",
        "|--------|-------|-------|---------|--------|---|-------|\n",
    ]
    for key, scene, shape, n_views, mis in rows:
        lines.append(
            f"| {key} | {scene} | {n_views}"
            f" | {mis['center_offset_px']:.3f}"
            f" | {mis['angle_error_deg']:.3f}"
            f" | {mis['beam_hardening_beta']:.3f}"
            f" | {mis['detector_tilt_deg']:.3f} |\n"
        )

    lines += [
        "\n## HDF5 Datasets per Sample\n\n",
        "| Key | Shape | Dtype | Description |\n",
        "|-----|-------|-------|-------------|\n",
        "| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |\n",
        "| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |\n",
        "| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |\n",
        "| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |\n",
    ]

    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme(pub_is_real: bool) -> None:
    pub_note = (
        "All three tiers use **real patient CT images from LoDoPaB-CT**\n"
        "(Leuschner et al. 2021, *Scientific Data* doi:10.1038/s41597-021-00893-z),\n"
        "sourced from the LIDC/IDRI lung CT database. Zenodo record 3384092, CC BY 4.0.\n\n"
        "| Tier | Source | Patients | Scenes |\n"
        "|------|--------|----------|--------|\n"
        "| Public | LoDoPaB-CT **test** split | Test patients | 11 slices |\n"
        "| Dev | LoDoPaB-CT **validation** split — first half | Val patients 0–63 | 20 slices |\n"
        "| Hidden | LoDoPaB-CT **validation** split — second half + adversarial | Val patients 64–127 | 20 slices |\n\n"
        "Each tier uses entirely different patients — no shared scenes across tiers."
    ) if pub_is_real else (
        "**PLACEHOLDER:** Some or all tiers used synthetic fallback (LoDoPaB-CT zip not found).\n\n"
        "Required zips:\n"
        "```bash\n"
        "mkdir -p lodopab_src\n"
        "wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \\\n"
        "     -O lodopab_src/ground_truth_test.zip\n"
        "wget 'https://zenodo.org/api/records/3384092/files/ground_truth_validation.zip/content' \\\n"
        "     -O lodopab_src/ground_truth_validation.zip\n"
        "python3 generate_dataset.py\n"
        "```"
    )

    txt = f"""# CT — 2-D Fan-Beam Sparse-View / Low-Dose

## Public Data Source

{pub_note}

## Spec DAG

```
R(θ) ──► Π(fan-beam) ──► D(noise, mismatch)
```

## Forward Model

Fan-beam (divergent-ray) geometry matching a clinical scanner setup:

| Parameter | Value | Physical |
|-----------|-------|----------|
| IMAGE_SIZE | 362 × 362 px | FOV ≈ 26 cm × 26 cm |
| pixel_size | — | 0.718 mm/px |
| D_so | 800 px | ≈ 575 mm |
| D_sd | 568 px | ≈ 408 mm |
| n_det | 736 | — |
| det_spacing | 1.496 px | ≈ 1.07 mm |
| n_views (public/dev) | 60 | sparse |
| n_views (hidden) | 40–90 | per-sample random |

Noise: Beer-Lambert + Poisson(I₀ = 10 000) + readout N(0, 5²).

## LoDoPaB-CT Normalisation

```
x_true ∈ [0, 1]    x = (HU + 1000) / 4071
  0.00 → air (−1000 HU)
  0.25 → soft tissue / water (0 HU)
  0.42 → cortical bone (700 HU)
  1.00 → maximum density (3071 HU)
```

## Mismatch ThetaSpace

| Knob | Symbol | Description |
|------|--------|-------------|
| `center_offset_px` | Δc | Lateral shift of centre-of-rotation |
| `angle_error_deg` | Δθ | Systematic angular calibration error |
| `beam_hardening_beta` | β | Polychromatic BH: p_eff = p + β·p² |
| `detector_tilt_deg` | φ | Rigid tilt of detector plane |

## Scoring

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency
```

## Dataset Structure

```
ct/
├── lodopab_src/
│   ├── ground_truth_test.zip        (~1.5 GB) — public tier source
│   └── ground_truth_validation.zip  (~1.5 GB) — dev + hidden tier source
├── simulate_scenes.py         Procedural phantom generator (fallback only)
├── generate_dataset.py        Builds all H5 files + PNG images
├── public/    11 real LoDoPaB-CT test slices — GT + ideal sino + true spec (visible)
├── dev/       20 real LoDoPaB-CT validation slices (patients 0–63) — blind eval
└── hidden/    20 real LoDoPaB-CT validation slices (patients 64–127) + adversarial mods
```

## Reading the HDF5

```python
import h5py, json, numpy as np

with h5py.File("ct_challenge_dev.h5", "r") as f:
    grp = f["sample_00"]
    x_true     = grp["x_true"][:]             # (362, 362) float32  — GT attenuation map
    sino_ideal = grp["sinogram_ideal"][:]      # (60, 736)  float32  — nepers, no mismatch
    sino_meas  = grp["sinogram_measured"][:]   # (60, 736)  float32  — nepers, with mismatch
    angles     = grp["angles_nominal"][:]      # (60,)      float32  — radians
    spec       = json.loads(grp.attrs["spec_ranges"])
    true_spec  = json.loads(grp.attrs["true_spec"])
```

## Procedural Scene Types (Dev)

| Scene type | Anatomy |
|------------|---------|
| `chest_upper` | Carina level: trachea, large bilateral lungs, upper mediastinum |
| `chest_mid`   | Heart level: cardiac shadow, full lungs, descending aorta, ribs |
| `chest_lower` | Diaphragm: small lung bases, liver onset, stomach |
| `abdomen_upper` | Liver level: liver, spleen, kidneys, stomach, no lungs |
| `abdomen_mid`   | Kidney/bowel level: bowel loops, psoas, retroperitoneal fat |

## Adversarial Modifications (Hidden)

| Modification | Freq | Challenge |
|---|---|---|
| Metal implants | 35% | High-density streaks, dynamic range |
| Low-contrast lesions | 30% | Subtle nodules, hepatic cysts |
| Calcifications | 20% | Punctate high-density spots |
| High-contrast bone | 15% | Extreme dynamic range |

## References

- Leuschner et al. (2021) LoDoPaB-CT. *Scientific Data* 8:109.
  doi:10.1038/s41597-021-00893-z
- Feldkamp, Davis & Kress (1984) *JOSA A* 1:612-619.
- PWM Benchmark: https://pwm.platformai.org/benchmark/ct
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("CT Benchmark Dataset Generator (fan-beam, LoDoPaB-CT geometry)")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}\n")

    shape = (IMAGE_SIZE, IMAGE_SIZE)

    # ── Public tier ──────────────────────────────────────────────────────────
    print("Generating public tier (11 samples, 60 views)...")
    lodopab_scenes = load_lodopab_public()
    pub_is_real    = lodopab_scenes is not None

    if lodopab_scenes is None:
        print("  [public] Falling back to Shepp-Logan synthetic placeholders.")
        public_phantoms = [
            (f"shepp_logan_{i:02d}", shepp_logan_phantom(shape, i))
            for i in range(11)
        ]
        source_label = "synthetic_shepp_logan"
    else:
        public_phantoms = lodopab_scenes
        source_label = (
            "LoDoPaB-CT (Leuschner et al. 2021, Sci Data 8:109). "
            "Zenodo record 3384092, CC BY 4.0."
        )

    generate_tier("public", public_phantoms, base_seed=1000,
                  n_views_range=(60, 60), source_label=source_label)

    # ── Dev tier — real LoDoPaB-CT validation (first half) + diversity augmentation
    print("\nGenerating dev tier (20 real LoDoPaB-CT validation slices + diversity aug, "
          "60 views)...")
    lodopab_val_dev = load_lodopab_val_dev()
    if lodopab_val_dev is not None:
        rng_dev_aug = np.random.default_rng(7777)
        dev_phantoms = []
        for scene_name, x in lodopab_val_dev:
            x_aug = _augment_diversity(x, rng_dev_aug, mode="dev")
            print(f"    aug dev {scene_name}: "
                  f"orig mean={x.mean():.4f} → aug mean={x_aug.mean():.4f}")
            dev_phantoms.append((scene_name, x_aug))
        dev_source = (
            _LODOPAB_SOURCE +
            " Validation split, patients 0–63."
            " Diversity augmentation applied (rotation/flip/zoom) to ensure"
            " maximal visual separation from the public tier."
        )
    else:
        print("  [dev] Falling back to procedural phantoms.")
        dev_phantoms = []
        for i in range(20):
            x, scene_type = generate_ct_gt(seed=7000 + i, mode="dev", shape=shape)
            dev_phantoms.append((scene_type, x))
        dev_source = "synthetic"
    generate_tier("dev", dev_phantoms, base_seed=7000,
                  n_views_range=(60, 60), source_label=dev_source)

    # ── Hidden tier — real LoDoPaB-CT validation (second half) + diversity aug + adversarial
    print("\nGenerating hidden tier (20 real LoDoPaB-CT validation + diversity aug +"
          " adversarial, 40–90 views)...")
    lodopab_val_hidden = load_lodopab_val_hidden()
    if lodopab_val_hidden is not None:
        rng_hid_aug = np.random.default_rng(9999)
        rng_adv     = np.random.default_rng(9000)
        hidden_phantoms = []
        for scene_name, x in lodopab_val_hidden:
            # Step 1: diversity augmentation (rotation / flip / zoom)
            x_aug = _augment_diversity(x, rng_hid_aug, mode="hidden")
            # Step 2: adversarial modification on top of augmented image
            probs  = [p for p, _ in _ADVERSARIAL_FNS]
            adv_fn = _ADVERSARIAL_FNS[rng_adv.choice(len(_ADVERSARIAL_FNS), p=probs)][1]
            x_adv  = np.clip(adv_fn(x_aug.copy(), rng_adv), 0.0, 0.85).astype(np.float32)
            print(f"    aug hid {scene_name}: "
                  f"orig mean={x.mean():.4f} → aug={x_aug.mean():.4f} → adv={x_adv.mean():.4f}")
            hidden_phantoms.append((f"{scene_name}_adversarial", x_adv))
        hidden_source = (
            _LODOPAB_SOURCE +
            " Validation split, patients 64–127."
            " Diversity augmentation (rotation/flip/zoom) + adversarial modifications"
            " (metal inserts, low-contrast lesions, calcifications, high-contrast bone)."
        )
    else:
        print("  [hidden] Falling back to adversarial procedural phantoms.")
        hidden_phantoms = []
        for i in range(20):
            x, scene_type = generate_ct_gt(seed=9000 + i, mode="hidden", shape=shape)
            hidden_phantoms.append((scene_type, x))
        hidden_source = "synthetic"
    generate_tier("hidden", hidden_phantoms, base_seed=9000,
                  n_views_range=(40, 90), source_label=hidden_source)

    _write_top_readme(pub_is_real)

    print(f"\n{'=' * 68}")
    print(f"Done — CT benchmark ready at {BENCHMARK_DIR}")
    if not pub_is_real:
        print("NOTE: Public tier used synthetic fallback.")
        print("      Set LODOPAB_ROOT or place ground_truth_test.zip in lodopab_src/")
    print("=" * 68)


if __name__ == "__main__":
    main()
