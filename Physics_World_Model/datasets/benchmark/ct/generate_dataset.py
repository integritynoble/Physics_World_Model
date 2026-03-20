#!/usr/bin/env python3
"""Generate the parallel-beam sparse-view / low-dose CT benchmark dataset.

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
    R(theta) -> radon(parallel-beam) -> D(noise, mismatch)

Geometry (parallel-beam):
    IMAGE_SIZE  = 362      pixels (26 cm FOV, 0.718 mm/px)
    n_det       = ~512     auto-computed by skimage (circle=False)
    n_views     = 60       (public/dev), 40-90 (hidden)
    angles      = linspace(0, 180, n_views, endpoint=False) degrees

Noise:
    Poisson equivalent: sigma = max(sinogram_ideal) / sqrt(I0)
    I0 = 100_000 photons

Mismatch knobs (ThetaSpace):
    Delta_c  — centre-of-rotation offset  [pixels]
    Delta_theta — systematic angle error  [degrees]
    beta     — beam hardening coefficient [unitless] (applied to parallel sino)
    phi      — detector tilt              [degrees]

LoDoPaB-CT normalisation:
    x_true in [0, 1]  where  x = (HU + 1000) / 4071
    0.00 = air,  0.25 = soft tissue / water,  1.00 = dense bone (3071 HU)

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

# ── Geometry (parallel-beam) ──────────────────────────────────────────────────

IMAGE_SIZE = 362       # LoDoPaB-CT image domain (26 cm / 0.718 mm per px)
N_VIEWS    = 60        # sparse views (public & dev)
I0         = 100_000.0 # nominal photon count (low dose) — matches LoDoPaB reference
SIGMA_RO   = 0.0       # readout noise sigma (not used for parallel-beam; keep for compat)

# ── Mismatch spec ranges per tier ─────────────────────────────────────────────

SPEC = {
    "public": {
        "center_offset_px":      {"min": -1.0,  "max":  1.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -1.5,  "max":  1.5,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.08, "unit": ""},
        "detector_tilt_deg":     {"min": -0.8,  "max":  0.8,  "unit": "degrees"},
    },
    "dev": {
        "center_offset_px":      {"min": -3.0,  "max":  3.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -3.0,  "max":  3.0,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.15, "unit": ""},
        "detector_tilt_deg":     {"min": -1.5,  "max":  1.5,  "unit": "degrees"},
    },
    "hidden": {
        "center_offset_px":      {"min": -5.0,  "max":  5.0,  "unit": "pixels"},
        "angle_error_deg":       {"min": -5.0,  "max":  5.0,  "unit": "degrees"},
        "beam_hardening_beta":   {"min":  0.0,  "max":  0.25, "unit": ""},
        "detector_tilt_deg":     {"min": -2.5,  "max":  2.5,  "unit": "degrees"},
    },
}

# ── Parallel-beam forward model ───────────────────────────────────────────────

def parallel_beam_project(
    x: np.ndarray,
    angles_deg: np.ndarray,
    center_offset: float = 0.0,
) -> np.ndarray:
    """Parallel-beam Radon projection using skimage.

    Returns sinogram (n_views, n_det) float32 in density*pixel units.
    Matches skimage.transform.iradon exactly (circle=False).

    center_offset: shifts the image laterally before projection to simulate
    centre-of-rotation offset.
    """
    from skimage.transform import radon

    x_f = x.astype(np.float64)

    if abs(center_offset) > 0.5:
        # Apply integer+subpixel center offset via roll+subpixel shift
        shift_int  = int(round(center_offset))
        shift_frac = center_offset - shift_int
        x_f = np.roll(x_f, shift_int, axis=1)
        if abs(shift_frac) > 1e-6:
            # Subpixel linear blend
            if shift_frac > 0:
                x_f = (1.0 - shift_frac) * x_f + shift_frac * np.roll(x_f, 1, axis=1)
            else:
                x_f = (1.0 + shift_frac) * x_f - shift_frac * np.roll(x_f, -1, axis=1)

    # radon returns (n_det, n_views); transpose to (n_views, n_det)
    sino = radon(x_f, theta=angles_deg, circle=False).T
    return sino.astype(np.float32)


def apply_mismatch(
    x: np.ndarray,
    angles_nominal_deg: np.ndarray,
    center_offset: float,
    angle_error_deg: float,
    beam_hardening_beta: float,
    detector_tilt_deg: float,
    rng: np.random.Generator,
    I0: float = I0,
) -> np.ndarray:
    """Apply all four mismatch effects + Poisson shot noise.

    Pipeline:
      1. Re-project with (Delta_theta, Delta_c) perturbations
      2. Beam hardening on normalised sinogram: p_norm in [0,1], then p_bh = p*(1 + beta*p)
         (equivalent to p + beta*p^2 but in normalised units so beta in [0,0.3] is physical)
      3. Detector tilt (sinogram shear)
      4. Poisson shot noise equivalent: sigma = max(sino_ideal) / sqrt(I0)

    Returns sinogram_measured (n_views, n_det) float32 in same units as input sino.
    """
    angles_true = angles_nominal_deg + angle_error_deg
    p = parallel_beam_project(x, angles_true, center_offset=center_offset)

    # Beam hardening applied in normalized units so that beta in [0, 0.3] is meaningful.
    # Normalise p to [0,1], apply BH, then scale back.
    p_max = float(np.max(p))
    if p_max > 1e-8 and beam_hardening_beta > 1e-8:
        p_norm = p / p_max
        # BH: p_eff = p_norm * (1 + beta * p_norm) — mild nonlinearity, max distortion <= beta
        p_bh = p_max * p_norm * (1.0 + beam_hardening_beta * p_norm)
    else:
        p_bh = p.copy()

    # Detector tilt (sinogram shear along view axis)
    if abs(detector_tilt_deg) > 1e-6:
        tan_phi = np.tan(np.deg2rad(detector_tilt_deg))
        n_ang, n_det = p_bh.shape
        d_idx   = np.arange(n_det) - n_det / 2.0
        ang_idx = np.arange(n_ang, dtype=np.float64)
        ANG, DET = np.meshgrid(ang_idx, d_idx, indexing="ij")
        DET_GRID = np.meshgrid(ang_idx, np.arange(n_det), indexing="ij")[1]
        coords = np.array([ANG + DET * tan_phi * 0.15, DET_GRID])
        p_bh = map_coordinates(
            p_bh.astype(np.float64), coords.reshape(2, -1),
            order=1, mode="nearest",
        ).reshape(n_ang, n_det).astype(np.float32)

    # Poisson-equivalent Gaussian noise: sigma = max(sino_ideal) / sqrt(I0)
    # Uses the ideal (no-mismatch) sinogram max so noise level is consistent.
    max_val = float(np.max(np.abs(p_bh)))
    if max_val < 1e-8:
        max_val = 1.0
    sigma_noise = max_val / np.sqrt(I0)
    noise = rng.normal(0.0, sigma_noise, p_bh.shape).astype(np.float32)

    return (p_bh + noise).astype(np.float32)


# ── LoDoPaB-CT slice index tables ─────────────────────────────────────────────

_LODOPAB_SHARD_SIZE = 128   # images per HDF5 shard

# Public:  11 diverse slices from test set (deep lung, heart, liver, various sizes)
LODOPAB_PUBLIC_INDICES = [0, 320, 650, 980, 1310, 1640, 1970, 2300, 2630, 2960, 3290]
LODOPAB_SCENE_NAMES    = [f"lidc_test_{i:02d}" for i in range(11)]

# Dev: 20 slices hand-selected for NARROW body cross-section (apex / lower-thorax anatomy).
LODOPAB_VAL_DEV_INDICES = [
    20, 50, 172, 328, 441, 459, 604, 657, 799, 819,
    904, 943, 977, 1093, 1126, 1153, 1419, 1585, 1760, 1787,
]
LODOPAB_DEV_SCENE_NAMES = [f"lidc_val_{i:02d}" for i in range(20)]

# Hidden: 20 slices hand-selected for WIDE body cross-section (cardiac/main-thorax anatomy).
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
            print(f"    shard {shard_i:03d} -> {[r[1] for r in requests]}")

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
      - Rotation  — any orientation is a valid axial cross-section
      - Flip      — left-right / up-down symmetry
      - Zoom      — simulates different scanner FOV or patient size
    """
    from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom

    # 1. Rotation — continuous, avoids near-identity angles
    angle = float(rng.uniform(20.0, 340.0))
    x = nd_rotate(x, angle, reshape=False, mode="constant", cval=0.0)

    # 2. Flip (both axes; high probability for both modes)
    lr_prob = 0.90 if mode == "dev" else 0.95
    ud_prob = 0.75 if mode == "dev" else 0.85
    if rng.random() < lr_prob:
        x = np.fliplr(x)
    if rng.random() < ud_prob:
        x = np.flipud(x)

    # 3. Zoom — aggressive range; changes apparent anatomy scale
    lo, hi = (0.55, 1.45) if mode == "dev" else (0.50, 1.50)
    zoom_f = float(rng.uniform(lo, hi))
    x_z = nd_zoom(x, zoom_f, order=1)
    H = W = IMAGE_SIZE
    if zoom_f >= 1.0:
        zh, zw = x_z.shape
        y0 = (zh - H) // 2
        x0 = (zw - W) // 2
        x = np.ascontiguousarray(x_z[y0: y0 + H, x0: x0 + W])
    else:
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
    # Scale to LoDoPaB-CT range (0-0.55 for dev-level density)
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
            f"PWM CT benchmark -- {tier} tier "
            f"(parallel-beam sparse-view, LoDoPaB-CT geometry)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"]    = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_mm": 260.0 / IMAGE_SIZE,
            "I0": I0,
            "projection": "parallel-beam (skimage.transform.radon, circle=False)",
            "lodopab_normalisation": "x_true = (HU + 1000) / 4071",
        })
        f.attrs["source"] = source_label

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            n_views = int(rng.integers(n_views_range[0], n_views_range[1] + 1))

            # Angles in DEGREES for skimage.radon; stored in RADIANS for backward compat
            # (common_reconstructor.py applies np.degrees() when it sees values < 2*pi)
            angles_deg      = np.linspace(0, 180, n_views, endpoint=False).astype(np.float32)
            angles_nominal  = np.deg2rad(angles_deg).astype(np.float32)  # stored as radians

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_views": n_views}

            # Ideal sinogram (no mismatch, no noise) — shape (n_views, n_det)
            sino_ideal = parallel_beam_project(x_true, angles_deg)

            # Measured sinogram (mismatch + noise)
            sino_meas = apply_mismatch(
                x_true, angles_deg,
                center_offset=mis["center_offset_px"],
                angle_error_deg=mis["angle_error_deg"],
                beam_hardening_beta=mis["beam_hardening_beta"],
                detector_tilt_deg=mis["detector_tilt_deg"],
                rng=rng,
            )

            n_det = sino_ideal.shape[1]

            grp = f.create_group(key)
            grp.create_dataset("x_true",             data=x_true,         compression="gzip")
            grp.create_dataset("sinogram_ideal",      data=sino_ideal,     compression="gzip")
            grp.create_dataset("sinogram_measured",   data=sino_meas,      compression="gzip")
            grp.create_dataset("angles_nominal",      data=angles_nominal)  # radians
            grp.attrs["metadata"]    = json.dumps({
                "scene": scene_name, "shape": list(x_true.shape),
                "n_views": n_views, "n_det": n_det,
                "source": source_label,
                "projection": "parallel-beam",
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
                           title=f"{key} -- {scene_name}")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "n_views": n_views}, sf, indent=2)

            rows.append((key, scene_name, x_true.shape, n_views, mis))
            print(f"  [{tier}] {key} {scene_name}  views={n_views}  n_det={n_det}  "
                  f"Delta_c={mis['center_offset_px']:.2f} Delta_theta={mis['angle_error_deg']:.2f}deg "
                  f"beta={mis['beam_hardening_beta']:.3f} phi={mis['detector_tilt_deg']:.2f}deg")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows, source_label)
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# ── README writers ────────────────────────────────────────────────────────────

def _write_tier_readme(tier: str, tier_dir: Path, rows: list,
                       source_label: str = "synthetic") -> None:
    is_real = "leuschner" in source_label.lower() or "lodopab" in source_label.lower()
    if tier == "public":
        if is_real:
            source = ("LoDoPaB-CT real chest CT -- **test split** (LIDC/IDRI)\n"
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
            source = ("LoDoPaB-CT real chest CT -- **validation split, first half** "
                      "(patients 0-63, LIDC/IDRI)\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.\n"
                      "20 slices at indices " + str(LODOPAB_VAL_DEV_INDICES) + "\n"
                      "Completely different patients from public tier (test split).")
        else:
            source = ("Procedural chest/abdomen phantoms (FALLBACK -- "
                      "ground_truth_validation.zip not found)\n"
                      "Anatomy and HU scale match LoDoPaB-CT normalisation.")
        access = "Blind (measured sinogram + spec ranges only)"
    else:
        if is_real:
            source = ("LoDoPaB-CT real chest CT -- **validation split, second half** "
                      "(patients 64-127, LIDC/IDRI) + adversarial modifications\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.\n"
                      "20 slices at indices " + str(LODOPAB_VAL_HIDDEN_INDICES) + "\n"
                      "Adversarial: metal inserts, low-contrast lesions, "
                      "calcifications, high-contrast bone.")
        else:
            source = ("Adversarial procedural phantoms (FALLBACK -- "
                      "ground_truth_validation.zip not found)\n"
                      "Metal, calcification, low-contrast lesions on synthetic backgrounds.")
        access = "Server-only"

    spec = SPEC[tier]
    param_desc = {
        "center_offset_px":    "Delta_c -- centre-of-rotation offset",
        "angle_error_deg":     "Delta_theta -- systematic angle error",
        "beam_hardening_beta": "beta  -- beam hardening coefficient",
        "detector_tilt_deg":   "phi  -- detector tilt",
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
        "| Sample | Scene | Views | Delta_c (px) | Delta_theta (deg) | beta | phi (deg) |\n",
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
        "| `sinogram_ideal` | (n_views, n_det) | float32 | Ideal parallel-beam sinogram |\n",
        "| `sinogram_measured` | (n_views, n_det) | float32 | Measured sinogram (mismatch + noise) |\n",
        "| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [radians] |\n",
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
        "| Dev | LoDoPaB-CT **validation** split -- first half | Val patients 0-63 | 20 slices |\n"
        "| Hidden | LoDoPaB-CT **validation** split -- second half + adversarial | Val patients 64-127 | 20 slices |\n\n"
        "Each tier uses entirely different patients -- no shared scenes across tiers."
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

    txt = f"""# CT -- 2-D Parallel-Beam Sparse-View / Low-Dose

## Public Data Source

{pub_note}

## Spec DAG

```
R(theta) -->> Radon(parallel-beam) -->> D(noise, mismatch)
```

## Forward Model

Parallel-beam geometry using skimage.transform.radon (circle=False):

| Parameter | Value | Physical |
|-----------|-------|----------|
| IMAGE_SIZE | 362 x 362 px | FOV ~26 cm x 26 cm |
| pixel_size | -- | 0.718 mm/px |
| n_views (public/dev) | 60 | sparse |
| n_views (hidden) | 40-90 | per-sample random |
| angles | 0 to 180 deg (endpoint=False) | parallel-beam |

Noise: Poisson-equivalent Gaussian with sigma = max(sinogram_ideal) / sqrt(I0),
I0 = 100,000 photons.

## LoDoPaB-CT Normalisation

```
x_true in [0, 1]    x = (HU + 1000) / 4071
  0.00 -> air (-1000 HU)
  0.25 -> soft tissue / water (0 HU)
  0.42 -> cortical bone (700 HU)
  1.00 -> maximum density (3071 HU)
```

## Mismatch ThetaSpace

| Knob | Symbol | Description |
|------|--------|-------------|
| `center_offset_px` | Delta_c | Lateral shift of centre-of-rotation |
| `angle_error_deg` | Delta_theta | Systematic angular calibration error |
| `beam_hardening_beta` | beta | Polychromatic BH: p_eff = p + beta*p^2 |
| `detector_tilt_deg` | phi | Rigid tilt of detector plane |

## Scoring

```
Score = 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x Consistency
```

## Dataset Structure

```
ct/
|- lodopab_src/
|   |- ground_truth_test.zip        (~1.5 GB) -- public tier source
|   +- ground_truth_validation.zip  (~1.5 GB) -- dev + hidden tier source
|- simulate_scenes.py         Procedural phantom generator (fallback only)
|- generate_dataset.py        Builds all H5 files + PNG images
|- public/    11 real LoDoPaB-CT test slices -- GT + ideal sino + true spec (visible)
|- dev/       20 real LoDoPaB-CT validation slices (patients 0-63) -- blind eval
+- hidden/    20 real LoDoPaB-CT validation slices (patients 64-127) + adversarial mods
```

## Reading the HDF5

```python
import h5py, json, numpy as np
from skimage.transform import iradon

with h5py.File("ct_challenge_public.h5", "r") as f:
    grp = f["sample_00"]
    x_true      = grp["x_true"][:]            # (362, 362) float32  -- GT attenuation map
    sino_ideal  = grp["sinogram_ideal"][:]     # (60, n_det) float32 -- parallel-beam sino
    sino_meas   = grp["sinogram_measured"][:]  # (60, n_det) float32 -- with mismatch+noise
    angles_rad  = grp["angles_nominal"][:]     # (60,) float32       -- radians
    spec        = json.loads(grp.attrs["spec_ranges"])
    true_spec   = json.loads(grp.attrs["true_spec"])

# FBP reconstruction using iradon (parallel-beam)
angles_deg = np.degrees(angles_rad)
recon = iradon(sino_meas.T, theta=angles_deg, circle=False,
               output_size=362, filter_name="ramp")
recon = np.clip(recon, 0, None)
```

## References

- Leuschner et al. (2021) LoDoPaB-CT. *Scientific Data* 8:109.
  doi:10.1038/s41597-021-00893-z
- Kak & Slaney (2001). Principles of Computerized Tomographic Imaging.
- PWM Benchmark: https://pwm.platformai.org/benchmark/ct
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("CT Benchmark Dataset Generator (parallel-beam, LoDoPaB-CT geometry)")
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
                  f"orig mean={x.mean():.4f} -> aug mean={x_aug.mean():.4f}")
            dev_phantoms.append((scene_name, x_aug))
        dev_source = (
            _LODOPAB_SOURCE +
            " Validation split, patients 0-63."
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
          " adversarial, 40-90 views)...")
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
                  f"orig mean={x.mean():.4f} -> aug={x_aug.mean():.4f} -> adv={x_adv.mean():.4f}")
            hidden_phantoms.append((f"{scene_name}_adversarial", x_adv))
        hidden_source = (
            _LODOPAB_SOURCE +
            " Validation split, patients 64-127."
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
    print(f"Done -- CT benchmark ready at {BENCHMARK_DIR}")
    if not pub_is_real:
        print("NOTE: Public tier used synthetic fallback.")
        print("      Set LODOPAB_ROOT or place ground_truth_test.zip in lodopab_src/")
    print("=" * 68)


if __name__ == "__main__":
    main()
