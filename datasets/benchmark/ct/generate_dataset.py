#!/usr/bin/env python3
"""Generate the fan-beam sparse-view / low-dose CT benchmark dataset.

Public tier  — 11 real chest CT cross-sections from LoDoPaB-CT (LIDC/IDRI)
Dev tier     — 20 procedural chest/abdomen phantoms (anatomy matches LoDoPaB-CT)
Hidden tier  — 20 adversarial procedural phantoms

Public data source — LoDoPaB-CT (most widely used CT reconstruction benchmark)
-------------------------------------------------------------------------------
Leuschner et al. (2021), Scientific Data 8:109, doi:10.1038/s41597-021-00893-z
Sourced from LIDC/IDRI lung CT database.  Zenodo record 3384092, CC BY 4.0.

To use real LoDoPaB-CT data for the public tier, either:
  (a) Set LODOPAB_ROOT to a directory containing ground_truth_test.zip, OR
  (b) Place the zip at   ct/lodopab_src/ground_truth_test.zip

Download command:
  mkdir -p lodopab_src
  wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \\
       -O lodopab_src/ground_truth_test.zip

Without the zip the public tier falls back to synthetic Shepp-Logan / chest
phantoms (clearly flagged in metadata as "source": "synthetic").

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
from simulate_scenes import generate_ct_gt  # noqa: E402

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


# ── LoDoPaB-CT public-tier loader ─────────────────────────────────────────────

# 11 diverse slices from the LoDoPaB-CT test set:
# deep lung, heart level, liver level, various body sizes
LODOPAB_PUBLIC_INDICES = [0, 320, 650, 980, 1310, 1640, 1970, 2300, 2630, 2960, 3290]
LODOPAB_SCENE_NAMES   = [f"lidc_chest_{i:02d}" for i in range(11)]
_LODOPAB_SHARD_SIZE   = 128   # images per HDF5 shard in the test zip


def _find_lodopab_zip() -> Path | None:
    """Locate ground_truth_test.zip from env var or default path."""
    root = os.environ.get("LODOPAB_ROOT", "")
    candidates = []
    if root:
        candidates.append(Path(root) / "ground_truth_test.zip")
        candidates.append(Path(root))
    candidates.append(BENCHMARK_DIR / "lodopab_src" / "ground_truth_test.zip")
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_lodopab_public() -> list[tuple[str, np.ndarray]] | None:
    """Extract 11 diverse ground-truth images from the LoDoPaB-CT test zip.

    Returns list of (scene_name, x_true float32 [0,1]) on success.
    Returns None if the zip is not found, with clear download instructions.
    """
    zip_path = _find_lodopab_zip()

    if zip_path is None:
        print("  [WARNING] LoDoPaB-CT ground_truth_test.zip not found.")
        print("  [WARNING] Public tier will use SYNTHETIC PLACEHOLDER images.")
        print("  [WARNING] To use real LoDoPaB-CT data:")
        print("  [WARNING]   export LODOPAB_ROOT=/path/containing/ground_truth_test.zip")
        print("  [WARNING]   — OR —")
        print("  [WARNING]   Place zip at: ct/lodopab_src/ground_truth_test.zip")
        print("  [WARNING] Download:")
        print("  [WARNING]   mkdir -p lodopab_src && wget \\")
        print("  [WARNING]     'https://zenodo.org/api/records/3384092/files/"
              "ground_truth_test.zip/content' \\")
        print("  [WARNING]     -O lodopab_src/ground_truth_test.zip")
        return None

    print(f"  Reading LoDoPaB-CT from {zip_path} ...")

    shard_map: dict[int, list[tuple[int, int]]] = {}
    for global_i in LODOPAB_PUBLIC_INDICES:
        shard_i = global_i // _LODOPAB_SHARD_SIZE
        local_i = global_i % _LODOPAB_SHARD_SIZE
        shard_map.setdefault(shard_i, []).append((local_i, global_i))

    found: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for shard_i, requests in sorted(shard_map.items()):
            shard_name = f"ground_truth_test_{shard_i:03d}.hdf5"
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

    if len(found) < len(LODOPAB_PUBLIC_INDICES):
        print(f"  [WARNING] Only {len(found)}/{len(LODOPAB_PUBLIC_INDICES)} slices found.")

    result = [(name, found[idx])
              for idx, name in zip(LODOPAB_PUBLIC_INDICES, LODOPAB_SCENE_NAMES)
              if idx in found]
    print(f"  Loaded {len(result)} LoDoPaB-CT images.")
    return result if result else None


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
        if tier == "public":
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
                "source": source_label if tier == "public" else "synthetic",
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
    if tier == "public":
        if "lodopab" in source_label.lower() or "lidc" in source_label.lower():
            source = ("LoDoPaB-CT real chest CT (LIDC/IDRI)\n"
                      "Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z\n"
                      "Zenodo record 3384092, CC BY 4.0.")
        else:
            source = ("Synthetic Shepp-Logan variants (PLACEHOLDER)\n"
                      "Set LODOPAB_ROOT or place ground_truth_test.zip in lodopab_src/ "
                      "for real LoDoPaB-CT data.")
        access = "Full (GT + true spec + ideal sinogram)"
    elif tier == "dev":
        source = "Procedural chest/abdomen phantoms — anatomy and HU scale match LoDoPaB-CT"
        access = "Blind (measured sinogram + spec ranges)"
    else:
        source = "Adversarial chest/abdomen phantoms (metal, calcification, low-contrast lesions)"
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
        "Real chest CT cross-sections from **LoDoPaB-CT** (Leuschner et al. 2021,\n"
        "*Scientific Data* doi:10.1038/s41597-021-00893-z), sourced from the\n"
        "LIDC/IDRI lung CT database.  11 diverse slices from the test set are used.\n\n"
        "Zenodo record 3384092, CC BY 4.0."
    ) if pub_is_real else (
        "**PLACEHOLDER:** Synthetic Shepp-Logan phantoms (LoDoPaB-CT zip not found).\n\n"
        "To use real data:\n"
        "```bash\n"
        "export LODOPAB_ROOT=/path/to/dir/with/ground_truth_test.zip\n"
        "python3 generate_dataset.py\n"
        "```\n"
        "Download: `wget 'https://zenodo.org/api/records/3384092/files/"
        "ground_truth_test.zip/content' -O lodopab_src/ground_truth_test.zip`"
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
├── lodopab_src/               LoDoPaB-CT source zip (gitignored; set LODOPAB_ROOT)
├── simulate_scenes.py         Procedural chest/abdomen CT phantom generator
├── generate_dataset.py        Builds all H5 files + PNG images
├── public/    11 LoDoPaB-CT slices (or synthetic fallback)  — GT + ideal sino + true spec
├── dev/       20 procedural chest phantoms (anatomy matches LoDoPaB-CT)
└── hidden/    20 adversarial — metal, low-contrast, calcifications (hard mismatch)
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

    # ── Dev tier ─────────────────────────────────────────────────────────────
    print("\nGenerating dev tier (20 procedural chest/abdomen, 60 views)...")
    dev_phantoms = []
    for i in range(20):
        x, scene_type = generate_ct_gt(seed=7000 + i, mode="dev", shape=shape)
        dev_phantoms.append((scene_type, x))
    generate_tier("dev", dev_phantoms, base_seed=7000,
                  n_views_range=(60, 60), source_label="synthetic")

    # ── Hidden tier ──────────────────────────────────────────────────────────
    print("\nGenerating hidden tier (20 adversarial, 40–90 views)...")
    hidden_phantoms = []
    for i in range(20):
        x, scene_type = generate_ct_gt(seed=9000 + i, mode="hidden", shape=shape)
        hidden_phantoms.append((scene_type, x))
    generate_tier("hidden", hidden_phantoms, base_seed=9000,
                  n_views_range=(40, 90), source_label="synthetic")

    _write_top_readme(pub_is_real)

    print(f"\n{'=' * 68}")
    print(f"Done — CT benchmark ready at {BENCHMARK_DIR}")
    if not pub_is_real:
        print("NOTE: Public tier used synthetic fallback.")
        print("      Set LODOPAB_ROOT or place ground_truth_test.zip in lodopab_src/")
    print("=" * 68)


if __name__ == "__main__":
    main()
