#!/usr/bin/env python3
"""Generate the fan-beam sparse-view / low-dose CT benchmark dataset.

Public tier  — 11 real chest CT cross-sections from LoDoPaB-CT (LIDC/IDRI)
Dev tier     — 20 procedural chest phantoms, anatomy matches LoDoPaB-CT
Hidden tier  — 20 adversarial procedural phantoms

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

import json
import shutil
import sys
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates, zoom

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "pwm_core"))

from simulate_scenes import generate_ct_gt  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent
LODOPAB_SRC   = BENCHMARK_DIR / "lodopab_src"

# ── Geometry (pixel units) ────────────────────────────────────────────────────

IMAGE_SIZE  = 362       # LoDoPaB-CT image domain (26 cm / 0.718 mm per px)
D_SO        = 800.0     # source-to-isocenter   (≈ 575 mm)
D_SD        = 568.0     # isocenter-to-detector (≈ 408 mm)
N_DET       = 736       # detector channels (matches PWM page spec)
DET_SPACING = 1.496     # detector pitch (px) ≈ 1.07 mm
N_VIEWS     = 60        # sparse views (public & dev)
I0          = 10_000.0  # nominal photon count (quarter dose, 120 kVp)
SIGMA_RO    = 5.0       # readout noise σ

# Physical attenuation scale: pixel-density units → nepers.
# pixel_size_mm = 260 mm / 362 px = 0.718 mm/px
# μ_max = 0.081 mm^-1  (3071 HU at 120 kVp)
# MU_SCALE = pixel_size_mm × μ_max ≈ 0.718 × 0.081 ≈ 0.058
MU_SCALE    = 0.058

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

    Args:
        x:             Attenuation map (H, W), values ≥ 0.
        angles_rad:    Projection angles in radians, shape (n_views,).
        center_offset: Δc — lateral shift of the centre-of-rotation (pixels).

    Returns:
        sinogram (n_views, n_det), float32.  Units: x_true × pixels.
        Multiply by MU_SCALE to convert to physical attenuation (nepers).
    """
    H, W = x.shape
    x64 = x.astype(np.float64)

    cy = H / 2.0
    cx = W / 2.0 + center_offset

    det_pos = (np.arange(n_det) - n_det / 2.0) * det_spacing   # (n_det,)

    diag = np.sqrt(H ** 2 + W ** 2)
    n_steps = max(int(diag * 1.5), 512)
    t_vals = np.linspace(0.0, 1.0, n_steps)                     # (n_steps,)

    sinogram = np.zeros((len(angles_rad), n_det), dtype=np.float32)

    for i, angle in enumerate(angles_rad):
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        src_y = -D_so * sin_a + cy
        src_x =  D_so * cos_a + cx

        det_y = D_sd * sin_a + det_pos * cos_a + cy   # (n_det,)
        det_x = -D_sd * cos_a + det_pos * sin_a + cx  # (n_det,)

        ray_y = det_y - src_y   # (n_det,)
        ray_x = det_x - src_x   # (n_det,)
        ray_len = np.sqrt(ray_y ** 2 + ray_x ** 2)  # (n_det,)

        sample_y = src_y + np.outer(ray_y, t_vals)
        sample_x = src_x + np.outer(ray_x, t_vals)

        coords = np.array([sample_y.ravel(), sample_x.ravel()])
        vals = map_coordinates(x64, coords, order=1, mode="constant", cval=0.0)
        vals = vals.reshape(n_det, n_steps)

        step_size = ray_len / n_steps                # (n_det,)
        sinogram[i] = (vals.sum(axis=1) * step_size).astype(np.float32)

    return sinogram


def apply_mismatch(
    sino_true: np.ndarray,
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
    """Apply all four mismatch effects + noise to produce the measured sinogram.

    Pipeline: re-project with (Δθ, Δc) → scale to physical nepers (MU_SCALE) →
              beam hardening (β) → detector tilt (φ) → Poisson + readout noise.

    Returns p_measured (n_views, n_det), float32, in physical attenuation units.
    """
    # ── 1. Re-project with angular error + centre-of-rotation offset ──────────
    angles_true = angles_nominal + np.deg2rad(angle_error_deg)
    p_geo = fan_beam_project(x, angles_true, center_offset=center_offset)

    # ── Scale pixel-density units → physical attenuation (nepers) ─────────────
    p_phys = p_geo * MU_SCALE

    # ── 2. Beam hardening: p_eff = p + β·p² (in physical units) ──────────────
    p_bh = p_phys + beam_hardening_beta * p_phys ** 2

    # ── 3. Detector tilt: row-dependent sinogram shear ─────────────────────────
    if abs(detector_tilt_deg) > 1e-6:
        tan_phi = np.tan(np.deg2rad(detector_tilt_deg))
        n_ang, n_det = p_bh.shape
        d_idx = np.arange(n_det) - n_det / 2.0
        ang_idx = np.arange(n_ang, dtype=np.float64)
        ANG, DET = np.meshgrid(ang_idx, d_idx, indexing="ij")
        coords = np.array([ANG + DET * tan_phi * 0.15,
                           np.meshgrid(ang_idx, np.arange(n_det), indexing="ij")[1]])
        p_tilt = map_coordinates(p_bh.astype(np.float64), coords.reshape(2, -1),
                                 order=1, mode="nearest").reshape(n_ang, n_det).astype(np.float32)
    else:
        p_tilt = p_bh

    # ── 4. Low-dose noise: Beer-Lambert + Poisson + readout ───────────────────
    p_clamped = np.clip(p_tilt, 0.0, 20.0)
    I_expect = I0 * np.exp(-p_clamped)
    I_noisy = rng.poisson(np.maximum(I_expect, 1e-3)).astype(np.float64)
    I_noisy += rng.normal(0.0, sigma_ro, I_noisy.shape)
    I_noisy = np.maximum(I_noisy, 1.0)
    p_measured = -np.log(I_noisy / I0).astype(np.float32)

    return p_measured


# ── LoDoPaB-CT public-tier loader ─────────────────────────────────────────────

# Indices of 11 diverse chest CT slices selected from the LoDoPaB-CT test set.
# Chosen to cover: deep lung, heart level, liver level, and various body sizes.
LODOPAB_PUBLIC_INDICES = [0, 320, 650, 980, 1310, 1640, 1970, 2300, 2630, 2960, 3290]

# Human-readable scene names for the 11 selected slices
LODOPAB_SCENE_NAMES = [
    "lidc_chest_00", "lidc_chest_01", "lidc_chest_02", "lidc_chest_03",
    "lidc_chest_04", "lidc_chest_05", "lidc_chest_06", "lidc_chest_07",
    "lidc_chest_08", "lidc_chest_09", "lidc_chest_10",
]


_LODOPAB_SHARD_SIZE = 128   # images per HDF5 shard in LoDoPaB-CT test zip


def load_lodopab_public() -> list[tuple[str, np.ndarray]]:
    """Extract 11 diverse ground-truth images from LoDoPaB-CT test set.

    Source: Zenodo record 3384092, lodopab_src/ground_truth_test.zip
    Shards: ground_truth_test_NNN.hdf5, each with dataset 'data' of
            shape (128, 362, 362) float32 in [0, 1].

    Returns:
        List of (scene_name, x_true) where x_true is float32 in [0, 1].
    """
    import io as _io

    zip_path = LODOPAB_SRC / "ground_truth_test.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"LoDoPaB-CT ground truth not found at {zip_path}\n"
            "Download with:\n"
            "  mkdir -p lodopab_src && wget \\\n"
            "    'https://zenodo.org/api/records/3384092/files/"
            "ground_truth_test.zip/content' \\\n"
            "    -O lodopab_src/ground_truth_test.zip"
        )

    print(f"  Reading LoDoPaB-CT ground truth from {zip_path.name} ...")

    # Map each target global index → (shard_idx, local_idx)
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
        missing = sorted(set(LODOPAB_PUBLIC_INDICES) - set(found.keys()))
        raise RuntimeError(f"Could not find LoDoPaB-CT indices: {missing}")

    samples = [(name, found[idx])
               for idx, name in zip(LODOPAB_PUBLIC_INDICES, LODOPAB_SCENE_NAMES)]
    print(f"  Loaded {len(samples)} LoDoPaB-CT public images.")
    return samples


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _save_img(arr: np.ndarray, path: Path, cmap: str = "gray") -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap=cmap, aspect="auto",
              vmin=arr.min(), vmax=arr.max())
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_overview(
    x_true: np.ndarray,
    sino_ideal: np.ndarray,
    sino_mismatch: np.ndarray,
    path: Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_true, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth μ", fontsize=9)
    axes[1].imshow(sino_ideal, cmap="gray", aspect="auto")
    axes[1].set_title(f"Ideal Sinogram ({sino_ideal.shape[0]} views)", fontsize=9)
    axes[2].imshow(sino_mismatch, cmap="gray", aspect="auto")
    axes[2].set_title("Measured (mismatch + noise)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Dataset tier generator ────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {
        k: float(rng.uniform(v["min"], v["max"]))
        for k, v in spec.items()
    }


def generate_tier(
    tier: str,
    phantoms: list[tuple[str, np.ndarray]],
    base_seed: int,
    n_views_range: tuple[int, int],
) -> None:
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
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
        f.attrs["geometry"] = json.dumps({
            "D_so_px": D_SO, "D_sd_px": D_SD,
            "n_det": N_DET, "det_spacing_px": DET_SPACING,
            "pixel_size_mm": 260.0 / IMAGE_SIZE,
            "I0": I0, "sigma_readout": SIGMA_RO,
            "mu_scale": MU_SCALE,
            "mu_scale_note": "sinogram_ideal × mu_scale = physical attenuation (nepers)",
            "lodopab_normalisation": "x_true = (HU + 1000) / 4071",
        })
        if tier == "public":
            f.attrs["source"] = (
                "LoDoPaB-CT ground-truth chest CT images from LIDC/IDRI "
                "(Leuschner et al. 2021, Sci Data, doi:10.1038/s41597-021-00893-z). "
                "Zenodo record 3384092, CC BY 4.0."
            )

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"

            n_views = int(rng.integers(n_views_range[0], n_views_range[1] + 1))
            angles_nominal = np.linspace(0, np.pi, n_views, endpoint=False)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_views": n_views}

            # Ideal sinogram in physical units (nepers)
            sino_ideal = fan_beam_project(x_true, angles_nominal) * MU_SCALE

            # Measured sinogram (mismatch + noise), also in nepers
            sino_meas = apply_mismatch(
                sino_ideal, x_true, angles_nominal,
                center_offset=mis["center_offset_px"],
                angle_error_deg=mis["angle_error_deg"],
                beam_hardening_beta=mis["beam_hardening_beta"],
                detector_tilt_deg=mis["detector_tilt_deg"],
                rng=rng,
            )

            grp = f.create_group(key)
            grp.create_dataset("x_true",             data=x_true,     compression="gzip")
            grp.create_dataset("sinogram_ideal",      data=sino_ideal,  compression="gzip")
            grp.create_dataset("sinogram_measured",   data=sino_meas,   compression="gzip")
            grp.create_dataset("angles_nominal",      data=angles_nominal.astype(np.float32))
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name, "shape": list(x_true.shape),
                "n_views": n_views, "n_det": N_DET,
                "D_so_px": D_SO, "D_sd_px": D_SD,
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"]   = json.dumps({**mis, "n_views": n_views})

            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_img(x_true, sample_dir / "ground_truth.png")
            _save_img(sino_ideal,  sample_dir / "sinogram_ideal.png")
            _save_img(sino_meas,   sample_dir / "sinogram_measured.png")
            _save_overview(x_true, sino_ideal, sino_meas,
                           sample_dir / "overview.png",
                           title=f"{key} — {scene_name}")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "n_views": n_views}, sf, indent=2)

            rows.append((key, scene_name, x_true.shape, n_views, mis))
            print(f"  [{tier}] {key} {scene_name}  views={n_views}  done")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows)
    print(f"  [{tier}] HDF5 → {h5_path}")


# ── README writers ────────────────────────────────────────────────────────────

def _write_tier_readme(tier: str, tier_dir: Path, rows: list) -> None:
    desc = {
        "public": ("Public",  "LoDoPaB-CT real chest CT (LIDC/IDRI)",
                   "Full (GT + true spec + ideal sinogram)"),
        "dev":    ("Dev",     "Procedural chest phantoms (anatomy matches LoDoPaB-CT)",
                   "Blind (measured sinogram + spec ranges)"),
        "hidden": ("Hidden",  "Adversarial chest phantoms",
                   "Server-only"),
    }
    label, source, access = desc[tier]
    lines = [
        f"# CT {label} Tier\n\n",
        f"Source: **{source}** | Access: **{access}**\n\n",
        "## Mismatch Parameters\n\n",
        "| Parameter | Description | Range |\n",
        "|-----------|-------------|-------|\n",
    ]
    spec = SPEC[tier]
    param_desc = {
        "center_offset_px":    "Δc — centre-of-rotation offset",
        "angle_error_deg":     "Δθ — systematic angle error",
        "beam_hardening_beta": "β  — beam hardening coefficient",
        "detector_tilt_deg":   "φ  — detector tilt",
    }
    for k, v in spec.items():
        lo, hi, u = v["min"], v["max"], v.get("unit", "")
        lines.append(f"| `{k}` | {param_desc[k]} | [{lo}, {hi}] {u} |\n")

    lines += [
        "\n## Samples\n\n",
        "| Sample | Scene | Shape | Views | Δc | Δθ | β | φ |\n",
        "|--------|-------|-------|-------|----|----|---|---|\n",
    ]
    for key, scene, shape, n_views, mis in rows:
        lines.append(
            f"| {key} | {scene} | {shape[0]}×{shape[1]} | {n_views}"
            f" | {mis['center_offset_px']:.3f}"
            f" | {mis['angle_error_deg']:.3f}"
            f" | {mis['beam_hardening_beta']:.3f}"
            f" | {mis['detector_tilt_deg']:.3f} |\n"
        )

    lines += [
        "\n## HDF5 Datasets per Sample\n\n",
        "| Key | Shape | Description |\n",
        "|-----|-------|-------------|\n",
        "| `x_true` | (362, 362) | Ground-truth attenuation map, x=(HU+1000)/4071 |\n",
        "| `sinogram_ideal` | (n_views, 736) | Ideal fan-beam sinogram (nepers, no mismatch/noise) |\n",
        "| `sinogram_measured` | (n_views, 736) | Measured sinogram (mismatch + Poisson + readout noise, nepers) |\n",
        "| `angles_nominal` | (n_views,) | Nominal projection angles [rad] |\n",
    ]
    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme() -> None:
    txt = """\
# CT — 2-D Fan-Beam Sparse-View / Low-Dose

## Public Data Source

Real chest CT cross-sections from **LoDoPaB-CT** (Leuschner et al. 2021,
*Scientific Data* doi:10.1038/s41597-021-00893-z), sourced from the
LIDC/IDRI lung CT database.  11 diverse slices from the test set are used.

Zenodo record 3384092, CC BY 4.0.

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
├── lodopab_src/               LoDoPaB-CT source zip (gitignored)
├── public/    11 real LoDoPaB-CT slices — full GT + ideal sinogram + true spec
├── dev/       20 procedural chest phantoms — anatomy matches LoDoPaB-CT
└── hidden/    20 adversarial — metal, low-contrast, calcifications (hard mismatch)
```

## Reading the HDF5

```python
import h5py, json, numpy as np

with h5py.File("ct_challenge_public.h5", "r") as f:
    grp = f["sample_00"]
    x_true     = grp["x_true"][:]             # (362, 362) float32
    sino_ideal = grp["sinogram_ideal"][:]      # (60, 736)  float32  [nepers]
    sino_meas  = grp["sinogram_measured"][:]   # (60, 736)  float32  [nepers]
    angles     = grp["angles_nominal"][:]      # (60,)      float32  [rad]
    spec       = json.loads(grp.attrs["spec_ranges"])
    true_spec  = json.loads(grp.attrs["true_spec"])
```

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
    print("================================================================")
    print(f"Output: {BENCHMARK_DIR}\n")

    shape = (IMAGE_SIZE, IMAGE_SIZE)

    # Public — 11 real LoDoPaB-CT slices, 60 views, mild mismatch
    print("Generating public tier (11 real LoDoPaB-CT slices, 60 views)...")
    public_phantoms = load_lodopab_public()
    generate_tier("public", public_phantoms, base_seed=1000,
                  n_views_range=(60, 60))

    # Dev — 20 procedural chest phantoms, 60 views, moderate mismatch
    print("\nGenerating dev tier (20 procedural chest, 60 views)...")
    dev_phantoms = []
    for i in range(20):
        x, recipe = generate_ct_gt(seed=7000 + i, mode="dev", shape=shape)
        dev_phantoms.append((f"proc_dev_{i:02d}", x))
    generate_tier("dev", dev_phantoms, base_seed=7000,
                  n_views_range=(60, 60))

    # Hidden — 20 adversarial phantoms, 40–90 views, hard mismatch
    print("\nGenerating hidden tier (20 adversarial, 40–90 views)...")
    hidden_phantoms = []
    for i in range(20):
        x, recipe = generate_ct_gt(seed=9000 + i, mode="hidden", shape=shape)
        hidden_phantoms.append((f"proc_hidden_{i:02d}", x))
    generate_tier("hidden", hidden_phantoms, base_seed=9000,
                  n_views_range=(40, 90))

    _write_top_readme()

    print(f"\nDone — CT benchmark ready at {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()
