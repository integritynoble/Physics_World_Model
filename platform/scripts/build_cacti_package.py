#!/usr/bin/env python3
"""
Build the CACTI dataset package with procedurally generated Dev/Hidden data.

Tier assignment:
  - Public  (20 samples): 6 CACTI scenes × all T=8 groups, 256×256
  - Dev     (20 samples): Procedural video, 512×512, T∈{8,16,32}  (mild)
  - Hidden  (20 samples): Procedural video, 512×512, T∈{8,16,32}  (hard)

The procedural generator creates ground-truth video clips deterministically
from (seed, recipe_id) — no external datasets needed. PWM keeps the
secret manifest (seeds + params) private so data is unreproducible.

Output:  datasets/benchmark/cacti/
"""

import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import shift, rotate, gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "cacti"
SIM_DIR = PROJECT_ROOT / "datasets" / "CACTI" / "simulation"

# Add package dir to path for importing the generator
sys.path.insert(0, str(PACKAGE_DIR))
from procedural_video_generator import (
    generate_video, RECIPE_NAMES, DEV_RECIPES, HIDDEN_RECIPES,
)

# ── Config ───────────────────────────────────────────────────────────────────

SPEC_RANGES = [
    {"name": "mask_dx",       "min": -0.5,  "max": 0.5,  "unit": "px"},
    {"name": "mask_dy",       "min": -0.3,  "max": 0.3,  "unit": "px"},
    {"name": "mask_rotation", "min": -0.2,  "max": 0.2,  "unit": "deg"},
    {"name": "mask_blur",     "min": 0.0,   "max": 0.3,  "unit": "px"},
    {"name": "clock_offset",  "min": -0.1,  "max": 0.1,  "unit": "frames"},
    {"name": "gain_drift",    "min": 0.95,  "max": 1.05, "unit": ""},
    {"name": "offset_drift",  "min": -0.02, "max": 0.02, "unit": ""},
]

NOISE_PARAMS = {"poisson_peak": 10000, "gaussian_sigma": 1.0}

# Mismatch severity by tier:
#   Public: very mild — expect GAP-TV ~26-28 dB (close to InverseNet paper)
#   Dev:    moderate  — expect GAP-TV ~24-25 dB
#   Hidden: harder    — expect GAP-TV ~22-24 dB
TRUE_SPEC_PUBLIC = {
    "mask_dx": 0.50, "mask_dy": 0.30, "mask_rotation": 0.10,
    "mask_blur": 0.0, "clock_offset": 0.05,
    "gain_drift": 1.02, "offset_drift": 0.002,
}
TRUE_SPEC_DEV = {
    "mask_dx": 0.20, "mask_dy": 0.10, "mask_rotation": 0.08,
    "mask_blur": 0.10, "clock_offset": -0.03,
    "gain_drift": 0.98, "offset_drift": -0.008,
}
TRUE_SPEC_HIDDEN = {
    "mask_dx": 0.35, "mask_dy": 0.20, "mask_rotation": 0.15,
    "mask_blur": 0.15, "clock_offset": 0.05,
    "gain_drift": 1.03, "offset_drift": 0.012,
}

# ── Secret manifests ─────────────────────────────────────────────────────────
# Dev (20 samples): 40% T=8, 40% T=16, 20% T=32
DEV_MANIFEST = [
    {"seed": 7001, "T": 8,  "recipe": DEV_RECIPES[0]},   # urban
    {"seed": 7002, "T": 8,  "recipe": DEV_RECIPES[1]},   # nature
    {"seed": 7003, "T": 8,  "recipe": DEV_RECIPES[2]},   # occlusion
    {"seed": 7004, "T": 8,  "recipe": DEV_RECIPES[3]},   # urban
    {"seed": 7005, "T": 8,  "recipe": DEV_RECIPES[4]},   # nature
    {"seed": 7006, "T": 8,  "recipe": DEV_RECIPES[5]},   # occlusion
    {"seed": 7007, "T": 8,  "recipe": DEV_RECIPES[6]},   # urban
    {"seed": 7008, "T": 8,  "recipe": DEV_RECIPES[7]},   # nature
    {"seed": 7009, "T": 16, "recipe": DEV_RECIPES[8]},   # cam_shake
    {"seed": 7010, "T": 16, "recipe": DEV_RECIPES[9]},   # occlusion
    {"seed": 7011, "T": 16, "recipe": DEV_RECIPES[10]},  # urban
    {"seed": 7012, "T": 16, "recipe": DEV_RECIPES[11]},  # nature
    {"seed": 7013, "T": 16, "recipe": DEV_RECIPES[12]},  # occlusion
    {"seed": 7014, "T": 16, "recipe": DEV_RECIPES[13]},  # cam_shake
    {"seed": 7015, "T": 16, "recipe": DEV_RECIPES[14]},  # urban
    {"seed": 7016, "T": 16, "recipe": DEV_RECIPES[15]},  # nature
    {"seed": 7017, "T": 32, "recipe": DEV_RECIPES[16]},  # occlusion
    {"seed": 7018, "T": 32, "recipe": DEV_RECIPES[17]},  # urban
    {"seed": 7019, "T": 32, "recipe": DEV_RECIPES[18]},  # nature
    {"seed": 7020, "T": 32, "recipe": DEV_RECIPES[19]},  # cam_shake
]

# Hidden (20 samples): 30% T=8, 30% T=16, 40% T=32
HIDDEN_MANIFEST = [
    {"seed": 9001, "T": 8,  "recipe": HIDDEN_RECIPES[0]},   # textile
    {"seed": 9002, "T": 8,  "recipe": HIDDEN_RECIPES[1]},   # particles
    {"seed": 9003, "T": 8,  "recipe": HIDDEN_RECIPES[2]},   # thin_struct
    {"seed": 9004, "T": 8,  "recipe": HIDDEN_RECIPES[3]},   # cam_shake
    {"seed": 9005, "T": 8,  "recipe": HIDDEN_RECIPES[4]},   # textile
    {"seed": 9006, "T": 8,  "recipe": HIDDEN_RECIPES[5]},   # particles
    {"seed": 9007, "T": 16, "recipe": HIDDEN_RECIPES[6]},   # thin_struct
    {"seed": 9008, "T": 16, "recipe": HIDDEN_RECIPES[7]},   # urban
    {"seed": 9009, "T": 16, "recipe": HIDDEN_RECIPES[8]},   # nature
    {"seed": 9010, "T": 16, "recipe": HIDDEN_RECIPES[9]},   # occlusion
    {"seed": 9011, "T": 16, "recipe": HIDDEN_RECIPES[10]},  # textile
    {"seed": 9012, "T": 16, "recipe": HIDDEN_RECIPES[11]},  # particles
    {"seed": 9013, "T": 32, "recipe": HIDDEN_RECIPES[12]},  # thin_struct
    {"seed": 9014, "T": 32, "recipe": HIDDEN_RECIPES[13]},  # cam_shake
    {"seed": 9015, "T": 32, "recipe": HIDDEN_RECIPES[14]},  # textile
    {"seed": 9016, "T": 32, "recipe": HIDDEN_RECIPES[15]},  # particles
    {"seed": 9017, "T": 32, "recipe": HIDDEN_RECIPES[16]},  # thin_struct
    {"seed": 9018, "T": 32, "recipe": HIDDEN_RECIPES[17]},  # urban
    {"seed": 9019, "T": 32, "recipe": HIDDEN_RECIPES[18]},  # nature
    {"seed": 9020, "T": 32, "recipe": HIDDEN_RECIPES[19]},  # occlusion
]


# ── Physics: CACTI forward model ─────────────────────────────────────────────

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def apply_cacti_mismatch(x, mask, true_spec):
    H, W, T = x.shape
    mm = mask.copy()
    dx, dy = true_spec["mask_dx"], true_spec["mask_dy"]
    rot, blur = true_spec["mask_rotation"], true_spec["mask_blur"]
    for t in range(T):
        f = mm[:, :, t]
        f = shift(f, [dy, dx], order=1, mode="constant")
        if abs(rot) > 1e-6:
            f = rotate(f, rot, reshape=False, order=1, mode="constant")
        if blur > 0:
            f = gaussian_filter(f, sigma=blur)
        mm[:, :, t] = f
    # Binarize warped mask before computing measurement (matching InverseNet paper)
    mm = (mm > 0.5).astype(np.float64)
    gain, offset = true_spec["gain_drift"], true_spec["offset_drift"]
    y = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        y += mm[:, :, t] * x[:, :, t]
    y = gain * y + offset
    return y, mask


def add_noise(y, rng):
    """Add Poisson-Gaussian noise matching InverseNet paper noise model.

    Scales measurement to peak photon count, applies Poisson shot noise,
    adds Gaussian read noise, then scales back. This gives ~40 dB measurement
    SNR (peak=10000, sigma=1.0).
    """
    peak = NOISE_PARAMS["poisson_peak"]
    sigma = NOISE_PARAMS["gaussian_sigma"]
    y = np.maximum(y, 0).astype(np.float64)
    y_max = y.max()
    if y_max < 1e-10:
        return y
    y_scaled = y / y_max * peak
    y_noisy = rng.poisson(np.maximum(y_scaled, 0).astype(np.int64)).astype(np.float64)
    y_noisy += rng.normal(0, sigma, y_noisy.shape)
    y_noisy = y_noisy / peak * y_max
    return np.maximum(y_noisy, 0)


def gen_mask(H, W, T, rng):
    return (rng.random((H, W, T)) > 0.5).astype(np.float64)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_public_samples():
    """Load all T=8 measurement groups from the 6 CACTI simulation scenes.

    Each scene may have multiple non-overlapping T=8 windows:
      kobe (4), traffic (6), runner (1), drop (1), crash (4), aerial (4) = 20 total.
    """
    files = [
        ("kobe_cacti.mat", "kobe"), ("traffic_cacti.mat", "traffic"),
        ("runner8_cacti.mat", "runner"), ("drop8_cacti.mat", "drop"),
        ("crash32_cacti.mat", "crash"), ("aerial32_cacti.mat", "aerial"),
    ]
    T_mask = 8  # frames per measurement group
    samples = []
    for fname, name in files:
        mat = sio.loadmat(str(SIM_DIR / fname))
        orig = np.array(mat["orig"], dtype=np.float64)
        mask = np.array(mat["mask"], dtype=np.float64)
        if orig.max() > 1.0:
            orig = orig / 255.0
        orig = np.clip(orig, 0, 1)
        mask = mask[:, :, :T_mask] / max(mask[:, :, :T_mask].max(), 1e-8)

        T_total = orig.shape[2]
        n_groups = T_total // T_mask
        print(f"    {name}: {T_total} frames -> {n_groups} group(s)")
        for g in range(n_groups):
            t0 = g * T_mask
            x = orig[:, :, t0 : t0 + T_mask]
            label = f"{name}_g{g}" if n_groups > 1 else name
            samples.append({
                "scene": label, "x": x, "mask": mask.copy(), "T": T_mask,
                "source": "CACTI simulation",
            })
    return samples


def load_procedural_samples(manifest, difficulty):
    samples = []
    for m in manifest:
        recipe_name = RECIPE_NAMES[m["recipe"]]
        label = f"proc_{recipe_name}_T{m['T']}_s{m['seed']}"
        print(f"      Generating {label} ...")
        x = generate_video(m["seed"], m["T"], m["recipe"],
                           size=512, difficulty=difficulty)
        samples.append({
            "scene": label, "x": x, "mask": None, "T": m["T"],
            "source": f"procedural/{recipe_name}",
        })
    return samples


# ── HDF5 generation ──────────────────────────────────────────────────────────

def generate_tier_h5(tier_name, samples, true_spec, seed,
                     include_gt, include_true_spec, out_dir):
    out_path = out_dir / f"cacti_challenge_{tier_name}.h5"
    rng = np.random.default_rng(seed)
    print(f"\n  Generating {tier_name} tier ({len(samples)} samples) -> {out_path}")

    with h5py.File(str(out_path), "w") as f:
        f.attrs["variant"] = "cacti"
        f.attrs["tier"] = tier_name
        f.attrs["version"] = "2.0"

        for i, s in enumerate(samples):
            x, mask, T = s["x"], s["mask"], s["T"]
            if mask is None:
                mask = gen_mask(x.shape[0], x.shape[1], T, rng)
            y, H_ideal = apply_cacti_mismatch(x, mask, true_spec)
            y = add_noise(y, rng)

            H, W = x.shape[:2]
            print(f"    sample_{i:02d} ({s['scene']}): ({H},{W},{T}) [{s['source']}]")

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("y", data=y, compression="gzip", compression_opts=4)
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip",
                               compression_opts=4)
            grp.attrs["spec_ranges"] = json.dumps(SPEC_RANGES)
            grp.attrs["metadata"] = json.dumps({
                "scene": s["scene"], "shape": list(x.shape), "T": T,
                "noise_model": "poisson_gaussian", "source": s["source"],
            })
            if include_gt:
                grp.create_dataset("x_true", data=x, compression="gzip",
                                   compression_opts=4)
            if include_true_spec:
                grp.attrs["true_spec"] = json.dumps(true_spec)

    return out_path


# ── Image generation ─────────────────────────────────────────────────────────

def save_img(data, path, title="", cmap="gray", vmin=None, vmax=None, dpi=150):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    norm = Normalize(vmin=vmin if vmin is not None else data.min(),
                     vmax=vmax if vmax is not None else data.max())
    ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def save_overview(y, H, x_true, out_dir, label, T):
    show_t = min(T, 8)
    indices = np.linspace(0, T-1, show_t, dtype=int)
    has_gt = x_true is not None
    n_rows = 3 if has_gt else 2
    fig, axes = plt.subplots(n_rows, show_t, figsize=(2*show_t, 2*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if show_t == 1:
        axes = axes[:, np.newaxis]
    for j in range(show_t):
        axes[0, j].axis('off')
    mid = show_t // 2
    axes[0, mid].imshow(y, cmap='viridis', interpolation='nearest')
    axes[0, mid].set_title('Measurement y', fontsize=8)
    for j, t in enumerate(indices):
        axes[1, j].imshow(H[:, :, t], cmap='gray', interpolation='nearest')
        axes[1, j].set_title(f'Mask {t}', fontsize=7)
        axes[1, j].axis('off')
    if has_gt:
        for j, t in enumerate(indices):
            axes[2, j].imshow(x_true[:, :, t], cmap='gray', interpolation='nearest')
            axes[2, j].set_title(f'GT {t}', fontsize=7)
            axes[2, j].axis('off')
    fig.suptitle(f"CACTI — {label} (T={T})", fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_dir / "overview.png"), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def generate_tier_images(h5_path, tier_name, img_dir):
    print(f"\n  Generating images for {tier_name} tier ...")
    ensure_dir(img_dir)
    with h5py.File(str(h5_path), 'r') as f:
        for sk in sorted(f.keys()):
            grp = f[sk]
            meta = json.loads(grp.attrs['metadata'])
            scene, T = meta['scene'], meta.get('T', 8)
            sd = img_dir / f"{sk}_{scene}"
            ensure_dir(sd)
            y, H_ideal = grp['y'][:], grp['H_ideal'][:]
            x_true = grp['x_true'][:] if 'x_true' in grp else None
            info = f"y={y.shape}, H={H_ideal.shape}, T={T}"
            if x_true is not None:
                info += f", x_true={x_true.shape}"
            print(f"    {sk} ({scene}): {info}")

            save_img(y, sd / "measurement_y.png",
                     title=f"Measurement y — {scene}", cmap="viridis")
            for t in range(H_ideal.shape[2]):
                save_img(H_ideal[:, :, t], sd / f"mask_frame_{t:02d}.png",
                         title=f"Mask {t}/{T-1} — {scene}", cmap="gray")
            if x_true is not None:
                for t in range(x_true.shape[2]):
                    save_img(x_true[:, :, t], sd / f"ground_truth_frame_{t:02d}.png",
                             title=f"GT Frame {t}/{T-1} — {scene}", cmap="gray")
            save_overview(y, H_ideal, x_true, sd, scene, T)


# ── README content ───────────────────────────────────────────────────────────

MAIN_README = """\
# CACTI — Coded Aperture Compressive Temporal Imaging

## Overview

CACTI is a computational imaging technique that captures high-speed video
from a single 2D snapshot measurement via a time-varying coded aperture.

This package provides benchmark data for evaluating reconstruction algorithms
under **forward-model mismatch**.

## Forward Model

```
y(h,w) = gain · Σ_t Φ_mismatch(h,w,t) · x(h,w,t) + offset + noise
```

## Dataset Design

| Tier   | Source               | Spatial  | Samples | T values    | Access           |
|--------|----------------------|----------|---------|-------------|------------------|
| Public | CACTI sim videos     | 256×256  | 20      | 8           | Full (GT+spec)   |
| Dev    | Procedural (mild)    | 512×512  | 20      | 8, 16, 32   | Blind (y+spec)   |
| Hidden | Procedural (hard)    | 512×512  | 20      | 8, 16, 32   | Server-only      |

**Dev/Hidden are generated procedurally** — no external datasets. The generator
code + secret seeds (kept private on PWM servers) fully determine each sample.
Even though the generator code is included, the derived datasets are
unreproducible without the private seed manifest.

## Procedural Scene Types

| Recipe ID | Name        | Difficulty | Description                              |
|-----------|-------------|------------|------------------------------------------|
| 0         | urban       | easy-med   | Rectangle blobs + grid patterns          |
| 1         | nature      | easy       | Smooth textures + large soft objects     |
| 2         | textile     | hard       | Near-periodic textures (stripes/checker) |
| 3         | particles   | hard       | Many tiny moving dots                    |
| 4         | thin_struct | hard       | Lines, wires, strokes                    |
| 5         | occlusion   | medium     | Layered objects crossing paths           |
| 6         | cam_shake   | medium     | Strong global camera motion              |

Dev uses mostly: urban, nature, occlusion (easy-medium)
Hidden adds: textile, particles, thin_struct, cam_shake (hard)

## T-value Distribution

- **Dev**: 40% T=8, 40% T=16, 20% T=32
- **Hidden**: 30% T=8, 30% T=16, 40% T=32 (harder, long integration)

## Mismatch Parameters

| Parameter        | Range           | Dev (mild)  | Hidden (severe) |
|------------------|-----------------|-------------|-----------------|
| `mask_dx`        | [0.2, 0.8] px  | 0.35        | 0.65            |
| `mask_dy`        | [0.1, 0.5] px  | 0.20        | 0.40            |
| `mask_rotation`  | [0.0, 0.3] deg | 0.08        | 0.22            |
| `mask_blur`      | [0.0, 0.5] px  | 0.10        | 0.35            |
| `clock_offset`   | [-0.1, 0.1]    | -0.03       | 0.08            |
| `gain_drift`     | [0.95, 1.05]   | 0.98        | 1.04            |
| `offset_drift`   | [-0.02, 0.02]  | -0.01       | 0.015           |

## Scoring

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency
```

## References

- Llull et al. "Coded aperture compressive temporal imaging." Opt. Express 2013.
- Yuan et al. "Snapshot compressive imaging." IEEE SPM 2021.
- PWM Benchmark: https://pwm.platformai.org/benchmark/cacti
"""

PUBLIC_README = """\
# CACTI Public Tier

Full-access data for algorithm development. All measurement groups from
6 CACTI simulation video scenes at 256×256, T=8.
Includes ground truth frames and true mismatch values.

## Scenes & Measurement Groups

| Scene   | Frames | Groups | Description              |
|---------|--------|--------|--------------------------|
| kobe    | 32     | 4      | Basketball player        |
| traffic | 48     | 6      | Highway traffic          |
| runner  | 8      | 1      | Running athlete          |
| drop    | 8      | 1      | Falling water drop       |
| crash   | 32     | 4      | Vehicle crash            |
| aerial  | 32     | 4      | Aerial/bird's-eye view   |
| **Total** |      | **20** |                          |

Each group contains T=8 consecutive frames compressed into one snapshot.

Per sample: `y` (256,256), `H_ideal` (256,256,8), `x_true` (256,256,8)

True mismatch: mask_dx=0.50, mask_dy=0.30, mask_rotation=0.10,
mask_blur=0.0, clock_offset=0.05, gain_drift=1.02, offset_drift=0.002

Noise: peak_photon=10000, gaussian_sigma=1.0 (~40 dB measurement SNR)
Forward model: binarized warped mask (matching InverseNet paper)
"""

DEV_README = """\
# CACTI Dev Tier

Blind evaluation. 20 procedurally generated samples at 512×512,
T ∈ {8, 16, 32}. Ground truth and mismatch values hidden.

Measurements can be shared to registered users (not publicly indexed).

## Samples

| #  | Recipe     | T  | Difficulty |
|----|------------|----|------------|
| 00 | urban      | 8  | easy-med   |
| 01 | nature     | 8  | easy       |
| 02 | occlusion  | 8  | medium     |
| 03 | urban      | 8  | easy-med   |
| 04 | nature     | 8  | easy       |
| 05 | occlusion  | 8  | medium     |
| 06 | urban      | 8  | easy-med   |
| 07 | nature     | 8  | easy       |
| 08 | cam_shake  | 16 | medium     |
| 09 | occlusion  | 16 | medium     |
| 10 | urban      | 16 | easy-med   |
| 11 | nature     | 16 | easy       |
| 12 | occlusion  | 16 | medium     |
| 13 | cam_shake  | 16 | medium     |
| 14 | urban      | 16 | easy-med   |
| 15 | nature     | 16 | easy       |
| 16 | occlusion  | 32 | medium     |
| 17 | urban      | 32 | easy-med   |
| 18 | nature     | 32 | easy       |
| 19 | cam_shake  | 32 | medium     |

Per sample: `y` (512,512), `H_ideal` (512,512,T). No ground truth.

Content: moderate motion, normal lighting, limited occlusion, mild camera shake.
Mismatch: mild (mask_dx=0.35, gain_drift=0.98, etc.)
"""

HIDDEN_README = """\
# CACTI Hidden Tier

Server-side evaluation only. 20 procedurally generated samples at 512×512,
T ∈ {8, 16, 32}. Strongest mismatch. Never leaves PWM servers.

## Samples

| #  | Recipe      | T  | Difficulty |
|----|-------------|----|------------|
| 00 | textile     | 8  | hard       |
| 01 | particles   | 8  | hard       |
| 02 | thin_struct | 8  | hard       |
| 03 | cam_shake   | 8  | medium     |
| 04 | textile     | 8  | hard       |
| 05 | particles   | 8  | hard       |
| 06 | thin_struct | 16 | hard       |
| 07 | urban       | 16 | easy-med   |
| 08 | nature      | 16 | easy       |
| 09 | occlusion   | 16 | medium     |
| 10 | textile     | 16 | hard       |
| 11 | particles   | 16 | hard       |
| 12 | thin_struct | 32 | hard       |
| 13 | cam_shake   | 32 | medium     |
| 14 | textile     | 32 | hard       |
| 15 | particles   | 32 | hard       |
| 16 | thin_struct | 32 | hard       |
| 17 | urban       | 32 | easy-med   |
| 18 | nature      | 32 | easy       |
| 19 | occlusion   | 32 | medium     |

~60-70% hard stress tests + ~30-40% plausible scenes for fairness.

Per sample: `y`, `H_ideal`, `x_true` (512,512,T). True mismatch stored.
Mismatch: severe (mask_dx=0.65, mask_blur=0.35, gain_drift=1.04, etc.)
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Building CACTI dataset package at: {PACKAGE_DIR}\n")

    # Keep generator file, remove everything else
    gen_file = PACKAGE_DIR / "procedural_video_generator.py"
    gen_code = gen_file.read_text() if gen_file.exists() else None

    # Files to preserve during cleanup
    PRESERVE_FILES = {"procedural_video_generator.py", "modal_runner.py"}

    if PACKAGE_DIR.exists():
        print("  Cleaning existing package ...")
        for item in PACKAGE_DIR.iterdir():
            if item.name in PRESERVE_FILES:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    for tier in ["public", "dev", "hidden"]:
        ensure_dir(PACKAGE_DIR / tier / "images")

    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Load / generate source data")
    print("=" * 60)

    print("\n  Loading Public (CACTI simulation) ...")
    public_samples = load_public_samples()
    print(f"    -> {len(public_samples)} samples")

    print("\n  Generating Dev (procedural, mild) ...")
    dev_samples = load_procedural_samples(DEV_MANIFEST, "dev")
    print(f"    -> {len(dev_samples)} samples")

    print("\n  Generating Hidden (procedural, hard) ...")
    hidden_samples = load_procedural_samples(HIDDEN_MANIFEST, "hidden")
    print(f"    -> {len(hidden_samples)} samples")

    # Step 2: Generate HDF5
    print("\n" + "=" * 60)
    print("Step 2: Generate challenge HDF5 files")
    print("=" * 60)

    h5p = generate_tier_h5("public", public_samples, TRUE_SPEC_PUBLIC,
                           1001, True, True, PACKAGE_DIR / "public")
    h5d = generate_tier_h5("dev", dev_samples, TRUE_SPEC_DEV,
                           2001, True, True, PACKAGE_DIR / "dev")
    h5h = generate_tier_h5("hidden", hidden_samples, TRUE_SPEC_HIDDEN,
                           3001, True, True, PACKAGE_DIR / "hidden")

    # Step 3: Generate images
    print("\n" + "=" * 60)
    print("Step 3: Generate images")
    print("=" * 60)
    for name, h5 in [("public", h5p), ("dev", h5d), ("hidden", h5h)]:
        generate_tier_images(h5, name, PACKAGE_DIR / name / "images")

    # Step 4: Write spec.json and true_spec.json
    print("\n" + "=" * 60)
    print("Step 4: Write spec.json and true_spec.json")
    print("=" * 60)
    for tier_name, true_spec, include_true in [
        ("public", TRUE_SPEC_PUBLIC, True),
        ("dev", TRUE_SPEC_DEV, True),
        ("hidden", TRUE_SPEC_HIDDEN, True),
    ]:
        tier_dir = PACKAGE_DIR / tier_name
        spec_path = tier_dir / "spec.json"
        spec_path.write_text(json.dumps(SPEC_RANGES, indent=2) + "\n")
        print(f"  Wrote {spec_path}")
        if include_true:
            ts_path = tier_dir / "true_spec.json"
            ts_path.write_text(json.dumps(true_spec, indent=2) + "\n")
            print(f"  Wrote {ts_path}")

    # Step 5: Write READMEs
    print("\n" + "=" * 60)
    print("Step 5: Write README files")
    print("=" * 60)
    for rp, txt in [("README.md", MAIN_README), ("public/README.md", PUBLIC_README),
                    ("dev/README.md", DEV_README), ("hidden/README.md", HIDDEN_README)]:
        p = PACKAGE_DIR / rp
        p.write_text(txt)
        print(f"  Wrote {p}")

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    n_img = sum(1 for _ in PACKAGE_DIR.rglob("*.png"))
    n_h5 = sum(1 for _ in PACKAGE_DIR.rglob("*.h5"))
    n_md = sum(1 for _ in PACKAGE_DIR.rglob("*.md"))
    print(f"  Package: {PACKAGE_DIR}")
    print(f"  Images:  {n_img}  |  HDF5: {n_h5}  |  READMEs: {n_md}")
    for d in ["public", "dev", "hidden"]:
        ni = sum(1 for _ in (PACKAGE_DIR / d).rglob("*.png"))
        nh = sum(1 for _ in (PACKAGE_DIR / d).rglob("*.h5"))
        print(f"    {d}: {nh} HDF5, {ni} images")


if __name__ == "__main__":
    main()
