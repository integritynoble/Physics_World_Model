# 04 — PWM MRI Benchmark

## 1. Overview

The PWM MRI benchmark evaluates reconstruction algorithms under
**physics model mismatch** — the gap between what the algorithm assumes
and what actually happened during acquisition. This is the central
challenge: real MRI scanners are never perfectly calibrated.

The benchmark uses a **3-tier** structure with increasing mismatch
severity, testing how gracefully algorithms degrade under realistic
conditions.

---

## 2. Three-Tier Structure

### 2.1 Tier Summary

| Tier | Samples | Mismatch | Data Source | Purpose |
|------|:-------:|----------|-------------|---------|
| **Public** | 11 | Mild | Synthetic / fastMRI brain | Algorithm development, debugging |
| **Dev** | 20 | Moderate | IXI T2w healthy brains | Validation, hyperparameter tuning |
| **Hidden** | 20 | Severe | BraTS T2w pathological brains | Final evaluation, leaderboard |

### 2.2 Design Philosophy

- **Public**: easy enough that all algorithms should work reasonably.
  Use this for development and debugging.
- **Dev**: moderate mismatch exposes the first signs of algorithm
  fragility. Use this for validation and tuning.
- **Hidden**: severe mismatch pushes algorithms to their limits. The
  performance gap between public and hidden reveals true robustness.

### 2.3 Data Sources

| Tier | Priority 1 | Priority 2 | Env Variable |
|------|-----------|-----------|--------------|
| Public | fastMRI brain multi-coil | Synthetic phantoms | `FASTMRI_BRAIN_ROOT` |
| Dev | IXI T2w healthy brains (578 subjects, 3 sites) | Synthetic | `IXI_T2_ROOT` |
| Hidden | BraTS T2w pathological brains | Synthetic | `BRATS_ROOT` |

The dev tier uses real IXI T2w brain MRI data, loaded via
`load_ixi_t2_slices()` from `real_loaders.py`. The hidden tier uses
BraTS pathological brains, providing a harder reconstruction target
due to tumour-related signal abnormalities.

---

## 3. HDF5 File Format

Each tier is stored as a single HDF5 file with groups `sample_00`,
`sample_01`, etc.

### 3.1 File Locations

```
datasets/benchmark/mri/
├── public/
│   └── mri_challenge_public.h5    (170 MB, 11 samples)
├── dev/
│   └── mri_challenge_dev.h5       (308 MB, 20 samples)
└── hidden/
    └── mri_challenge_hidden.h5    (307 MB, 20 samples)
```

### 3.2 Dataset Keys and Shapes

Each sample group contains these datasets:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (320, 320) | float32 | Ground-truth magnitude image, normalised to [0, 1] |
| `y_kspace` | (15, 320, 320) | complex64 | Undersampled multi-coil k-space (with mismatch) |
| `mask` | (320,) | uint8 | 1D Cartesian undersampling mask (0 or 1) |
| `coil_maps` | (15, 320, 320) | complex64 | Nominal coil sensitivity maps |
| `B0_map` | (320, 320) | float32 | B₀ field inhomogeneity map, normalised to [-1, 1] |
| `warp_field` | (2, 320, 320) | float32 | Gradient nonlinearity warp (dy, dx) in pixels |

### 3.3 Metadata Attributes

Each sample group has three JSON string attributes:

#### `metadata`
```json
{
  "scene": "brain_ixi_042",
  "shape": [320, 320],
  "n_coils": 15,
  "accel_factor": 4,
  "acs_frac": 0.08,
  "te_s": 0.025,
  "recipe": "real_ixi_t2",
  "n_sampled_lines": 80,
  "source": "ixi_t2"
}
```

#### `spec_ranges`
The SPEC_RANGES entry for this tier (see section 4 below).

#### `true_spec`
The actual mismatch parameter values used for this specific sample:
```json
{
  "B0_inhomog_hz": 12.3,
  "gradient_nonlin_frac": 0.0024,
  "coil_sensitivity_frac": 0.035,
  "k_trajectory_frac": 0.0018,
  "noise_sigma": 0.022
}
```

### 3.4 Loading a Sample

```python
import h5py
import numpy as np
import json

with h5py.File("datasets/benchmark/mri/dev/mri_challenge_dev.h5", "r") as hf:
    grp = hf["sample_00"]

    x_true    = grp["x_true"][:]       # (320, 320) float32
    y_kspace  = grp["y_kspace"][:]     # (15, 320, 320) complex64
    mask      = grp["mask"][:]         # (320,) uint8
    coil_maps = grp["coil_maps"][:]    # (15, 320, 320) complex64
    b0_map    = grp["B0_map"][:]       # (320, 320) float32
    warp      = grp["warp_field"][:]   # (2, 320, 320) float32

    metadata  = json.loads(grp.attrs["metadata"])
    true_spec = json.loads(grp.attrs["true_spec"])
```

---

## 4. Mismatch Ranges (SPEC_RANGES)

Defined in `build_dataset.py` lines 74–96:

### 4.1 Parameter Table

| Parameter | Unit | Public (Mild) | Dev (Moderate) | Hidden (Severe) |
|-----------|------|:-------------:|:--------------:|:---------------:|
| B₀ inhomogeneity | Hz | 5 – 15 | 5 – 20 | 20 – 60 |
| Gradient nonlinearity | frac | 0.001 – 0.003 | 0.001 – 0.005 | 0.005 – 0.02 |
| Coil sensitivity error | frac | 0.01 – 0.03 | 0.01 – 0.05 | 0.05 – 0.15 |
| k-trajectory deviation | frac | 0.001 – 0.003 | 0.001 – 0.005 | 0.005 – 0.02 |
| Noise σ | relative | 0.01 – 0.02 | 0.01 – 0.03 | 0.03 – 0.06 |

### 4.2 Mismatch Effects

| Parameter | Physical Effect | Reconstruction Impact |
|-----------|----------------|----------------------|
| B₀ inhomogeneity | Spatially varying phase | Blurring, signal voids |
| Gradient nonlinearity | Geometric distortion | Misaligned features |
| Coil sensitivity error | Wrong unfolding weights | SENSE ghosts, intensity errors |
| k-trajectory deviation | Phase ramps in k-space | Ghosting, striping |
| Noise | Random perturbation | Overall SNR degradation |

### 4.3 Severity Progression

The hidden tier has **4–10×** stronger mismatch than the public tier:
- B₀: 15 Hz max → 60 Hz max (4×)
- Coil error: 3% max → 15% max (5×)
- Gradient nonlin: 0.3% max → 2% max (6.7×)
- Noise: 2% max → 6% max (3×)

---

## 5. Acquisition Parameters

These are constant across all tiers:

| Parameter | Value | Variable |
|-----------|-------|----------|
| Image size | 320 × 320 | `SHAPE` |
| Number of coils | 15 | `N_COILS` |
| Acceleration factor | 4 | `ACCEL` |
| ACS fraction | 8% (≈26 lines) | `ACS_FRAC` |
| Echo time | 25 ms | `TE_S` |
| Sampling pattern | Variable-density Cartesian | — |
| Mask dimension | 1D (phase-encode) | — |

---

## 6. Scoring

### 6.1 Primary Metrics

| Metric | Function | Range | Goal |
|--------|----------|-------|------|
| **PSNR** (primary) | `compute_psnr(x_true, x_hat, max_val=1.0)` | 0 – 100 dB | Higher is better |
| **SSIM** | `compute_ssim(x_true, x_hat, data_range=1.0)` | 0 – 1.0 | Higher is better |

Both metrics are computed on magnitude images normalised to [0, 1].

Source: `benchmarks/framework/metrics.py`

### 6.2 Normalisation

Before computing metrics, reconstructions should be normalised:

```python
x_hat = np.abs(x_hat).astype(np.float32)  # magnitude
x_hat = x_hat / (x_hat.max() + 1e-10)     # normalise to [0, 1]
```

### 6.3 Expected Performance Ranges

| Algorithm | Public PSNR | Dev PSNR | Hidden PSNR |
|-----------|:-----------:|:--------:|:-----------:|
| Zero-Filled RSS | 24–27 | 22–26 | 20–24 |
| SENSE | 28–35 | 26–32 | 24–30 |
| CS-MRI | 28–35 | 26–32 | 24–30 |
| PnP-HQS | 28–35 | 26–32 | 24–30 |
| VarNet (random) | 22–28 | 20–26 | 18–24 |
| MoDL (random) | 22–28 | 20–26 | 18–24 |

VarNet and MoDL with random weights will perform near the zero-filled
baseline. With properly trained weights they would be expected to
significantly outperform classical methods.

---

## 7. Submission Format

### 7.1 JSON Output Structure

```json
{
  "modality": "mri",
  "timestamp": "2026-03-02T12:00:00Z",
  "shape": [320, 320],
  "n_coils": 15,
  "acceleration": 4,
  "tiers": {
    "public": {
      "n_samples": 11,
      "per_sample": [
        {
          "idx": 0,
          "scene": "brain_synthetic_00",
          "mismatch": {
            "B0_inhomog_hz": 10.5,
            "gradient_nonlin_frac": 0.002,
            "coil_sensitivity_frac": 0.02,
            "k_trajectory_frac": 0.002,
            "noise_sigma": 0.015
          },
          "solvers": {
            "zerofilled_rss": {"psnr": 25.3, "ssim": 0.72, "time_s": 0.1},
            "sense": {"psnr": 31.2, "ssim": 0.88, "time_s": 2.3}
          }
        }
      ],
      "aggregate": {
        "zerofilled_rss": {
          "mean_psnr": 25.1, "std_psnr": 1.2,
          "mean_ssim": 0.71, "std_ssim": 0.03,
          "mean_time_s": 0.1
        }
      }
    }
  },
  "cross_tier_summary": {
    "zerofilled_rss": {
      "public_psnr": 25.1, "dev_psnr": 24.5, "hidden_psnr": 22.8
    }
  }
}
```

### 7.2 Running the Benchmark

```bash
# All tiers, all solvers
python papers/pwm_flagship/scripts/run_mri_multiphantom.py

# Quick test: public tier only, 2 samples
python papers/pwm_flagship/scripts/run_mri_multiphantom.py \
    --tier public --max-samples 2

# Specific solver
python papers/pwm_flagship/scripts/run_mri_multiphantom.py \
    --solver sense --tier dev
```

---

## 8. Verification

Before running the benchmark, verify datasets with:

```bash
python datasets/benchmark/mri/verify_datasets.py --tier all
```

This checks all shapes, dtypes, value ranges, and metadata for every
sample in every tier. Exit code 0 means all checks passed.

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
