# MRI — Multi-Coil Parallel Imaging (fastMRI Knee, 4-Knob Mismatch)

## Overview

Magnetic Resonance Imaging (MRI) acquires images by measuring the Fourier
transform of the magnetisation density in k-space.  This benchmark uses an
**15-coil parallel imaging 2D Cartesian acquisition** with 4× acceleration
(variable-density random undersampling, 8% ACS centre fraction).

The dataset is anchored to the widely-used **fastMRI multi-coil knee** corpus
(Zbontar et al., 2018):
- Sequence: 2D Cartesian Turbo Spin Echo (TSE), T2-weighted
- Resolution: 320 × 320 pixels
- Coils: 15 receive elements
- Acceleration: R = 4 (variable-density Cartesian, ACS fraction 0.08)

The mismatch scenario combines four physically motivated sources of forward-model
error that commonly appear together in real scanners:

1. **B0 field inhomogeneity** — spatially-varying phase ramp in image domain
2. **Gradient non-linearity** — geometric warp of the imaged object
3. **Coil sensitivity perturbation** — smooth complex multiplicative error in each coil map
4. **k-space trajectory error** — per-line phase ramp from gradient timing delays

## Forward Model

**Ideal (assumed by reconstructor):**

```
y_c = F_u · S_c · x + n_c       for c = 1 … 15
```

**True acquisition (with 4-knob mismatch):**

```
Step 1  (gradient nonlinearity):   x'      = warp(x, δr)
Step 2  (B0 inhomogeneity):        x''     = x' · exp(i · 2π · B0_hz · TE · b0_map)
Step 3  (coil sensitivity error):  y_c_raw = F(S_c_true · x'')    S_c_true = S_c · (1 + ε_c)
Step 4  (k-trajectory error):      y_c[ky] = y_c_raw[ky] · exp(i · 2π · Δk_ky · kx)
Step 5  (noise):                   y_c     = mask · y_c + N(0, σ²)
```

Where:
- **x** ∈ ℝ^{320×320} — MR magnitude image (ground truth)
- **C = 15** — number of receive coils
- **S_c** ∈ ℂ^{320×320} — nominal coil sensitivity map for coil c
- **F** — centred 2D Discrete Fourier Transform
- **F_u** — undersampled F (VDS mask in ky)
- **mask** ∈ {0,1}^{320} — 1D Cartesian ky undersampling mask
- **b0_map** ∈ [-1, 1]^{320×320} — smooth B0 field inhomogeneity map
- **B0_hz** — scalar field offset in Hz
- **TE** = 25 ms — echo time
- **δr** — smooth 2D displacement field (gradient non-linearity warp)
- **ε_c** — smooth complex perturbation on coil sensitivity map c
- **Δk_ky** — per-line k-space fractional shift (trajectory error)
- **σ** — complex Gaussian noise level relative to k-space RMS

## Mismatch Parameters

| Parameter              | Description                          | Dev range       | Hidden range    | Unit          |
|------------------------|--------------------------------------|-----------------|-----------------|---------------|
| `B0_inhomog_hz`        | B0 field inhomogeneity offset        | [5, 20]         | [20, 60]        | Hz            |
| `gradient_nonlin_frac` | Gradient non-linearity warp strength | [0.001, 0.005]  | [0.005, 0.02]   | frac of FOV   |
| `coil_sensitivity_frac`| Coil sensitivity perturbation amp.   | [0.01, 0.05]    | [0.05, 0.15]    | frac          |
| `k_trajectory_frac`    | k-trajectory per-line shift          | [0.001, 0.005]  | [0.005, 0.02]   | frac of kmax  |
| `noise_sigma`          | Complex Gaussian noise level         | [0.01, 0.03]    | [0.03, 0.06]    | rel           |

Each sample has its **own independently randomized** mismatch parameters.
Algorithms must estimate or compensate for all per-sample deviations.

## Scoring

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency
```

RSS reconstruction of the zero-filled multi-coil k-space is the baseline.

## Dataset Structure

```
mri/
├── README.md                     ← This file
├── simulate_scenes.py            ← Procedural knee phantom generator (dev/hidden)
├── build_dataset.py              ← Builds H5 files + PNG images from scratch
├── public/                       ← fastMRI knee real data (11 samples)
│   ├── README.md
│   ├── mri_challenge_public.h5
│   └── images/
├── dev/                          ← Procedural knee-like (20 samples, mild mismatch)
│   ├── README.md
│   ├── mri_challenge_dev.h5
│   └── images/
└── hidden/                       ← Adversarial stress-test (20 samples, severe mismatch)
    ├── README.md
    ├── mri_challenge_hidden.h5
    └── images/
```

## Scene Assignment

| Tier   | Source                           | Samples | Mismatch | Access           |
|--------|----------------------------------|---------|----------|------------------|
| Public | fastMRI multi-coil knee (real)   | 11      | Mild     | Full (GT + spec) |
| Dev    | Procedural knee TSE phantoms     | 20      | Mild     | Blind (y + spec) |
| Hidden | Adversarial knee phantoms        | 20      | Severe   | Server-only      |

## Using Real fastMRI Data (Public Tier)

The public tier is designed to use real fastMRI multi-coil knee raw k-space data.
To rebuild with real data:

```bash
# 1. Register and download from https://fastmri.med.nyu.edu/
#    (Knee MRI — multicoil_train or multicoil_val)

# 2. Point the builder at your download directory
export FASTMRI_ROOT=/path/to/knee_multicoil_train

# 3. Rebuild
python build_dataset.py
```

Without `FASTMRI_ROOT`, the public tier falls back to Shepp-Logan synthetic placeholders
(clearly labelled in metadata `"source": "synthetic"`).

## Procedural Knee Scene Types (Dev)

| Recipe                 | Mix  | Description                                            |
|------------------------|------|--------------------------------------------------------|
| `knee_coronal_normal`  | 55%  | Coronal TSE: condyles, tibial plateau, mild effusion   |
| `knee_coronal_effusion`| 30%  | Prominent synovial fluid (very bright joint space)     |
| `knee_axial_patella`   | 15%  | Axial: patella, trochlear groove, Hoffa fat pad        |

## Procedural Knee Scene Types (Hidden)

| Recipe                  | Mix  | Description                                           |
|-------------------------|------|-------------------------------------------------------|
| `knee_osteophyte`       | 35%  | Bony spurs on condyle margins (dark cortex stress)    |
| `knee_multicompartment` | 35%  | Baker's cyst + extra posterior fluid collection       |
| `knee_high_contrast`    | 20%  | Extreme fluid/bone ratio, low muscle signal           |
| `knee_thin_cartilage`   | 10%  | Very thin/absent articular cartilage (edge stress)    |

## HDF5 File Format

```python
import h5py, json
import numpy as np

with h5py.File("mri_challenge_dev.h5", "r") as f:
    for sample_key in sorted(f.keys()):
        grp = f[sample_key]

        # Core arrays
        x_true     = grp["x_true"][:]      # (320, 320)     float32  — ground truth image
        y_kspace   = grp["y_kspace"][:]    # (15, 320, 320) complex64 — undersampled k-space
        mask       = grp["mask"][:]        # (320,)          uint8    — 1D ky mask
        coil_maps  = grp["coil_maps"][:]   # (15, 320, 320) complex64 — nominal coil maps
        b0_map     = grp["B0_map"][:]      # (320, 320)     float32  — B0 field map
        warp_field = grp["warp_field"][:]  # (2, 320, 320)  float32  — (dy, dx) warp in pixels

        # Metadata
        metadata    = json.loads(grp.attrs["metadata"])
        spec_ranges = json.loads(grp.attrs["spec_ranges"])   # per-tier parameter bounds
        true_spec   = json.loads(grp.attrs["true_spec"])     # per-sample ground-truth params!
```

### true_spec fields

```python
{
    "B0_inhomog_hz":         float,  # Hz — B0 offset used in acquisition
    "gradient_nonlin_frac":  float,  # fraction of FOV — warp amplitude
    "coil_sensitivity_frac": float,  # fractional perturbation amplitude
    "k_trajectory_frac":     float,  # fractional k-space shift per line
    "noise_sigma":           float,  # relative noise level
}
```

### Zero-filled RSS baseline

```python
def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1))

rss = np.sqrt(np.sum(np.abs(np.stack([ifft2c(y_kspace[c]) for c in range(15)])) ** 2, axis=0))
rss /= rss.max()  # normalise to [0, 1]
```

## Coil Sensitivity Estimation (Public Tier)

For real fastMRI samples, nominal coil maps are estimated from the ACS region using
low-pass filtered coil images divided by RSS:

```
S_c(r) = LPF(coil_c)(r) / RSS_LPF(r)
```

This provides the reconstructor's "best available" sensitivity estimate, consistent
with the mismatch scenario (true maps differ by the `coil_sensitivity_frac` perturbation).

## Baseline Performance (Scenario II — mismatch, no correction)

| Method               | PSNR (dB) | SSIM  |
|----------------------|-----------|-------|
| Zero-filled RSS      | 26.2      | 0.710 |
| SENSE (no B0 corr.)  | 30.1      | 0.820 |
| PnP-DRUNet           | 28.5      | 0.798 |
| E2E-VarNet           | 31.8      | 0.851 |

## References

- Zbontar, J., et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI."
  arXiv:1811.08839 (2018). https://arxiv.org/abs/1811.08839
- Pruessmann, K.P., et al. "SENSE: sensitivity encoding for fast MRI."
  MRM 42.5 (1999): 952–962.
- Lustig, M., Donoho, D., Pauly, J.M. "Sparse MRI: The application of
  compressed sensing for rapid MR imaging." MRM 58.6 (2007): 1182–1195.
- Hammernik, K., et al. "Learning a variational network for reconstruction of
  accelerated MRI data." MRM 79.6 (2018): 3055–3071.
- PWM Benchmark: https://pwm.platformai.org/benchmark/mri
