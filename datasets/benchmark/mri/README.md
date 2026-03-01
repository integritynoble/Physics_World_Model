# MRI — Multi-Coil Parallel Imaging (4-Knob Mismatch)

## Overview

Magnetic Resonance Imaging (MRI) acquires images by measuring the Fourier
transform of the magnetisation density in k-space.  This benchmark uses an
**8-coil parallel imaging 2D Cartesian acquisition** with 4× acceleration
(variable-density random undersampling along the phase-encode direction).

The mismatch scenario combines four physically motivated sources of forward-model
error that commonly appear together in real scanners:

1. **B0 field inhomogeneity** — spatially-varying phase ramp in image domain
2. **Gradient non-linearity** — geometric warp of the imaged object
3. **Coil sensitivity perturbation** — smooth complex multiplicative error in each coil map
4. **k-space trajectory error** — per-line phase ramp from timing/gradient delays

## Forward Model

**Ideal (assumed by reconstructor):**

```
y_c = F_u · S_c · x + n_c       for c = 1 … C
```

**True acquisition (with 4-knob mismatch):**

```
Step 1  (gradient nonlinearity):   x'      = warp(x, δr)
Step 2  (B0 inhomogeneity):        x''     = x' · exp(i · 2π · B0_hz · TE · b0_map)
Step 3  (coil sensitivity error):  y_c_raw = F(S_c_true · x'')      S_c_true = S_c · (1 + ε_c)
Step 4  (k-trajectory error):      y_c[ky] = y_c_raw[ky] · exp(i · 2π · Δk_ky · kx / W)
Step 5  (noise):                   y_c     = mask · y_c + N(0, σ²)
```

Where:
- **x** ∈ ℝ^{256×256} — MR magnitude image (ground truth)
- **C = 8** — number of receive coils (Gaussian ring geometry)
- **S_c** ∈ ℂ^{256×256} — nominal coil sensitivity map for coil c
- **F** — centred 2D Discrete Fourier Transform
- **F_u** — undersampled F (mask applied in ky)
- **mask** ∈ {0,1}^{256} — 1D Cartesian ky undersampling mask
- **b0_map** ∈ [-1, 1]^{256×256} — smooth B0 field inhomogeneity map
- **B0_hz** — scalar field offset in Hz
- **TE** = 25 ms — echo time
- **δr** — smooth 2D displacement field (gradient non-linearity warp)
- **ε_c** — smooth complex perturbation on coil sensitivity map c
- **Δk_ky** — per-line k-space shift fraction (trajectory error)
- **σ** — complex Gaussian noise level relative to k-space signal RMS

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

RSS reconstruction of the zero-filled multi-coil k-space is used as the baseline.

## Dataset Structure

```
mri/
├── README.md                     ← This file
├── simulate_scenes.py            ← Procedural brain phantom generator (dev/hidden)
├── build_dataset.py              ← Builds H5 files + PNG images from scratch
├── public/                       ← Shepp-Logan variants (11 samples)
│   ├── README.md
│   ├── mri_challenge_public.h5
│   └── images/
├── dev/                          ← Procedural brain-like (20 samples, mild mismatch)
│   ├── README.md
│   ├── mri_challenge_dev.h5
│   └── images/
└── hidden/                       ← Adversarial stress-test (20 samples, severe mismatch)
    ├── README.md
    ├── mri_challenge_hidden.h5
    └── images/
```

## Scene Assignment

| Tier   | Source                      | Samples | Mismatch | Access           |
|--------|-----------------------------|---------|----------|------------------|
| Public | Shepp-Logan variants        | 11      | Mild     | Full (GT + spec) |
| Dev    | Procedural (brain-like)     | 20      | Mild     | Blind (y + spec) |
| Hidden | Procedural (adversarial)    | 20      | Severe   | Server-only      |

## Procedural Scene Types (Dev)

| Recipe              | Mix  | Description                                       |
|---------------------|------|---------------------------------------------------|
| `gray_white_matter` | 60%  | GM/WM/CSF contrast, ventricles, scalp ring        |
| `with_vessels`      | 25%  | Brain + small bright vessel cross-sections        |
| `fat_saturated`     | 15%  | Fat-suppressed T2, no scalp ring                  |

## Procedural Scene Types (Hidden)

| Recipe                | Mix  | Description                                      |
|-----------------------|------|--------------------------------------------------|
| `lesion_pathological` | 35%  | Focal hyperintense lesions (T2 bright spots)     |
| `fine_structure`      | 35%  | Many vessels + fine-scale texture                |
| `high_contrast`       | 20%  | Extreme WM/GM contrast, HDR clipping stress      |
| `edge_heavy`          | 10%  | Many sharp tissue-boundary rims                  |

## HDF5 File Format

```python
import h5py, json
import numpy as np

with h5py.File("mri_challenge_dev.h5", "r") as f:
    for sample_key in sorted(f.keys()):
        grp = f[sample_key]

        # Core arrays
        x_true    = grp["x_true"][:]     # (256, 256)    float32  — ground truth image
        y_kspace  = grp["y_kspace"][:]   # (8, 256, 256) complex64 — undersampled multi-coil k-space
        mask      = grp["mask"][:]       # (256,)         uint8    — 1D ky undersampling mask
        coil_maps = grp["coil_maps"][:] # (8, 256, 256) complex64 — nominal coil sensitivity maps
        b0_map    = grp["B0_map"][:]     # (256, 256)    float32  — B0 field map (normalised)
        warp_field = grp["warp_field"][:] # (2, 256, 256) float32 — (dy,dx) displacement in pixels

        # Metadata attributes
        metadata    = json.loads(grp.attrs["metadata"])
        spec_ranges = json.loads(grp.attrs["spec_ranges"])  # parameter bounds for this tier
        true_spec   = json.loads(grp.attrs["true_spec"])    # per-sample ground-truth parameters!
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
# Zero-fill: treat unsampled lines as zero, then IFFT each coil
ifft2d = lambda k: np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k)))
rss = np.sqrt(np.sum(np.abs(np.stack([ifft2d(y_kspace[c]) for c in range(8)])) ** 2, axis=0))
rss /= rss.max()  # normalise to [0, 1]
```

## Baseline Performance (Scenario II — mismatch, no correction)

| Method              | PSNR (dB) | SSIM  |
|---------------------|-----------|-------|
| Zero-filled RSS     | 26.2      | 0.710 |
| SENSE (no B0 corr.) | 30.1      | 0.820 |
| PnP-DRUNet          | 28.5      | 0.798 |
| E2E-VarNet          | 31.8      | 0.851 |

## References

- Pruessmann, K.P., et al. "SENSE: sensitivity encoding for fast MRI." MRM 42.5 (1999): 952–962.
- Lustig, M., Donoho, D., Pauly, J.M. "Sparse MRI: The application of
  compressed sensing for rapid MR imaging." MRM 58.6 (2007): 1182–1195.
- Hammernik, K., et al. "Learning a variational network for reconstruction of
  accelerated MRI data." MRM 79.6 (2018): 3055–3071.
- PWM Benchmark: https://pwm.platformai.org/benchmark/mri
