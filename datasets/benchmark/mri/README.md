# MRI — Multi-Coil Parallel Imaging (Axial T2w Brain, 4-Knob Mismatch)

## Overview

Magnetic Resonance Imaging (MRI) acquires images by measuring the Fourier
transform of the magnetisation density in k-space.  This benchmark uses a
**15-coil parallel imaging 2D Cartesian acquisition** with 4× acceleration
(variable-density random undersampling, 8% ACS centre fraction).

Anatomy: **axial T2-weighted brain** slices (320 × 320 px).

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

| Parameter              | Description                          | Public range    | Dev range       | Hidden range    | Unit          |
|------------------------|--------------------------------------|-----------------|-----------------|-----------------|---------------|
| `B0_inhomog_hz`        | B0 field inhomogeneity offset        | [5, 15]         | [5, 20]         | [20, 60]        | Hz            |
| `gradient_nonlin_frac` | Gradient non-linearity warp strength | [0.001, 0.003]  | [0.001, 0.005]  | [0.005, 0.02]   | frac of FOV   |
| `coil_sensitivity_frac`| Coil sensitivity perturbation amp.   | [0.01, 0.03]    | [0.01, 0.05]    | [0.05, 0.15]    | frac          |
| `k_trajectory_frac`    | k-trajectory per-line shift          | [0.001, 0.003]  | [0.001, 0.005]  | [0.005, 0.02]   | frac of kmax  |
| `noise_sigma`          | Complex Gaussian noise level         | [0.01, 0.02]    | [0.01, 0.03]    | [0.03, 0.06]    | rel           |

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
├── simulate_scenes.py            ← Procedural brain T2w phantom generator (dev/hidden)
├── build_dataset.py              ← Builds H5 files + PNG images from scratch
├── public/                       ← Real axial T2w brain MRI (11 samples)
│   ├── README.md
│   ├── mri_challenge_public.h5
│   └── images/
├── dev/                          ← Procedural brain T2w phantoms (20 samples, mild mismatch)
│   ├── README.md
│   ├── mri_challenge_dev.h5
│   └── images/
└── hidden/                       ← Adversarial brain T2w phantoms (20 samples, severe mismatch)
    ├── README.md
    ├── mri_challenge_hidden.h5
    └── images/
```

## Scene Assignment

| Tier   | Source                              | Samples | Mismatch | Access           |
|--------|-------------------------------------|---------|----------|------------------|
| Public | Real multi-coil axial T2w brain MRI | 11      | Mild     | Full (GT + spec) |
| Dev    | Procedural brain T2w phantoms       | 20      | Mild     | Blind (y + spec) |
| Hidden | Adversarial brain T2w phantoms      | 20      | Severe   | Server-only      |

---

## Building the Dataset

### Prerequisites

```bash
pip install h5py numpy scipy Pillow
```

### Quick start

```bash
cd datasets/benchmark/mri
python build_dataset.py
```

Builds all three tiers in one pass.  The build is fully deterministic —
the same seeds always produce identical H5 files.

---

### Public Tier — Real Brain T2w MRI (11 samples)

Real multi-coil axial T2w brain MRI slices are read from:

```
datasets/real_mri/multicoil_val/
├── 2022061203_T201.h5   ← preferred (AXT2, T2-weighted)
├── 2022061203_T101.h5   ← fallback (AXT1)
└── 2022061204_T101.h5   ← fallback (AXT1)
```

The builder automatically:
- Prefers `T2`-labelled files over `T1`
- Skips the outer 20 % of slices (edge / scout slices)
- Bicubic-zooms from native resolution (256 × 256) to 320 × 320
- Uses RSS as `x_true` and applies a synthetic 15-coil forward model

**Custom path:**

```bash
export REAL_MRI_ROOT=/path/to/your/multicoil_val
python build_dataset.py
```

**Expected H5 format** (standard `reconstruction_rss` layout):

```
reconstruction_rss  : (n_slices, kH, kW)  float32
attrs["acquisition"]: e.g. "AXT2"
```

If `REAL_MRI_ROOT` is not set and no files are found at the default path,
the public tier falls back to synthetic brain T2w phantoms
(labelled `"source": "synthetic"` in metadata).

---

### Dev Tier — Procedural Brain T2w Phantoms (20 samples)

Seeds: `5000–5019`.  Mild mismatch.

**Recipes:**

| Recipe               | Mix  | Anatomy                                                    |
|----------------------|------|------------------------------------------------------------|
| `brain_t2_normal`    | 55%  | Mid-brain: gyral cortex ribbon, WM, lateral ventricles, basal ganglia, corpus callosum |
| `brain_t2_csf_rich`  | 30%  | Enlarged ventricles (hydrocephalus-like, very bright CSF)  |
| `brain_t2_posterior` | 15%  | Posterior fossa: cerebellum folia, brainstem, 4th ventricle, prepontine + CPA cisterns |

**Phantom anatomy (layered alpha compositing, back → front):**

```
background (0.0)
  → scalp fat      (~0.82)   SCALP_T  = 6.0 % of radius
  → calvarium bone (~0.03)   SKULL_T  = 4.8 %
  → subarachnoid CSF (~0.92) SAS_T    = 2.8 %   explicit bright ring
  → GM cortex      (~0.64)   CORTEX_T = 9.6 % (≈ 13 px) — gyral outer boundary
  → white matter   (~0.40)   interior
  → lateral ventricles + 3rd ventricle (CSF ~0.92)
  → basal ganglia / thalami (~0.55)
  → corpus callosum (slight WM darkening)
```

Gyral folding: `R_cortex_outer(θ) = R_SAS_inner + G(θ) · GYRAL_AMP`
where `G(θ)` is a random angular sinusoidal field (5–13 harmonics).

MRI field effects: B1+ centre brightening, receive-coil roll-off ramp,
Rician-like noise.

---

### Hidden Tier — Adversarial Brain T2w Phantoms (20 samples)

Seeds: `8000–8019`.  Severe mismatch.

**Recipes:**

| Recipe                  | Mix  | Challenge                                              |
|-------------------------|------|--------------------------------------------------------|
| `brain_t2_wm_lesions`   | 35%  | 3–8 focal WM hyperintensities (MS-like plaques)        |
| `brain_t2_atrophy`      | 30%  | Cortical atrophy — widened sulci, enlarged ventricles  |
| `brain_t2_high_contrast`| 20%  | Extreme CSF / WM intensity ratio (very low WM signal)  |
| `brain_t2_fine_gyri`    | 15%  | Very fine cortical folding (high-frequency gyri)       |

---

## HDF5 File Format

```python
import h5py, json, numpy as np

with h5py.File("mri_challenge_dev.h5", "r") as f:
    for sample_key in sorted(f.keys()):
        grp = f[sample_key]

        # Core arrays
        x_true     = grp["x_true"][:]      # (320, 320)     float32   — GT magnitude image [0, 1]
        y_kspace   = grp["y_kspace"][:]    # (15, 320, 320) complex64 — undersampled k-space
        mask       = grp["mask"][:]        # (320,)          uint8    — 1D ky undersampling mask
        coil_maps  = grp["coil_maps"][:]   # (15, 320, 320) complex64 — nominal coil sensitivity maps
        b0_map     = grp["B0_map"][:]      # (320, 320)     float32   — B0 field map (oracle)
        warp_field = grp["warp_field"][:]  # (2, 320, 320)  float32   — (dy, dx) warp in pixels

        # Metadata
        metadata    = json.loads(grp.attrs["metadata"])
        spec_ranges = json.loads(grp.attrs["spec_ranges"])   # per-tier parameter bounds
        true_spec   = json.loads(grp.attrs["true_spec"])     # per-sample ground-truth params
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
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=(-2,-1)), axes=(-2,-1)),
        axes=(-2,-1))

rss = np.sqrt(np.sum(np.abs(np.stack([ifft2c(y_kspace[c]) for c in range(15)])) ** 2, axis=0))
rss /= rss.max()   # normalise to [0, 1]
```

## Coil Sensitivity Estimation

Nominal coil maps are estimated from the ACS (auto-calibration signal) region
using low-pass filtered coil images divided by RSS:

```
S_c(r) = LPF(coil_c)(r) / RSS_LPF(r)
```

This provides the reconstructor's "best available" sensitivity estimate,
consistent with the mismatch scenario (true maps differ by the
`coil_sensitivity_frac` perturbation).

## Baseline Performance (mismatch, no correction)

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
