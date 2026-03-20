# PWM Blind Reconstruction Challenge — Getting Started Guide

This guide walks you through **joining the competition** and **contributing datasets** to the Physics World Model (PWM) benchmark platform.

All referenced files are in this directory:

```
platform/pwm_platform/static/examples/
```

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Quick Start (5 minutes)](#2-quick-start)
3. [Step-by-Step: Joining the Competition](#3-step-by-step-joining-the-competition)
4. [Algorithms & Methods](#4-algorithms--methods)
5. [Dataset Format Reference](#5-dataset-format-reference)
6. [Contributing a Dataset](#6-contributing-a-dataset)
7. [File Inventory](#7-file-inventory)

---

## 1. Competition Overview

The **Blind Reconstruction Challenge** tests your algorithm's ability to reconstruct signals from measurements with **unknown mismatch parameters**. In real-world imaging systems (CT scanners, MRI machines, etc.), the actual system parameters differ from the assumed ideal model. Your task:

1. **Reconstruct** the original signal `x` from corrupted measurements `y`
2. **Estimate** the true mismatch parameters (the "spec") that caused the corruption

**Scoring formula:**

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖ / ‖y‖)
```

- **PSNR_norm**: Peak Signal-to-Noise Ratio (normalized to 0–1)
- **SSIM**: Structural Similarity Index
- **Consistency**: How well your reconstruction matches the measurements under the estimated model

---

## 2. Quick Start

### Prerequisites

```bash
pip install numpy h5py scipy
```

### Run the CT baseline in 3 commands

```bash
# 1. Go to the examples directory
cd platform/pwm_platform/static/examples/

# 2. Run the CT baseline on the dev example
python3 ct_baseline_algorithm.py ct_example_dev.h5 my_ct_submission.h5

# 3. Inspect the output
python3 -c "
import h5py, json
f = h5py.File('my_ct_submission.h5', 'r')
for key in f.keys():
    print(key, '-> x_hat shape:', f[key]['x_hat'].shape)
    print('   corrected_spec:', json.loads(f[key].attrs['corrected_spec']))
f.close()
"
```

### Run the MRI baseline

```bash
python3 mri_baseline_algorithm.py mri_example_dev.h5 my_mri_submission.h5
```

---

## 3. Step-by-Step: Joining the Competition

The competition has **3 tiers** that you progress through in order:

### Step 1: Public Tier (Free) — Develop & Debug

**What you get:** Measurements + ground truth + true mismatch parameters.

Use the public tier to develop your algorithm with full visibility:

```python
import h5py, json

f = h5py.File("ct_example_public.h5", "r")

for sample_key in sorted(f.keys()):
    grp = f[sample_key]

    # Available data
    y        = grp["y"][:]          # measurements (sinogram: 32×46)
    H_ideal  = grp["H_ideal"][:]    # ideal forward model (angles: 32,)
    x_true   = grp["x_true"][:]     # ground truth image (32×32)

    # Metadata (stored as HDF5 attributes)
    spec_ranges = json.loads(grp.attrs["spec_ranges"])   # parameter ranges
    true_spec   = json.loads(grp.attrs["true_spec"])      # actual mismatch values
    metadata    = json.loads(grp.attrs["metadata"])        # scene info

    print(f"{sample_key}:")
    print(f"  y shape:      {y.shape}")
    print(f"  x_true shape: {x_true.shape}")
    print(f"  true_spec:    {true_spec}")
    print(f"  spec_ranges:  {[p['name'] for p in spec_ranges]}")

f.close()
```

**Output:**
```
sample_00:
  y shape:      (32, 46)
  x_true shape: (32, 32)
  true_spec:    {'center_offset': 0.7, 'angle_error': -1.2, 'beam_hardening': 0.15, 'detector_tilt': 0.3}
  spec_ranges:  ['center_offset', 'angle_error', 'beam_hardening', 'detector_tilt']
```

**What to do:** Run your algorithm, compare `x_hat` against `x_true`, compute PSNR/SSIM, and report your score on the platform.

### Step 2: Dev Tier (Free) — Blind Evaluation

**What you get:** Measurements + ideal model only. No ground truth.

```python
f = h5py.File("ct_example_dev.h5", "r")
grp = f["sample_00"]

y       = grp["y"][:]          # measurements
H_ideal = grp["H_ideal"][:]    # ideal forward model
spec_ranges = json.loads(grp.attrs["spec_ranges"])  # parameter ranges

# NO x_true — you must reconstruct blindly!
# NO true_spec — you must estimate the parameters!
```

**What to do:**
1. Run your algorithm to produce `x_hat` and `corrected_spec`
2. Save as a submission HDF5 file (see format below)
3. Upload to the platform — PWM will score it and return your results

### Step 3: Hidden Tier (10 Credits) — Final Evaluation

**What you submit:** Your algorithm as a Python script (`.py`) or archive (`.zip`/`.tar.gz`).

PWM runs your code server-side against hidden data with unknown mismatch parameters. This is the final score that determines your **leaderboard ranking**.

### Submission File Format

Your submission HDF5 must contain one group per sample with:

```python
import h5py, json

with h5py.File("my_submission.h5", "w") as f:
    # File-level attributes
    f.attrs["variant"] = "ct"           # or "mri"
    f.attrs["tier"] = "dev"
    f.attrs["submission_type"] = "reconstruction"

    for i in range(n_samples):
        grp = f.create_group(f"sample_{i:02d}")

        # Your reconstructed signal (same spatial shape as x_true)
        grp.create_dataset("x_hat", data=x_hat_array, compression="gzip")

        # Your estimated mismatch parameters
        grp.attrs["corrected_spec"] = json.dumps({
            "center_offset": 0.5,
            "angle_error": -1.0,
            "beam_hardening": 0.2,
            "detector_tilt": 0.1,
        })
```

See `ct_example_submission.h5` and `mri_example_submission.h5` for working examples.

---

## 4. Algorithms & Methods

### 4.1 CT Baseline: Filtered Back-Projection (FBP)

**File:** `ct_baseline_algorithm.py`

**Method overview:**

1. **Ramp filtering** — Apply a ramp (Ram-Lak) filter in the frequency domain to each sinogram row to compensate for the oversampling of low spatial frequencies:
   ```python
   freqs = np.fft.fftfreq(n_detectors)
   ramp = np.abs(freqs)
   filtered_proj = np.real(np.fft.ifft(np.fft.fft(projection) * ramp))
   ```

2. **Back-projection** — Smear each filtered projection across the image grid at its corresponding angle, then sum all contributions:
   ```python
   for i, angle in enumerate(angles):
       # Replicate filtered projection into a 2D image
       # Rotate by the projection angle
       rotated = scipy.ndimage.rotate(proj_img, angle, reshape=False)
       recon += rotated
   recon *= np.pi / (2 * n_angles)
   ```

3. **Mismatch estimation:**
   - **Center offset:** Compare the 0° and 180° projections. A misaligned center causes asymmetry between `proj(0°)` and `flip(proj(180°))`. Grid-search small pixel shifts to minimize the discrepancy.
   - **Angle error:** After initial reconstruction, re-project `x_hat` at candidate angle offsets and compare residuals with the measured sinogram. Pick the offset that minimizes `‖y - Hx̂‖²`.

4. **Refinement** — Re-run FBP with corrected angles for the final reconstruction.

**CT mismatch parameters:**

| Parameter | Range | Unit | Physical meaning |
|-----------|-------|------|-----------------|
| `center_offset` | [-2, 2] | px | Detector center misalignment |
| `angle_error` | [-3, 3] | deg | Global rotation offset of gantry |
| `beam_hardening` | [0, 0.5] | a.u. | Polychromatic beam nonlinearity |
| `detector_tilt` | [-1, 1] | deg | Detector array tilt |

### 4.2 MRI Baseline: Zero-Filled IFFT + Iterative Soft-Thresholding (ISTA)

**File:** `mri_baseline_algorithm.py`

**Method overview:**

1. **Zero-filled IFFT** — Convert log-magnitude k-space back to linear scale, apply zero-filled inverse FFT:
   ```python
   kspace_mag = np.expm1(y_kspace)  # undo log(1+|k|)
   x0 = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_mag * mask)))
   ```

2. **Iterative soft-thresholding (ISTA)** — Alternate between data consistency in k-space and sparsity promotion in image domain:
   ```python
   for iteration in range(30):
       # Forward: go to k-space
       kx = np.fft.fftshift(np.fft.fft2(x))
       # Data consistency: replace sampled locations with measured data
       kx = kx * (1 - mask) + kspace_measured * mask
       # Back to image domain
       x = np.abs(np.fft.ifft2(np.fft.ifftshift(kx)))
       # Soft-threshold for sparsity
       x = sign(x) * max(|x| - lambda, 0)
   ```

3. **Mismatch estimation:**
   - **B0 inhomogeneity:** Compute row/column intensity profiles of the reconstruction. Spatial variation in the magnitude image indicates B0 field non-uniformity.
   - **Coil sensitivity:** Apply a large-kernel Gaussian filter to get the low-frequency intensity envelope. Non-uniformity of this envelope estimates coil sensitivity variation.

**MRI mismatch parameters:**

| Parameter | Range | Unit | Physical meaning |
|-----------|-------|------|-----------------|
| `B0_inhomog` | [0, 0.5] | a.u. | Main field inhomogeneity |
| `gradient_nonlin` | [0, 0.3] | a.u. | Gradient coil nonlinearity |
| `coil_sensitivity` | [0, 0.8] | a.u. | Receive coil non-uniformity |
| `k_trajectory` | [-1, 1] | px | k-space trajectory deviation |

### 4.3 Improving Beyond the Baseline

These baselines are intentionally simple. Here are ideas for better performance:

- **Learned reconstruction:** Unrolled optimization networks (e.g., ADMM-Net, ISTA-Net) that learn the proximal operator from training data
- **Dictionary learning:** Learn a sparsifying dictionary from the public tier data instead of using fixed wavelets
- **Joint estimation:** Alternate between reconstructing `x` and estimating spec parameters in a single optimization loop
- **Deep image prior:** Use an untrained neural network as a regularizer — no training data needed
- **Physics-informed networks:** Embed the forward model `y = H(spec) @ x + noise` directly into the network architecture

---

## 5. Dataset Format Reference

### Challenge HDF5 Structure

All challenge datasets follow this schema:

```
file.h5
├── [file attributes]
│   ├── variant     = "ct"          # variant key
│   ├── tier        = "public"      # "public" | "dev" | "hidden"
│   ├── version     = "1.0"         # schema version
│   └── runner_type = "radon"       # forward model type
│
├── sample_00/
│   ├── y           [dataset]       # measurements array
│   ├── H_ideal     [dataset]       # ideal forward model operator
│   ├── x_true      [dataset]       # ground truth (public + hidden only)
│   ├── spec_ranges [attribute]     # JSON: parameter ranges
│   ├── metadata    [attribute]     # JSON: scene info
│   └── true_spec   [attribute]     # JSON: true params (public + hidden only)
│
├── sample_01/
│   └── ...
└── sample_02/
    └── ...
```

### What each field contains

| Field | Type | Shape (CT) | Shape (MRI) | Description |
|-------|------|-----------|-------------|-------------|
| `y` | dataset | (n_angles, n_det) | (H, W) | Corrupted measurements |
| `H_ideal` | dataset | (n_angles,) | (H, W) | Ideal operator: angles / mask |
| `x_true` | dataset | (H, W) | (H, W) | Ground truth signal |
| `spec_ranges` | attr (JSON) | — | — | `[{name, min, max, unit}, ...]` |
| `metadata` | attr (JSON) | — | — | `{scene, shape, noise_model}` |
| `true_spec` | attr (JSON) | — | — | `{param_name: value, ...}` |

### Tier visibility

| Field | Public | Dev | Hidden |
|-------|--------|-----|--------|
| `y` | Yes | Yes | Yes |
| `H_ideal` | Yes | Yes | Yes |
| `spec_ranges` | Yes | Yes | Yes |
| `metadata` | Yes | Yes | Yes |
| `x_true` | Yes | **No** | Yes (server only) |
| `true_spec` | Yes | **No** | Yes (server only) |

### Spec ranges format

```json
[
    {"name": "center_offset", "min": -2.0, "max": 2.0, "unit": "px"},
    {"name": "angle_error",   "min": -3.0, "max": 3.0, "unit": "deg"},
    {"name": "beam_hardening","min":  0.0, "max": 0.5, "unit": "a.u."},
    {"name": "detector_tilt", "min": -1.0, "max": 1.0, "unit": "deg"}
]
```

Each parameter has a known range — the true value lies somewhere within `[min, max]`. Your algorithm must estimate where.

### Reading example data in Python

```python
import h5py
import json
import numpy as np

with h5py.File("ct_example_public.h5", "r") as f:
    print("Variant:", f.attrs["variant"])
    print("Tier:", f.attrs["tier"])

    for sample_key in sorted(f.keys()):
        grp = f[sample_key]

        y       = grp["y"][:]
        H_ideal = grp["H_ideal"][:]

        spec_ranges = json.loads(grp.attrs["spec_ranges"])
        metadata    = json.loads(grp.attrs["metadata"])

        # Public/hidden only:
        if "x_true" in grp:
            x_true = grp["x_true"][:]
            true_spec = json.loads(grp.attrs["true_spec"])
```

---

## 6. Contributing a Dataset

You can contribute new challenge datasets to grow the benchmark. Contributed data is reviewed by the PWM team before being added.

### What to prepare

Your contributed HDF5 file should follow the same schema described in Section 5. At minimum:

**For Public tier contributions:**
- `y` — your measurements
- `H_ideal` — your ideal forward model
- `x_true` — your ground truth signal
- `spec_ranges` — the mismatch parameter ranges (as JSON attribute)
- `true_spec` — the actual mismatch values used (as JSON attribute)
- `metadata` — scene description (as JSON attribute)

**For Dev tier contributions:**
- Same as Public, but `x_true` and `true_spec` are held server-side (you still provide them, but they're hidden from contestants)

**For Hidden tier contributions:**
- Complete dataset with all fields — everything is kept private for blind evaluation

### Creating a contribution file

```python
import h5py
import json
import numpy as np

# Your data
signals = [...]        # list of ground truth images
measurements = [...]   # list of measurement arrays
forward_models = [...] # list of ideal forward model components

spec_ranges = [
    {"name": "my_param_1", "min": 0.0, "max": 1.0, "unit": "a.u."},
    {"name": "my_param_2", "min": -5.0, "max": 5.0, "unit": "deg"},
]
true_spec = {"my_param_1": 0.42, "my_param_2": -1.7}

with h5py.File("my_contribution_public.h5", "w") as f:
    # File-level metadata
    f.attrs["variant"] = "my_modality"
    f.attrs["tier"] = "public"
    f.attrs["version"] = "1.0"
    f.attrs["runner_type"] = "psf"  # or radon, kspace, ctf, mask, tip

    for i, (x, y, H) in enumerate(zip(signals, measurements, forward_models)):
        grp = f.create_group(f"sample_{i:02d}")
        grp.create_dataset("y", data=y, compression="gzip")
        grp.create_dataset("H_ideal", data=H, compression="gzip")
        grp.create_dataset("x_true", data=x, compression="gzip")
        grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
        grp.attrs["true_spec"] = json.dumps(true_spec)
        grp.attrs["metadata"] = json.dumps({
            "scene": f"scene_{i}",
            "shape": list(x.shape),
            "noise_model": "gaussian",
        })
```

### Submission guidelines

- **Format:** HDF5 (`.h5` / `.hdf5`) or NumPy archives (`.npy` / `.npz`)
- **Max size:** 50 MB per file (contact the team for larger datasets)
- **Include:** A clear description of the imaging setup, number of samples, and noise model
- **Optional:** Paper URL and code repository URL
- **Review:** All contributions are reviewed by the PWM team — you'll receive feedback

### Supported forward model types

| Runner Type | Physical Systems | H_ideal Format |
|-------------|-----------------|----------------|
| `radon` | CT, particle imaging | Projection angles array |
| `kspace` | MRI, remote sensing | k-space sampling mask |
| `psf` | Microscopy, optics, astronomy | PSF kernel |
| `ctf` | Electron microscopy | CTF parameter array |
| `mask` | Compressive sensing, ultrafast | Binary coded aperture |
| `tip` | Scanning probe microscopy | Tip convolution kernel |

### Noise models

| Model | Use for | Parameters |
|-------|---------|------------|
| `gaussian` | Most modalities | `sigma` |
| `poisson` | Photon-limited (astronomy, EM) | `peak_counts` |
| `poisson_gaussian` | Medical imaging (CT, CASSI) | `poisson_alpha`, `gaussian_sigma` |
| `speckle` | SAR, ultrasound | `n_looks` |

---

## 7. File Inventory

All files in `platform/pwm_platform/static/examples/`:

### Algorithm Scripts

| File | Size | Description |
|------|------|-------------|
| `ct_baseline_algorithm.py` | 7.2 KB | CT baseline — FBP + mismatch estimation |
| `mri_baseline_algorithm.py` | 6.3 KB | MRI baseline — IFFT + ISTA + mismatch estimation |

**Usage:**
```bash
python3 ct_baseline_algorithm.py <challenge.h5> <output_submission.h5>
python3 mri_baseline_algorithm.py <challenge.h5> <output_submission.h5>
```

### Example Challenge Data (32x32, 3 samples each)

| File | Size | Tier | Ground Truth? | Description |
|------|------|------|---------------|-------------|
| `ct_example_public.h5` | 88 KB | Public | Yes | CT with full data for development |
| `ct_example_dev.h5` | 60 KB | Dev | No | CT blind evaluation format |
| `ct_example_hidden.h5` | 88 KB | Hidden | Yes | CT server-side format reference |
| `mri_example_public.h5` | 79 KB | Public | Yes | MRI with full data for development |
| `mri_example_dev.h5` | 52 KB | Dev | No | MRI blind evaluation format |
| `mri_example_hidden.h5` | 79 KB | Hidden | Yes | MRI server-side format reference |

### Example Submissions

| File | Size | Description |
|------|------|-------------|
| `ct_example_submission.h5` | 40 KB | What a CT dev-tier submission looks like |
| `mri_example_submission.h5` | 40 KB | What an MRI dev-tier submission looks like |

### Generator Script

| File | Location | Description |
|------|----------|-------------|
| `generate_example_data.py` | `platform/scripts/` | Regenerate all example HDF5 files |

To regenerate:
```bash
cd platform
python3 scripts/generate_example_data.py
```

---

## Questions?

- **Platform:** [https://pwm.platformai.org/benchmark](https://pwm.platformai.org/benchmark)
- **Contact:** platformaigpt@gmail.com
