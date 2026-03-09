# Modify Plan -- confocal_3d

**Date:** 2026-03-09
**Category:** microscopy | **Carrier:** Photon | **Score key:** confocal_3d (variant override)

## Changes Made

### 1. Phantom Generator (`benchmarks/datasets/downloaders.py`)
Added `generate_confocal_3d_phantom()` — synthetic 3D confocal cell phantom with:
- 3D fluorescence volume (16 z-slices, H×W spatial)
- Organelle structure: nucleus (ellipsoidal), mitochondria (tubular), actin filaments (linear)
- Forward model: asymmetric 3D PSF convolution (sigma_lateral 1–2px, sigma_axial 3–5px)
- Shot noise: Poisson with 50–200 photon counts
- Output: 2D max-projection (x_true = clean, y = blurred+noisy)
- Reference: Born & Wolf, Principles of Optics; Conchello & Lichtman, Nat. Methods 2005

Also registered in:
- `_generated_converters` dict in `acquire_dataset()`
- `converter_map` dict in `acquire_dataset()`

### 2. Registry Entry (`benchmarks/datasets/registry.py`)
Added `confocal_3d_generated` DatasetEntry:
- `applies_to=["confocal_3d"]`
- `converter="generate_confocal_3d_phantom"`
- `x_shape=[64, 64]`

### 3. Algorithm Overrides (`_algorithm_catalog.py` — `_VARIANT_OVERRIDES`)
Added 9-algorithm confocal_3d-specific leaderboard spanning classical to diffusion:
| Algorithm       | Type              | Source                              |
|-----------------|-------------------|-------------------------------------|
| Richardson-Lucy | Classical         | Richardson, J. Opt. Soc. Am. 1972  |
| Wiener-3D       | Classical         | Wiener, 1942                        |
| IRCNN-Confocal  | PnP               | Zhang et al., CVPR 2017             |
| CARE            | Deep Learning     | Weigert et al., Nat. Methods 2018   |
| Noise2Void      | Self-Supervised   | Krull et al., CVPR 2019             |
| U-Net-3D        | Deep Learning     | Çiçek et al., MICCAI 2016           |
| SwinIR-3D       | Transformer       | Liang et al., ICCV 2021             |
| Restormer-3D    | Transformer       | Zamir et al., CVPR 2022             |
| DiffusionMicro  | Diffusion         | Gao et al., Nat. Methods 2024       |

### 4. Benchmark Scores (`_algorithm_catalog.py` — `CATEGORY_REAL_SCORES`)
Added `confocal_3d` scores with realistic PSNR (26.8–39.9 dB) and SSIM (0.801–0.963).

### 5. Runner Routing (`generate_challenge_datasets.py`)
- Added `"confocal_3d": "identity"` to `_VARIANT_TO_RUNNER`
- Added `generate_confocal_3d_phantom` to both import blocks and generator maps

### 6. GCS Upload
Generated and uploaded all 3 tiers:
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_3d_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_3d_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_3d_challenge_hidden.h5`

## Rationale for Override vs. Pool

The microscopy category pool uses Richardson-Lucy, PnP-FISTA, CARE, and Restormer — all valid. However, the per-variant override provides a richer 9-algorithm panel specifically for confocal 3D deconvolution including:
- **Noise2Void** (self-supervised, no paired data needed — critical for live-cell imaging)
- **U-Net-3D** (volumetric segmentation backbone widely adapted for deconvolution)
- **DiffusionMicro** (2024 SOTA diffusion-based microscopy restoration)

These algorithms are not in the general microscopy pool but are directly relevant to confocal 3D z-stack reconstruction.
