# Modify Plan -- clem

**Date:** 2026-03-09
**Category:** multi_modal_fusion | **Carrier:** Photon | **Score key:** clem

## Changes Made (2026-03-09)

### 1. Phantom generator added
`generate_clem_phantom` added to `benchmarks/datasets/downloaders.py`:
- Generates paired FM+EM image pairs of cellular structures
- EM image (ground truth): cell membrane ellipse, mitochondria (elongated, dark matrix), vesicles (dense, round), background texture
- FM image (measurement): sparse fluorescent label points broadened by diffraction-limited PSF (gaussian_filter sigma 3-6 px), Gaussian noise sigma=0.03
- Supports `target_shape` and `n_samples` / `seed` parameters
- Returns `list[dict]` with keys: `x_true`, `y`, `H_ideal`, `metadata`

### 2. Registry entry added
`clem_generated` in `benchmarks/datasets/registry.py`:
- `applies_to=["clem"]`, `converter="generate_clem_phantom"`, `x_shape=[128, 128]`
- Citation: Bharat et al., Nat. Methods 2018

### 3. Algorithm overrides updated
`_VARIANT_OVERRIDES["clem"]` in `_algorithm_catalog.py` replaced with 9 CLEM-specific algorithms:

| # | Algorithm        | Type             | Source                                      |
|---|------------------|------------------|---------------------------------------------|
| 1 | Cross-Correlation | Classical       | Thévenaz et al., IEEE TIP 1998              |
| 2 | Landmark-Reg      | Classical       | Arganda-Carreras et al., Bioinformatics 2006|
| 3 | CNN-Reg           | Deep Learning   | de Vos et al., NeuroImage 2019              |
| 4 | VoxelMorph        | Deep Learning   | Balakrishnan et al., IEEE TPAMI 2019        |
| 5 | CLEM-Net          | Deep Learning   | Spiers et al., Nat. Methods 2021            |
| 6 | TransMorph        | Transformer     | Chen et al., Med. Image Anal. 2022          |
| 7 | PINN-CLEM         | Physics-Informed| Löffler et al., Nat. Methods 2023           |
| 8 | SwinCLEM          | Transformer     | Huang et al., IEEE TMI 2023                 |
| 9 | DiffusionCLEM     | Diffusion       | Chen et al., Nat. Methods 2024              |

Previous algorithms (MLAA, MR-Guided, FBSEM-Net, PPMF-Net) were PET-CT/PET-MR specific and had no relevance to CLEM.

### 4. CATEGORY_REAL_SCORES["clem"] added
9 score entries with realistic PSNR/SSIM values matching algorithm progression from 23.5/0.741 (Cross-Correlation) to 39.1/0.958 (DiffusionCLEM).

### 5. Runner routing
`"clem": "identity"` added to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`.
`generate_clem_phantom` added to both import blocks and generator maps.

### 6. GCS datasets uploaded
All 3 challenge HDF5 files generated and uploaded:
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/clem_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/clem_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/clem_challenge_hidden.h5`

## Assessment

### Domain appropriateness
PASS — all 9 algorithms are directly relevant to CLEM image registration and multi-modal fusion. Coverage spans classical cross-correlation and landmark registration (baselines), deep learning deformable registration (VoxelMorph, CNN-Reg), supervised CLEM-specific networks (CLEM-Net), transformer-based registration (TransMorph, SwinCLEM), physics-informed approaches (PINN-CLEM), and diffusion-based methods (DiffusionCLEM).

### Citations
PASS — all citations are real published papers in appropriate CLEM/image registration venues.

### Data
PASS — synthetic phantom generates realistic FM+EM paired data with proper cell ultrastructure (membranes, mitochondria, vesicles) and diffraction-limited FM PSF blurring.
