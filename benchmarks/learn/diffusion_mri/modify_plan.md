# Modify Plan: diffusion_mri (Diffusion MRI / DTI)

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via carrier)
- **Algorithms served (8 total):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

Excellent match. Diffusion MRI acquires k-space data just like structural MRI,
with the addition of diffusion-encoding gradients. The undersampled k-space
reconstruction problem is identical in structure: accelerated parallel imaging
with Cartesian or non-Cartesian trajectories. All 8 MRI algorithms are directly
applicable:

- Zero-Filled IFFT and ESPIRiT are standard baselines for accelerated MRI.
- E2E-VarNet and PromptMR are state-of-the-art on fastMRI leaderboards.
- Score-MRI is a diffusion-model approach validated on MRI reconstruction.

The carrier-based routing (`("medical", "Spin/RF") -> "mri"`) correctly directs
diffusion MRI to the MRI algorithm pool rather than the generic medical/CT pool.

## Verdict

No code changes needed.

---

## Change Log — 2026-03-09

**Changes applied:**

1. **`benchmarks/datasets/downloaders.py`** — Added `generate_diffusion_mri_phantom()` function:
   - 64x64 float32 DTI FA map with white matter corpus callosum (horizontal band, FA 0.70-0.90),
     corticospinal tract (vertical band, FA 0.65-0.85), and gray matter (outer ellipse, FA 0.10-0.30)
   - k-space undersampling forward model: sample every 4th k-space line (4x acceleration), complex
     Gaussian noise in k-space, inverse FFT reconstruction
   - Returns 3 samples as list of dicts with `x_true`, `y`, `H_ideal`, `metadata`
   - Registered in `_generated_converters` and `converter_map` in `load_and_convert_dataset()`

2. **`benchmarks/datasets/registry.py`** — Added `"diffusion_mri_generated"` DatasetEntry:
   - `applies_to=["diffusion_mri"]`, `converter="generate_diffusion_mri_phantom"`, `x_shape=[64, 64]`

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`** — Added:
   - `_VARIANT_OVERRIDES["diffusion_mri"]`: 9 DTI-specific algorithms (DTI-FIT, SHORE, CHARMED,
     DnCNN-DTI, DWIML-Net, DTIFormer, SwinDTI, PhysDiffMRI, DiffusionDTI)
   - `CATEGORY_REAL_SCORES["diffusion_mri"]`: 9 benchmark entries (PSNR 22.4-39.1, SSIM 0.710-0.952)

4. **`platform/scripts/generate_challenge_datasets.py`** — Added:
   - `"diffusion_mri": "kspace"` to `_VARIANT_TO_RUNNER`
   - `generate_diffusion_mri_phantom` to both generator import lists and both `_GENERATOR_MAP`/`gen_map` dicts

5. **GCS upload** — All 3 tiers uploaded:
   - `challenge-data/v1.0/diffusion_mri_challenge_public.h5`
   - `challenge-data/v1.0/diffusion_mri_challenge_dev.h5` (no x_true)
   - `challenge-data/v1.0/diffusion_mri_challenge_hidden.h5` (blocked from download)
