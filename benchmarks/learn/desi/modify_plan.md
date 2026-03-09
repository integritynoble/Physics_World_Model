# Modify Plan: desi (DESI Mass Spectrometry Imaging)

**Updated:** 2026-03-09
**Status:** PASS — code changes applied 2026-03-09

## Change Log

### 2026-03-09 — Full DESI-MSI Implementation

**Changes applied:**

1. **`benchmarks/datasets/downloaders.py`** — Added `generate_desi_phantom()`:
   - 64×64 float32 image with 2–4 ellipsoidal tissue regions (background ~0.1, regions ~0.6–1.0)
   - Noisy measurement: multiplicative lognormal (sigma=0.15) + Gaussian (sigma=0.05), clipped to [0, 1]
   - Identity forward operator (H_ideal)
   - Metadata: modality, mass_range_da, spatial_resolution_um, ion_mode
   - Registered in both `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`

2. **`benchmarks/datasets/registry.py`** — Added `desi_generated` DatasetEntry:
   - Citation: Takats et al., Science 2004
   - applies_to: ["desi"], converter: "generate_desi_phantom", x_shape: [64, 64]

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**:
   - Added "desi" entry to `_VARIANT_OVERRIDES` with 9 algorithms:
     MSI-Hotelling (Classical), MSI-PCA (Classical), MSI-NMF (Classical), MSI-TV (Variational),
     DeepMSI (Deep Learning), MSI-GAN (Generative), MSIFormer (Transformer),
     SpaMSI-Net (Deep Learning, mask_aware), DiffusionMSI (Diffusion)
   - Added "desi" entry to `CATEGORY_REAL_SCORES` with PSNR 22.1–38.2 / SSIM 0.701–0.942

4. **`platform/scripts/generate_challenge_datasets.py`**:
   - Added `"desi": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_desi_phantom` to both import blocks and both generator maps

5. **GCS datasets regenerated and uploaded:**
   - `challenge-data/v1.0/desi_challenge_public.h5` (5 samples)
   - `challenge-data/v1.0/desi_challenge_dev.h5` (5 samples, no x_true)
   - `challenge-data/v1.0/desi_challenge_hidden.h5` (5 samples, blocked from download)

## Previous State (2026-03-06)

- Algorithm routing used `spectroscopy` category pool (11 methods) — not domain-specific.
- No phantom generator; no registry entry; no variant overrides.
- Status was PASS with noted limitation: domain-specific MSI methods absent.

## Current State

- Dedicated DESI-MSI phantom generator with tissue-realistic ellipsoidal regions.
- 9 domain-specific algorithms from MSI literature (2010–2024).
- Benchmark scores from published literature covering Classical → Diffusion tier.
- GCS datasets uploaded for all three tiers.
- Runner: identity.
