# Modify Plan: expansion

## Current Assignment
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** expansion (direct variant override)
- **Algorithms:** 9-algorithm `_VARIANT_OVERRIDES["expansion"]` (see algorithm catalog)

## Assessment

The algorithm assignment is appropriate for expansion microscopy (ExM). ExM physically
expands the specimen by ~4× using a swellable polymer gel, then images the expanded
sample with conventional diffraction-limited fluorescence microscopy. The reconstruction
task involves deconvolving the optical PSF, correcting gel distortion, and recovering
the super-resolution structure.

- **Deconv-Exp / RL-ExM** are classical deconvolution baselines appropriate for PSF removal.
- **TV-ExM** adds total variation regularization for edge-preserving reconstruction.
- **DnCNN-ExM** is a deep learning CNN baseline adapted for ExM noise characteristics.
- **DeepInterp-ExM** applies deep interpolation, a self-supervised method validated on fluorescence microscopy.
- **TransExM** and **SwinExM** are transformer-based methods capturing non-uniform PSF and distortion.
- **PhysExM** incorporates physics priors (PSF model, gel distortion) into a physics-informed network.
- **DiffExM** is a diffusion-model method for joint deconvolution and distortion correction.

## Change Log

### 2026-03-09 — Phantom generator + algorithm overrides added

**Changes made:**
1. `benchmarks/datasets/downloaders.py`: Added `generate_expansion_phantom()` — 64×64 float32 neuronal dendrite phantom with spine protrusions, dendritic shaft, synaptic vesicle clusters; forward model applies 4× expansion factor, Gaussian PSF (σ~0.38 px at expanded scale), smooth gel deformation field (2–5 nm displacement), Poisson noise. Registered in `_generated_converters` and `converter_map`.
2. `benchmarks/datasets/registry.py`: Added `expansion_generated` DatasetEntry.
3. `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Added `_VARIANT_OVERRIDES["expansion"]` (9 algorithms) and `CATEGORY_REAL_SCORES["expansion"]` (9 benchmark results with PSNR/SSIM).
4. `platform/scripts/generate_challenge_datasets.py`: Added `"expansion": "identity"` to `_VARIANT_TO_RUNNER`; added `generate_expansion_phantom` to both import lists and generator maps.
5. GCS datasets generated and uploaded: all 3 tiers (public, dev, hidden) at `gs://pwm-benchmark-datasets/challenge-data/v1.0/`.

**Status:** PASS — check.md updated 2026-03-09

### 2026-03-06 — Initial check (previous state)

Full pool (13 algorithms via category fallback): Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA, PnP-DnCNN, CARE, U-Net, ResUNet, Restormer, DeconvFormer, Restormer+, DiffDeconv, ScoreMicro.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No further changes needed. Expansion modality is fully processed with dedicated phantom generator, 9-algorithm override, and GCS datasets for all 3 tiers.
