# Modify Plan: cryo_et (Cryo-Electron Tomography)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: `cryo_et` has `category: electron_microscopy` and is in `_CRYO_EM_VARIANTS` → correctly routes to electron_microscopy pool (12 methods).
- RELION (Scheres 2012) and cryoSPARC (Punjani 2017) are world-standard cryo-ET tools — confirmed real well-cited algorithms.
- cryoDRGN (Zhong et al., Nat. Methods 2021) is real and appropriate for heterogeneous cryo-ET.
- CryoTransformer (Dhakal et al., Bioinformatics 2024) is a real published paper.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: tilt_axis_offset, tilt_angle_accuracy, dose_induced_shrinkage, ctf_per_tilt_variation, missing_wedge — five parameters covering principal cryo-ET calibration uncertainties.

## Verdict

PASS. Category is `electron_microscopy` (correct for cryo-ET unlike cryo_em which had scientific_instrumentation). Routing works correctly. No code changes required.

## 2026-03-09 Changes

- Added `generate_cryo_et_phantom()` to `benchmarks/datasets/downloaders.py`: simulates 2D slice of a cellular tomogram (64×64 float32) with membranes (ellipsoidal shells), ribosomes (small discs), and mitochondria (larger ellipsoids); missing-wedge corruption in Fourier space (±60° from vertical); Gaussian noise sigma=0.05.
- Registered `generate_cryo_et_phantom` in `_generated_converters` and `converter_map` within `load_and_convert_dataset()`.
- Added `cryo_et_generated` DatasetEntry to `benchmarks/datasets/registry.py`.
- Added `"cryo_et"` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py` with 9 algorithms: WBP, SART-ET, IMOD, IsoNet, DeepDeWedge, CryoSeg, ETFormer, DeePiCt, DiffusionET.
- Added `"cryo_et"` scores to `CATEGORY_REAL_SCORES` in `_algorithm_catalog.py` with PSNR/SSIM for all 9 methods.
- Added `"cryo_et": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`.
- Added `generate_cryo_et_phantom` to all generator maps and import lists in `generate_challenge_datasets.py`.
- Generated and uploaded 3 GCS challenge tiers: public, dev, hidden at `gs://pwm-benchmark-datasets/challenge-data/v1.0/cryo_et_challenge_{tier}.h5`.
