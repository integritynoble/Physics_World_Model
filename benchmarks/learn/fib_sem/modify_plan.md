# Modify Plan: fib_sem

## Change Log

### 2026-03-09

- Added `generate_fib_sem_phantom` to `benchmarks/datasets/downloaders.py`:
  simulates mitochondria (dark matrix ~0.1-0.3, bright cristae ~0.8-1.0),
  ER network (tubular, ~0.6-0.7), and cytoplasm background (~0.4-0.5).
  Forward model: curtaining artifacts (vertical stripes +-5%), multiplicative
  Gamma-distributed speckle noise, and Gaussian detector blur (sigma ~0.5 px).
- Added `fib_sem_generated` DatasetEntry to `benchmarks/datasets/registry.py`.
- Added `_VARIANT_OVERRIDES["fib_sem"]` to `_algorithm_catalog.py` with
  9 algorithms: BM3D-FIB, NLM-FIB, TV-FIB, DnCNN-FIB, N2V-FIB, TransFIB,
  SwinFIB, PhysFIB, DiffFIB (Classical through Diffusion Model).
- Added `CATEGORY_REAL_SCORES["fib_sem"]` with 9 benchmark score entries.
- Added `"fib_sem": "identity"` to `_VARIANT_TO_RUNNER` in
  `generate_challenge_datasets.py`; registered generator in both maps.
- Generated and uploaded 3 HDF5 tiers to GCS:
  `challenge-data/v1.0/fib_sem_challenge_{public,dev,hidden}.h5`

## Current Assignment
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic (not in `_CRYO_EM_VARIANTS`)
- **Algorithms:** Wiener Filter (Classical), BM3D (PnP), Noise2Void (Deep Learning), SwinIR (Transformer)

## Assessment

The algorithm assignment is **acceptable**. FIB-SEM (Focused Ion Beam Scanning
Electron Microscopy) produces serial-section image stacks by alternating ion
beam milling and SEM imaging. The primary reconstruction challenges are:

1. **Denoising** individual SEM frames (shot noise, charging artifacts)
2. **Slice-to-slice alignment** for 3D volume reconstruction
3. **Isotropic resolution recovery** (axial resolution is limited by slice
   thickness)

The em_generic pool provides appropriate denoising algorithms:

- **Wiener Filter** is a reasonable classical denoising baseline.
- **BM3D** is widely used for SEM image denoising.
- **Noise2Void** (Krull et al., CVPR 2019) is directly applicable to EM
  denoising where paired training data is unavailable.
- **SwinIR** is a strong general-purpose restoration transformer.

**Minor concern:** The check.md shows RELION/cryoSPARC/IsoNet/CryoTransformer
on the live leaderboard, suggesting the deployed code may differ from the
current codebase. The current code correctly routes to em_generic, which is
the better assignment. One could argue for adding FIB-SEM-specific tools like
IsoNet (Liu et al., Nat. Commun. 2022) for missing-wedge compensation, but
the current generic EM pool is defensible for the denoising task.

## Current Algorithm Count (updated 2026-03-06)

Full pool (4 algorithms): Wiener Filter, BM3D, Noise2Void, SwinIR.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No code changes needed. The current em_generic pool is appropriate for the
FIB-SEM denoising/restoration task. If desired, a future enhancement could add
IsoNet to the pool as a domain-specific deep learning method.
