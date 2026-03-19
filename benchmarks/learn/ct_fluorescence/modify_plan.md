# Modify Plan: ct_fluorescence

## Current State (2026-03-09 — COMPLETE)

- **Category:** multi_modal_fusion (score routing alias retained)
- **Carrier:** X-ray
- **Routing:** `_VARIANT_OVERRIDES["ct_fluorescence"]` — 9 XRF-CT-specific algorithms
- **Score key:** `ct_fluorescence` (direct entry in `CATEGORY_REAL_SCORES`)
- **Runner:** `identity` (Poisson noise + Compton scatter handled in phantom generator)
- **Algorithms served:**
  1. FBP-XRF (Classical) — Boisseau & Grodzins, Hyperfine Int. 1987
  2. MLEM-XRF (Classical) — Jaszczak et al., IEEE TNS 1981 (XRF adapt.)
  3. TV-XRFCT (Variational) — Larsson et al., Phys. Med. Biol. 2020
  4. DnCNN-XRF (Deep Learning) — Zhang et al., IEEE TIP 2017 (XRF adapt.)
  5. U-Net-XRF (Deep Learning) — Ronneberger et al., MICCAI 2015 (XRF adapt.)
  6. PnP-XRF (PnP) — Chan et al., IEEE TIP 2016 (XRF adapt.)
  7. SwinXRF (Transformer) — Liu et al., ICCV 2021 (XRF adapt.)
  8. PhysXRF-Net (Physics-Informed) — Raissi et al., J. Comput. Phys. 2019 (XRF)
  9. DiffusionXRF (Diffusion) — Song et al., ICLR 2021 (XRF adapt.)

## Previous State (before 2026-03-09)

**Problem:** ct_fluorescence was incorrectly routed to the `multi_modal_fusion` pool, which contained PET/SPECT-specific algorithms (MLAA, MR-Guided, FBSEM-Net, PPMF-Net) — none of which are applicable to XRF-CT imaging.

**Previous algorithms (all wrong):**
- Born/Rytov + FBP (Classical) — Arridge & Schotland, Inverse Probl. 2009
- PnP-ADMM (Joint) — Venkatakrishnan et al., 2013
- FDot-Net (Deep Learning) — Gao et al., BOE 2021
- Cross-Modal Xformer (Transformer) — Multi-modal transformer, 2024

## Changes Made (2026-03-09)

### 1. `benchmarks/datasets/downloaders.py`
- Added `generate_ct_fluorescence_phantom()` after `generate_ct_phantom()`
- Phantom generates: 64×64 float32 fluorophore concentration map with 2-4 ellipsoidal clusters on low background
- Forward model: Poisson noise (lambda=50 counts) + Compton scatter background (~5 counts uniform), normalised to [0, 1]
- Returns 3 dicts per call, each with `x_true`, `y`, `H_ideal`, `metadata`
- Registered in both converter maps inside `load_and_convert_dataset()`:
  - `_generated_converters` map (for no-download path)
  - `converter_map` (for download path)

### 2. `benchmarks/datasets/registry.py`
- Added `"ct_fluorescence_generated"` DatasetEntry before closing `}` of `DATASET_REGISTRY`
- `source_type="generated"`, `applies_to=["ct_fluorescence"]`, `converter="generate_ct_fluorescence_phantom"`
- Citation: "Synthetic phantom based on Larsson et al., Phys. Med. Biol. 2020"

### 3. `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
- REPLACED `_VARIANT_OVERRIDES["ct_fluorescence"]` (4 incorrect entries → 9 XRF-CT-specific entries)
- ADDED `CATEGORY_REAL_SCORES["ct_fluorescence"]` with 9 entries (PSNR range 22.8–40.1, SSIM 0.701–0.955)

### 4. `platform/scripts/generate_challenge_datasets.py`
- Added `"ct_fluorescence": "identity"` to `_VARIANT_TO_RUNNER`
- Added `generate_ct_fluorescence_phantom` to 2 import blocks and 2 generator maps

## GCS Upload Result (2026-03-09)

All 3 challenge tiers generated and uploaded successfully:
```
gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_public.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_dev.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_hidden.h5
```
- 3 samples per tier
- Seed offsets: public=0, dev=+10000, hidden=+20000 (per-tier differentiation)
- Dev tier: no x_true (stripped per policy)
- Hidden tier: blocked from download (GCS proxy _BLOCKED_PATTERNS)

## Physics Rationale for XRF-CT Phantom Design

XRF-CT measures the spatial distribution of K-edge fluorescent elements (Au, I, Gd, Ba) injected as contrast agents or nanoparticles. The physical forward model is:
- Incident X-ray beam (pencil beam, energy > K-edge) excites fluorescence emission
- Emitted photons collected at ~90° to beam direction (minimises scatter contribution)
- Compton scatter from primary beam creates spatially uniform background
- Measurement: Poisson-distributed fluorescence counts + Poisson-distributed scatter

The phantom uses ellipsoidal clusters to simulate nanoparticle accumulation regions (e.g., tumour uptake), with low uniform background representing tissue autofluorescence. This is the standard phantom geometry used in Larsson et al. (2020) and Boisseau & Grodzins (1987).
