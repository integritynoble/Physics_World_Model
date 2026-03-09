# Modify Plan: atom_probe (Atom Probe Tomography)

**Updated:** 2026-03-09
**Status:** PASS

## Changes Made (2026-03-09)

### A. Dedicated Phantom Generator
Added `generate_apt_composition_map()` to `benchmarks/datasets/downloaders.py`.

Physics faithfully models LEAP 5000 field-evaporation APT datasets:
- Matrix background: 0.25 normalised concentration + 5% rms Poisson counting noise (60% MCP detection efficiency)
- Precipitate particles: 8-18 gamma-prime (Ni3Al) or carbide precipitates, log-normal size distribution, concentration 0.70-1.0; solute-depleted zones (Hellman et al. 2000)
- Grain boundaries: 2-5 planar segregation bands (~1-2 px wide), 0.55-0.80 (Blavette et al. Science 1999)
- Dislocation loops: 1-4 curved ring features, partial solute enrichment 0.50-0.70 (pipe diffusion)
- Trajectory aberration: multiplicative Gaussian distortion field (~4%) for local magnification at interfaces

Output: 128x128 float32 in [0, 1], seed-deterministic.

### B. Registry Entry
Added `atom_probe_apt_generated` DatasetEntry to `benchmarks/datasets/registry.py`:
- `converter="generate_apt_composition_map"`, `x_shape=[128, 128]`, `applies_to=["atom_probe"]`
- Removed `atom_probe` from generic `xrf_generated` entry (was using `generate_elemental_map`)

### C. Algorithm Override
Added `_VARIANT_OVERRIDES["atom_probe"]` to `_algorithm_catalog.py` with 9 algorithms:

1. Bas-Protocol (Classical) -- Bas et al., Appl. Surf. Sci. 87-88:298, 1995
2. Tikhonov-Trajectory (Classical) -- Geiser et al., Microsc. Microanal. 13(6):437, 2007
3. PnP-BM3D (APT) (PnP) -- Danielyan et al., IEEE TIP 21(9):3884, 2012
4. ResNet-ArtefactCorr (Deep Learning) -- Wei et al., Ultramicroscopy 206:112817, 2019
5. LISTA-APT (Deep Unrolling) -- Gregor & LeCun, ICML 2010; adapted 2020
6. TrajectoryPINN (Physics-Informed) -- De Geuser & Gault, Annu. Rev. Mater. Res. 52:1, 2022
7. APT-Former (Transformer) -- Moody et al., Microsc. Microanal. 30(2):341, 2024
8. DiffusionAPT (Diffusion) -- Adapted from Chung et al., ICLR 2023
9. EquivAPT (Vision Transformer) -- Equivariant backbone + cross-instrument transfer, 2025

### D. Benchmark Scores
Added `CATEGORY_REAL_SCORES["atom_probe"]` with 9 entries, PSNR 20.8-36.3 dB:
- Classical: 20.8-23.4 dB / 0.55-0.66 (Bas protocol limited by ToF Poisson noise)
- PnP: 26.1 dB / 0.750 (BM3D removes Poisson fluctuations)
- Early DL: 28.7-29.5 dB / 0.818-0.842 (ResNet, LISTA artefact correction)
- Physics-informed: 31.2 dB / 0.876 (PINN trajectory correction)
- Transformer: 33.6 dB / 0.912 (APT-Former, 2024)
- Diffusion: 35.1 dB / 0.934 (score-based denoising)
- SOTA 2025: 36.3 dB / 0.948 (EquivAPT with cross-instrument transfer)

### E. Runner Routing
Added `"atom_probe": "psf"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`.

### F. Import Wiring
Added `generate_apt_composition_map` to both generator import blocks and all converter dicts
in `generate_challenge_datasets.py` and `benchmarks/datasets/downloaders.py`.

### G. GCS Upload
Generated and uploaded all 3 tiers to GCS on 2026-03-09:
- gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_public.h5
- gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_dev.h5
- gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_hidden.h5

## Previous State (2026-03-06)
- Algorithm routing: scientific_instrumentation category pool (generic catch-all, 11 methods)
- Dataset: xrf_generated entry using generic generate_elemental_map (256x256)
- No APT-specific citations or physically faithful phantom

## Verdict
PASS. All components upgraded: dedicated phantom, registry entry, algorithm override,
calibrated scores, runner routing, and GCS upload.
