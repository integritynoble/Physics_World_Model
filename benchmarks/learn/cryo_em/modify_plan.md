# Modify Plan: cryo_em (Cryo-EM Single Particle Analysis)

**Updated:** 2026-03-09
**Status:** PASS — phantom generator, algorithm overrides, GCS datasets deployed

## Change Log

### 2026-03-09
- Added `generate_cryo_em_phantom()` to `benchmarks/datasets/downloaders.py`:
  - Simulates 2D projection of protein ellipsoid with internal density blobs
  - Applies CTF corruption (defocus 1–3 µm, Cs=2.7mm, V=300kV) in Fourier domain
  - Adds Poisson noise at ~10 electrons/Å²
  - Returns list of 3 dicts with x_true, y, H_ideal, metadata
  - Registered in both `_generated_converters` and `converter_map`
- Added `cryo_em_generated` DatasetEntry to `benchmarks/datasets/registry.py`
- Added `cryo_em` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py` with 9 algorithms:
  CTFFIND4, RELION-3D, cryoSPARC, IsoNet, cryoDRGN, CryoGEM, CryoFormer, CryoSTAR, DiffusionCryo
- Added `cryo_em` scores to `CATEGORY_REAL_SCORES` with PSNR 22.3–39.8 dB
- Added `"cryo_em": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- Added `generate_cryo_em_phantom` to both import blocks and generator maps in
  `generate_challenge_datasets.py`
- Generated and uploaded 3 challenge tiers (public, dev, hidden) to GCS:
  `gs://pwm-benchmark-datasets/challenge-data/v1.0/cryo_em_challenge_{tier}.h5`

### 2026-03-06 (prior)
- Algorithm routing: `cryo_em` variant receives the correct cryo-EM pool (RELION, cryoSPARC,
  cryoDRGN, CryoTransformer, etc.) as confirmed by direct Python inspection.
- All key algorithms (RELION 1.0, cryoSPARC, cryoDRGN, CryoTransformer) are real, well-cited packages.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: defocus_error, astigmatism, beam_tilt, ice_thickness_variation.

## Verdict

PASS. Phantom generator, 9 algorithm overrides, benchmark scores, and GCS datasets all deployed.
Identity runner used since the phantom handles CTF+Poisson noise internally.
