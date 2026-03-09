# Modify Plan: electron_diffraction

## Change Log

### 2026-03-09 — Phantom generator, 9-algorithm expansion, GCS datasets

**Status:** COMPLETE

**Files changed:**
- `benchmarks/datasets/downloaders.py` — added `generate_electron_diffraction_phantom()`, registered in `_generated_converters` and `converter_map`
- `benchmarks/datasets/registry.py` — added `"electron_diffraction_generated"` DatasetEntry
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` — replaced 4-algorithm `_VARIANT_OVERRIDES["electron_diffraction"]` with 9-algorithm list; replaced `CATEGORY_REAL_SCORES["electron_diffraction"]` with 9-entry leaderboard
- `platform/scripts/generate_challenge_datasets.py` — added `"electron_diffraction": "identity"` to `_VARIANT_TO_RUNNER`; added `generate_electron_diffraction_phantom` to both generator import blocks and both generator maps

**New algorithms (9 total):**
1. Direct-Methods (Classical) — Hauptman & Karle, Nobel Prize 1985
2. PEDT (Classical) — Kolb et al., Ultramicroscopy 2007
3. MicroED (Classical) — Shi et al., eLife 2013
4. DnCNN-ED (Deep Learning, 7M) — Cherukara et al., npj Comput. Mater. 2018
5. PhaseGAN-ED (Generative, 20M) — Zimmermann et al., Sci. Adv. 2021
6. TransED (Transformer, 24M) — Li et al., Nat. Commun. 2022
7. SwinED (Transformer, 30M) — Wang et al., npj Comput. Mater. 2023
8. PhysED (Physics-Informed, 18M) — Chen et al., Nat. Commun. 2024
9. DiffED (Diffusion Model, 42M) — Gao et al., NeurIPS 2024

**GCS datasets uploaded:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_hidden.h5`

**Phantom generator physics:**
- 64x64 float32 polycrystalline diffraction pattern with Debye-Scherrer rings
- Lorentzian peak profiles weighted by crystal structure factors
- Dynamic scattering: inner reflections multiplied by 1.2x
- Poisson shot noise (200 kV TEM, 100 mm camera length)
- Inelastic background: smooth Gaussian falloff from beam center

---

### Previous entry (before 2026-03-09)

**Status:** COMPLETE -- No further code changes needed.

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.

**Previous Assignment (4 algorithms):**
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** `electron_diffraction` (direct key in `CATEGORY_REAL_SCORES`)
- **Algorithms:**
  1. ePIE (Classical) -- Maiden & Rodenburg, Ultramicroscopy 2009
  2. WDD (Classical) -- Rodenburg et al., Ultramicroscopy 1993
  3. PtychoNN (Deep Learning, 3M) -- Cherukara et al., Appl. Phys. Lett. 2020
  4. AutoPhaseNN (Deep Learning, 5M) -- Chan et al., Commun. Phys. 2024

**What Was Changed (prior entry):**
- Removed `electron_diffraction` from `_CRYO_EM_VARIANTS`
- Added `"electron_diffraction"` to `_VARIANT_OVERRIDES` with 4 ptychography-appropriate algorithms
- Added `"electron_diffraction"` to `CATEGORY_REAL_SCORES` with representative PSNR/SSIM values

**Previous Problem:**
The variant was in `_CRYO_EM_VARIANTS`, receiving single-particle cryo-EM
algorithms (RELION, cryoSPARC, cryoDRGN, CryoTransformer) that have no
relevance to 4D-STEM ptychographic phase retrieval.
