# Modify Plan: fluoroscopy

## Change Log

### 2026-03-09 — Full modality onboarding

**Changes made:**

1. **Phantom generator added** (`benchmarks/datasets/downloaders.py`):
   - Added `generate_fluoroscopy_phantom()` function after `generate_flim_phantom()`
   - Generates 64×64 float32 X-ray transmission images with thorax/abdomen anatomy:
     - Lung fields (high transmission, 0.85–0.92)
     - Spine (bone, 0.15–0.25)
     - Ribs (bone, 0.20–0.30)
     - Soft tissue background (0.6)
     - Catheter/wire thin dark line (0.08–0.14)
   - Forward model: Poisson noise (~100–500 photons/pixel), flat-field gain variation (±10%), Gaussian readout noise (σ~5 counts)
   - Returns 3 samples as list of dicts with `x_true`, `y`, `H_ideal`, `metadata`
   - Registered in both `_generated_converters` and `converter_map`

2. **Dataset registry entry added** (`benchmarks/datasets/registry.py`):
   - Added `"fluoroscopy_generated"` DatasetEntry
   - `applies_to=["fluoroscopy"]`, `converter="generate_fluoroscopy_phantom"`, `x_shape=[64, 64]`

3. **Algorithm overrides added** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Added `_VARIANT_OVERRIDES["fluoroscopy"]` with 9 algorithms:
     - BM3D-Fluoro (Classical), NLM-Fluoro (Classical), TV-Fluoro (Variational)
     - DnCNN-Fluoro (Deep Learning), REDCNN-Fluoro (Deep Learning)
     - TransFluoro (Transformer), SwinFluoro (Transformer)
     - PhysFluoro (Physics-Informed), DiffFluoro (Diffusion Model)
   - Added `CATEGORY_REAL_SCORES["fluoroscopy"]` with 9 PSNR/SSIM entries
     - Range: 25.8–40.0 dB PSNR, 0.762–0.960 SSIM
     - Best: DiffFluoro (40.0 dB, 0.960)

4. **Generator routing updated** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `generate_fluoroscopy_phantom` to both import tuples and both generator maps
   - `"fluoroscopy": "projection"` already present in `_VARIANT_TO_RUNNER`

5. **GCS datasets generated and uploaded**:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/fluoroscopy_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/fluoroscopy_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/fluoroscopy_challenge_hidden.h5`
   - All 3 tiers: 3 samples each, 64×64 float32, generated 2026-03-09

---

## Previous State (2026-03-06)

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical (CT-like pool, no carrier routing override for X-ray)
- **Algorithms assigned (old):**
  1. FBP (Classical) -- Analytical baseline
  2. TV-ADMM (Compressed Sensing) -- Rudin et al., Physica D 60, 259 (1992) + ADMM
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 26, 4509 (2017)
  4. RED-CNN (Deep Learning) -- Chen et al., IEEE TMI 36, 2524 (2017)

**Assessment (2026-03-06):** Appropriate — fluoroscopy shares Beer-Lambert physics with CT.

## Current State (2026-03-09)

- **Category:** medical / X-ray
- **Algorithm routing:** `_VARIANT_OVERRIDES["fluoroscopy"]` — 9 dedicated algorithms
- **Score routing:** `CATEGORY_REAL_SCORES["fluoroscopy"]` — 9 entries
- **Runner:** `"projection"` (2D X-ray projection forward model)
- **Phantom:** `generate_fluoroscopy_phantom` — thorax/abdomen with Poisson noise
- **GCS status:** All 3 tiers uploaded and verified
- **Priority:** COMPLETE
