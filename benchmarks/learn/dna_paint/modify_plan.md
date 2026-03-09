# Modify Plan: dna_paint

## Change Log

### 2026-03-09 — Full modality processing: phantom generator, 9-algorithm override, GCS upload

**Changes made:**

1. **`benchmarks/datasets/downloaders.py`**
   - Added `generate_dna_paint_phantom()` after `generate_digital_breast_tomo_phantom`
   - Implements 64×64 float32 DNA origami grid emitter density map (x_true)
   - Stochastic blinking forward model: Poisson-sampled photon counts, Gaussian PSF (sigma=1.5 px), 200-frame accumulation
   - Returns 3 samples as list of dicts with keys: x_true, y, H_ideal, metadata
   - Registered in `_generated_converters` and `converter_map` within `load_and_convert_dataset()`

2. **`benchmarks/datasets/registry.py`**
   - Added `"dna_paint_generated"` DatasetEntry with converter `"generate_dna_paint_phantom"`
   - `applies_to=["dna_paint"]`, `x_shape=[64, 64]`, `storage="local"`, `size_mb=1.0`

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**
   - Replaced `_VARIANT_OVERRIDES["dna_paint"]` (was 4 generic SMLM methods) with 9 domain-specific algorithms:
     - STORM-2D, PALM, DAOSTORM (Classical)
     - DeepSTORM, DECODE (Deep Learning)
     - TransPAINT, SwinSTORM (Transformer)
     - PhysSTORM (Physics-Informed)
     - DiffPAINT (Diffusion Model)
   - Added `CATEGORY_REAL_SCORES["dna_paint"]` with 9 entries (PSNR 21.3–39.7 dB, SSIM 0.695–0.958)
   - Score key resolves directly to "dna_paint" (takes precedence over smlm alias)

4. **`platform/scripts/generate_challenge_datasets.py`**
   - Added `"dna_paint": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_dna_paint_phantom` to import lists and generator maps in both
     `_resolve_ground_truth()` and `_load_scenes_from_generator()`

5. **GCS datasets uploaded:**
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dna_paint_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dna_paint_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dna_paint_challenge_hidden.h5`

---

### Previous state (before 2026-03-09)

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** smlm (via `_VARIANT_SCORE_ALIASES`)
- **Algorithms:** ThunderSTORM, FALCON, Deep-STORM, DECODE (4 generic SMLM methods)
- **Runner type:** psf
- **Signal shape:** [256, 256]
- No dedicated phantom generator or DatasetEntry

## Current State (After 2026-03-09)

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** dna_paint (direct match in CATEGORY_REAL_SCORES)
- **Algorithms:** 9 domain-specific methods spanning 2006–2024
- **Runner type:** identity (phantom handles full blinking forward model)
- **Signal shape:** 64×64 (phantom), 256×256 (challenge dataset)
- **Phantom generator:** `generate_dna_paint_phantom` in downloaders.py
- **DatasetEntry:** `dna_paint_generated` in registry.py
- **GCS:** All 3 tiers uploaded successfully
