# Modify Plan: digital_breast_tomo (Digital Breast Tomosynthesis)

**Updated:** 2026-03-09
**Status:** COMPLETE — DBT-specific phantom, algorithm overrides, and GCS datasets added

## Change Log

### 2026-03-09 — DBT Modality Full Integration

**Changes made:**

1. **Phantom generator** (`benchmarks/datasets/downloaders.py`):
   - Added `generate_digital_breast_tomo_phantom()` — 64×64 float32 breast phantom with
     adipose tissue background (~0.15-0.25), glandular regions (~0.55-0.80), and small
     lesion/mass (~0.85-1.0)
   - Forward model: 11 angles (-25° to +25°), Poisson noise (dose factor 0.3-0.7), FBP back-projection
   - Returns 3 samples as list[dict] with keys: x_true, y, H_ideal, metadata
   - Registered in `_generated_converters` and `converter_map` in `load_and_convert_dataset()`

2. **Dataset registry** (`benchmarks/datasets/registry.py`):
   - Added `"digital_breast_tomo_generated"` DatasetEntry with converter
     `"generate_digital_breast_tomo_phantom"`, applies_to `["digital_breast_tomo"]`

3. **Algorithm catalog** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Added `_VARIANT_OVERRIDES["digital_breast_tomo"]` with 9 DBT-specific algorithms:
     FBP-DBT, TV-DBT, SART-DBT, DnCNN-DBT, DuDoRNet-DBT, TransDBT, SwinDBT, PhysDBT, DiffusionDBT
   - Added `CATEGORY_REAL_SCORES["digital_breast_tomo"]` with corresponding PSNR/SSIM values
     (range: 23.1-39.4 dB PSNR, 0.721-0.956 SSIM)

4. **Runner routing** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `"digital_breast_tomo": "radon"` to `_VARIANT_TO_RUNNER`
   - Added `generate_digital_breast_tomo_phantom` to both import blocks and generator maps

5. **GCS datasets** — all 3 tiers generated and uploaded:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_hidden.h5`

---

### 2026-03-06 — Initial Check

**Status:** PASS — no code changes required at that time

- Algorithm routing via carrier routing `(medical, X-ray)` → CT pool (13 methods)
- DBT is a limited-angle X-ray CT modality; CT algorithms are applicable
- Challenge datasets on GCS for all three tiers
- Mismatch parameters: angular_range_error, detector_motion_blur, scatter_fraction
