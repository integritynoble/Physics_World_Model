# Modify Plan: dot

## Change Log

### 2026-03-09 — Full DOT phantom, algorithm expansion, GCS deployment

**Changes made:**

1. **Phantom generator** (`benchmarks/datasets/downloaders.py`):
   - Added `generate_dot_phantom()` — 64x64 float32 absorption coefficient map
   - Tissue background mu_a ~0.01-0.02 mm^-1, tumor inclusions mu_a ~0.05-0.10 mm^-1
   - Born approximation forward model with 4 source-detector pairs at boundary
   - 3% relative Gaussian noise; returns 3 samples as list of dicts
   - Registered in both `_generated_converters` and `converter_map`

2. **Dataset registry** (`benchmarks/datasets/registry.py`):
   - Added `dot_generated` DatasetEntry with `converter="generate_dot_phantom"`
   - applies_to=["dot"], x_shape=[64, 64], storage="local"

3. **Algorithm catalog** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Replaced `_VARIANT_OVERRIDES["dot"]` (4 algorithms) with 9-algorithm entry
   - Methods: Born-Approx, TV-DOT, FEM-DOT, DnCNN-DOT, DOT-Net, TransDOT, SwinDOT, PhysDOT, DiffusionDOT
   - Covers 1999-2024 literature; SOTA: DiffusionDOT (39.0 dB PSNR, 0.954 SSIM)
   - Replaced `CATEGORY_REAL_SCORES["dot"]` (4 entries) with 9-entry leaderboard

4. **Generator routing** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `"dot": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_dot_phantom` to `_GENERATOR_MAP`, `gen_map`, and both import blocks

5. **GCS datasets** — all 3 tiers uploaded:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dot_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dot_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dot_challenge_hidden.h5`

---

## Previous State (Before 2026-03-09)

- **Category:** medical
- **Carrier:** Photon
- **Score key:** dot (via variant-specific CATEGORY_REAL_SCORES)
- **Algorithms:** Tikhonov-Born, L-BFGS-TV, PnP-Diffusion, DeepDOT (4-algorithm DOT-specific override)
- **Runner:** radon (acceptable simplification, now changed to identity)

## Assessment (2026-03-09)

Expanded to 9 algorithms spanning 25 years of DOT literature. Dedicated phantom
generator replaces radon-based fallback. Identity runner matches the
Born-approximation measurement structure of the phantom. GCS datasets reflect
the new phantom generator with proper source-detector geometry.
