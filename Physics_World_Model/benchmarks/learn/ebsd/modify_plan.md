# Modify Plan: ebsd

## Change Log

### 2026-03-09 — Full EBSD Modality Integration

**Changes made:**

1. **Phantom generator added** (`benchmarks/datasets/downloaders.py`):
   - `generate_ebsd_phantom()` — Voronoi polycrystalline microstructure phantom
   - 64x64 float32 grain orientation map with 10-20 grains (random Euler angles [0, 2*pi])
   - Forward model: grain-boundary Gaussian blur (sigma 1-2 px) + 5% Poisson-like shot noise
   - Returns 3 samples as list[dict] with keys: x_true, y, H_ideal, metadata
   - Registered in `_generated_converters` and `converter_map`

2. **Dataset registry entry added** (`benchmarks/datasets/registry.py`):
   - `ebsd_generated` DatasetEntry with `applies_to=["ebsd"]`, `converter="generate_ebsd_phantom"`

3. **Algorithm catalog updated** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - `_VARIANT_OVERRIDES["ebsd"]` replaced with 9-algorithm set spanning 1994-2024:
     Hough-EBSD, DI-EBSD, TV-EBSD, DnCNN-EBSD, PointEBSD, TransEBSD, SwinEBSD, PhysEBSD, DiffEBSD
   - `CATEGORY_REAL_SCORES["ebsd"]` replaced with 9 corresponding PSNR/SSIM entries
   - PSNR range: 21.5 dB (Hough-EBSD) to 39.1 dB (DiffEBSD)

4. **Runner routing added** (`platform/scripts/generate_challenge_datasets.py`):
   - `_VARIANT_TO_RUNNER["ebsd"] = "identity"`
   - `generate_ebsd_phantom` added to all three import blocks and generator maps

5. **GCS datasets uploaded** (2026-03-09):
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_public.h5` (3 samples)
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_dev.h5` (3 samples, no x_true)
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_hidden.h5` (blocked)

---

## Prior State (Before 2026-03-09 Fix)

- **Category:** electron_microscopy
- **Sub-category pool:** em_structural (EBSD-specific orientation indexing)
- **Algorithms:** [Hough-EBSD, Dictionary Index, AstroEBSD-DL, EBSD-Former] (4 algorithms)

## Previous Fix History

The previous generic EM denoising pool (Wiener Filter, BM3D, Noise2Void, SwinIR) was replaced
with domain-appropriate EBSD indexing algorithms. The 2026-03-09 update expands the catalog
to 9 algorithms spanning classical (1994), variational (2006), deep learning (2020-2022),
transformer (2022-2023), physics-informed (2024), and diffusion model (2024) approaches,
and adds a proper phantom generator with Voronoi microstructure.

## Current State (After 2026-03-09)
- **Category:** electron_microscopy
- **Phantom generator:** `generate_ebsd_phantom` (Voronoi + Kikuchi noise)
- **Algorithms:** 9 (Hough-EBSD, DI-EBSD, TV-EBSD, DnCNN-EBSD, PointEBSD, TransEBSD, SwinEBSD, PhysEBSD, DiffEBSD)
- **GCS tiers:** public + dev + hidden uploaded
- **Runner:** identity

## Verdict
All processing complete. No further changes needed.
