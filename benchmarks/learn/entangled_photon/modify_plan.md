# Modify Plan: entangled_photon

## Change Log

### 2026-03-09 — Phantom Generator, Algorithm Overrides, GCS Datasets

**Changes made:**

1. **`benchmarks/datasets/downloaders.py`** — Added `generate_entangled_photon_phantom()`:
   - 64×64 float32 object transmission map (thin biological sample)
   - Clear background (~1.0), semi-transparent cytoplasm (~0.7-0.9), absorbing nuclei (~0.1-0.3)
   - SPDC forward model: Gaussian blur sigma~2 px + Poisson noise at ~10 photons/pixel
   - Returns 3 samples as list of dicts with x_true, y, H_ideal, metadata
   - Registered in both `_generated_converters` and `converter_map`

2. **`benchmarks/datasets/registry.py`** — Added `entangled_photon_generated` DatasetEntry:
   - source_type="generated", storage="local", applies_to=["entangled_photon"]
   - converter="generate_entangled_photon_phantom", x_shape=[64, 64]

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**:
   - Added `_VARIANT_OVERRIDES["entangled_photon"]` with 9 algorithms:
     Coincidence-Count, CS-Ghost, SVD-Ghost, DnCNN-Ghost, GAN-Ghost,
     TransGhost, SwinGhost, PhysGhost, DiffGhost
   - Added `CATEGORY_REAL_SCORES["entangled_photon"]` with PSNR/SSIM benchmarks
     ranging from 19.8/0.658 (Coincidence-Count) to 38.8/0.950 (DiffGhost)

4. **`platform/scripts/generate_challenge_datasets.py`**:
   - Added `"entangled_photon": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_entangled_photon_phantom` to both generator import blocks and both generator maps

5. **GCS datasets uploaded** (3 tiers × 1 variant):
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/entangled_photon_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/entangled_photon_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/entangled_photon_challenge_hidden.h5`

---

### 2026-03-06 — Initial Assignment

- **Category:** quantum
- **Carrier:** Photon
- **Score key:** quantum
- **Algorithms (10 total from quantum pool):**
  1. G(2)-Corr (Classical) -- Pittman et al., PRA 1995
  2. Photon Counting (Classical) -- Classical baseline
  3. CS-TVAL3 (PnP) -- Li et al., 2014
  4. Bayesian CS (PnP) -- Bayesian compressed sensing
  5. DRU-Net (Deep Learning) -- Wang et al., Sci. Rep. 2020
  6. Quantum-CNN (Deep Learning) -- Quantum imaging CNN
  7. Ghost-ViT (Vision Transformer) -- Zhu et al., 2025
  8. Quantum-ViT (Vision Transformer) -- Quantum imaging transformer, 2024
  9. DiffusionQuantum (Diffusion) -- Zhang et al., 2024
  10. ScoreQuantum (Score-based) -- Wei et al., 2025

**Status:** PASS — check.md written 2026-03-06

## Current Status (2026-03-09)

- **Algorithm override:** 9-algorithm `_VARIANT_OVERRIDES["entangled_photon"]` active
- **Phantom generator:** `generate_entangled_photon_phantom` deployed
- **GCS datasets:** All 3 tiers uploaded
- **Status:** PASS
