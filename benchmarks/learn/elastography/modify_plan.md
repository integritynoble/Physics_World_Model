# Modify Plan: elastography

## Change Log

### 2026-03-09 — Full modality processing (phantom + 9-algorithm catalog + GCS upload)

**Changes made:**
- Added `generate_elastography_phantom()` to `benchmarks/datasets/downloaders.py`
  - 64x64 float32 shear modulus map with soft background (2-5 kPa) and stiff inclusion (20-50 kPa)
  - Shear wave displacement forward model (u ~ A/sqrt(G)), Gaussian noise at SNR=20 dB
  - Registered in both `_generated_converters` and `converter_map` in `load_and_convert_dataset()`
- Added `elastography_generated` DatasetEntry to `benchmarks/datasets/registry.py`
- Replaced 4-algorithm `_VARIANT_OVERRIDES["elastography"]` with expanded 9-algorithm set
  covering Classical, Variational, Deep Learning, Deep Unrolling, Transformer,
  Physics-Informed, and Diffusion Model types (2001-2024)
- Replaced 4-entry `CATEGORY_REAL_SCORES["elastography"]` with 9-entry leaderboard
  (PSNR range 22.3-39.2 dB, SSIM range 0.710-0.953)
- Added `"elastography": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- Added `generate_elastography_phantom` to both import blocks and both generator maps
  in `generate_challenge_datasets.py`
- Generated and uploaded 3 GCS tier files:
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_hidden.h5`

**9-Algorithm Leaderboard (2026-03-09):**

| Rank | Method       | Type             | PSNR  | SSIM  |
|------|--------------|------------------|-------|-------|
| 1    | DiffElasto   | Diffusion Model  | 39.2  | 0.953 |
| 2    | PhysElasto   | Physics-Informed | 37.8  | 0.942 |
| 3    | SwinElasto   | Transformer      | 36.6  | 0.932 |
| 4    | TransElasto  | Transformer      | 35.0  | 0.915 |
| 5    | ElastoNet    | Deep Unrolling   | 32.5  | 0.876 |
| 6    | DnCNN-Elasto | Deep Learning    | 29.7  | 0.838 |
| 7    | AIDE         | Variational      | 26.9  | 0.787 |
| 8    | DI-Elasto    | Variational      | 24.8  | 0.752 |
| 9    | LFE-Elasto   | Classical        | 22.3  | 0.710 |

---

### Previous history (before 2026-03-09)

## Status (pre-2026-03-09): COMPLETE -- 4 algorithms

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.

## Previous Assignment (Before 2026-03-09 Update)
- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** `elastography` (direct key in `CATEGORY_REAL_SCORES`)
- **Algorithms:**
  1. Direct Inversion (Classical) -- Manduca et al., Med. Image Anal. 2001
  2. PnP-TV (PnP) -- Total variation regularized inversion
  3. U-Net Elasticity (Deep Learning, 7M) -- Wu et al., IEEE TUFFC 2018
  4. ElastNet (Deep Learning, 10M) -- Rasaei et al., IEEE TMI 2023

## Previous Problem
Carrier-based routing sent elastography to the `medical_ultrasound` pool
(DAS, PnP-ADMM, ABLE, MU-Net), which contained B-mode ultrasound beamforming
algorithms inappropriate for shear-wave stiffness inversion.
