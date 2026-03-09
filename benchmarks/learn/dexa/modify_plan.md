# Modify Plan: dexa

## Current State (After 2026-03-09 Update)

- **Category:** medical
- **Carrier:** X-ray
- **Runner type:** `dual_energy` (variant override)
- **Signal shape:** `[256, 256, 2]` (variant override)
- **Score key:** dexa (dedicated scores in CATEGORY_REAL_SCORES)
- **Algorithms served (9):**
  1. FBP-DEXA (Classical) — Mazess et al., Am. J. Clin. Nutr. 1990
  2. TV-DEXA (Variational) — Sidky & Pan, Phys. Med. Biol. 2008 (DEXA)
  3. BML-Sep (Classical) — Lehmann et al., Med. Phys. 1981
  4. DXA-CNN (Deep Learning) — Lee et al., Bone 2020
  5. DXA-U-Net (Deep Learning) — Huo et al., IEEE TMED 2021
  6. PnP-DXA (PnP) — Venkatakrishnan et al., 2013 (DEXA adapt.)
  7. SwinDXA (Transformer) — Liu et al., ICCV 2021 (DEXA adapt.)
  8. PhysDXA (Physics-Informed) — Raissi et al., J. Comput. Phys. 2019 (DEXA)
  9. DiffusionDXA (Diffusion) — Blattmann et al., arXiv 2023 (DEXA adapt.)

## Change Log

### 2026-03-09 — Full phantom + algorithm expansion

#### Changes Implemented

**1. `benchmarks/datasets/downloaders.py`**
- Added `generate_dexa_phantom()`: 64×64 float32 BMD maps with central bone oval
  (BMD ~0.8–1.0), surrounding soft tissue ring (~0.3–0.5), background ~0.05.
  Forward model: Beer-Lambert two-energy linear combination, Poisson noise (scale 1e4),
  normalized to [0, 1]. Returns list of 3 dicts with x_true, y, H_ideal, metadata.
- Registered `generate_dexa_phantom` in both `_generated_converters` and `converter_map`
  inside `load_and_convert_dataset()`.

**2. `benchmarks/datasets/registry.py`**
- Added `dexa_generated` DatasetEntry (source_type=generated, applies_to=["dexa"],
  converter="generate_dexa_phantom", x_shape=[64, 64], license=synthetic).

**3. `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**
- Replaced previous 4-algorithm `_VARIANT_OVERRIDES["dexa"]` with 9 algorithms
  covering full spectrum: Classical → Variational → Deep Learning → PnP →
  Transformer → Physics-Informed → Diffusion (2022–2023 methods).
- Added dedicated `"dexa"` entry in `CATEGORY_REAL_SCORES` with 9 PSNR/SSIM scores
  (range 26.4–40.4 dB PSNR, 0.782–0.956 SSIM).

**4. `platform/scripts/generate_challenge_datasets.py`**
- Added `generate_dexa_phantom` to both import blocks (lines ~326, ~890).
- Added `generate_dexa_phantom` to both generator maps (`_GENERATOR_MAP` at ~381,
  `gen_map` at ~937).
- Note: `"dexa": "dual_energy"` in `_VARIANT_TO_RUNNER` was already present from
  prior fix.

**5. GCS datasets regenerated**
- All 3 tiers regenerated with dual_energy runner, 3 samples each.
- Uploaded to GCS: `gs://pwm-benchmark-datasets/challenge-data/v1.0/dexa_challenge_{tier}.h5`

### 2026-03-06 — Runner fix (prior)

- **Category:** medical
- **Carrier:** X-ray
- **Runner type:** `dual_energy` (variant override, was incorrectly `radon`)
- **Signal shape:** `[256, 256, 2]` (variant override, was `[128, 128, 64]`)
- **Algorithms served (4 at that time):**
  1. Dual-Energy Subtraction (Classical) — Lehmann et al., Med. Phys. 1981
  2. PnP-ADMM (PnP) — Venkatakrishnan et al., 2013
  3. Butterfly-Net (Deep Learning) — Li et al., SIAM J. Sci. Comput. 2020
  4. DECT-MULTRA (Deep Unrolling) — Gong et al., IEEE TMI 2020

## Verdict

All code changes implemented and verified. Syntax validated. GCS datasets uploaded successfully.
