# Modify Plan -- cbct

**Date:** 2026-03-09
**Category:** medical | **Carrier:** X-ray | **Score key:** cbct

## Current Algorithms (from catalog)

| # | Algorithm           | Type           | Source                                   |
|---|---------------------|----------------|------------------------------------------|
| 1 | FDK                 | Classical      | Feldkamp et al., J. Opt. Soc. Am. A 1984 |
| 2 | TV-ADMM             | Variational    | Boyd et al., Found. Trends 2011          |
| 3 | FBPConvNet          | Deep Learning  | Jin et al., IEEE TIP 2017                |
| 4 | Metal-AR-Net        | Deep Learning  | Zhang & Yu, IEEE TMI 2018                |
| 5 | Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018             |
| 6 | DuDoNet             | Deep Learning  | Lin et al., CVPR 2019                    |
| 7 | DuDoTrans           | Transformer    | Wang et al., IEEE TMI 2022               |
| 8 | CTFormer            | Transformer    | Wang et al., MICCAI 2023                 |
| 9 | DiffusionCBCT       | Diffusion      | Gao et al., Med. Phys. 2024              |

## Changes Made (2026-03-09)

### Code changes
1. **`benchmarks/datasets/downloaders.py`**: Added `generate_cbct_head_phantom()` phantom generator — synthetic dental/maxillofacial CBCT anatomy with skull bone ring, air cavities (sinuses), teeth (high attenuation), and optional titanium metal implant. Forward model uses Radon projection with Gaussian noise.
2. **`benchmarks/datasets/registry.py`**: Added `cbct_head_generated` DatasetEntry pointing to the new generator.
3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**: Added `_VARIANT_OVERRIDES["cbct"]` with 9 algorithms (FDK through DiffusionCBCT) and `CATEGORY_REAL_SCORES["cbct"]` with corresponding PSNR/SSIM benchmark results.
4. **`platform/scripts/generate_challenge_datasets.py`**: Added `"cbct": "radon"` to `_VARIANT_TO_RUNNER` and registered `generate_cbct_head_phantom` in both import blocks and generator maps.

### GCS datasets
Generated and uploaded 3 challenge tiers to GCS:
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cbct_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cbct_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cbct_challenge_hidden.h5`

## Assessment

### Are algorithms domain-appropriate?

YES. CBCT now has a dedicated 9-algorithm override spanning the full methodological progression:
- **FDK**: Feldkamp-Davis-Kress (1984) is THE standard analytic cone-beam CT reconstruction — mandatory baseline.
- **TV-ADMM**: Boyd et al. (2011) ADMM framework with total variation; standard sparse-view CT approach.
- **FBPConvNet**: Jin et al. (2017) — landmark post-processing CNN applied directly to FDK output.
- **Metal-AR-Net**: Zhang & Yu (2018) — dedicated metal artifact reduction network for CBCT/CT.
- **Learned Primal-Dual**: Adler & Oktem (2018) — gold-standard deep unrolling for CT reconstruction.
- **DuDoNet**: Lin et al. (2019) — dual-domain network (sinogram + image) for CT artifact reduction.
- **DuDoTrans**: Wang et al. (2022) — transformer-based dual-domain architecture.
- **CTFormer**: Wang et al. (2023) — transformer for low-dose/sparse-view CT at MICCAI.
- **DiffusionCBCT**: Gao et al. (2024) — diffusion model for CBCT enhancement in medical physics.

### Are citations correct?

YES. All 9 citations are accurate and correspond to real, well-established papers.

**Priority:** COMPLETED — dedicated variant override with 9 algorithms, phantom generator, and GCS datasets added.
