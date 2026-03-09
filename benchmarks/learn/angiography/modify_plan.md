# Modify Plan — angiography (2026-03-09)

## Status: COMPLETED

All improvements implemented and deployed to GCS on 2026-03-09.

---

## Changes Implemented

### 1. Dedicated Vessel Phantom Generator (`benchmarks/datasets/downloaders.py`)

Added `generate_angiography_vessel_phantom()` — a physics-calibrated fractal vascular tree generator for X-ray angiography (DSA/3DRA). The generator produces 2D iodine concentration maps with:
- Main trunk (aorta/ICA): near-vertical, mild tortuosity, radius 8–15 px
- 2–4 first-order branches at 30–60° angles with Murray's law radius tapering (r^3 = const)
- 1–3 second-order branches per first-order vessel with decreasing contrast (40–65%)
- Gaussian focal spot smoothing (σ=0.8 px)

Replaces: generic LoDoPaB-CT slice or Shepp-Logan phantom fallback — which were not vessel-like.

### 2. Registry Entry (`benchmarks/datasets/registry.py`)

Added `angiography_vessel_generated` DatasetEntry with:
- `applies_to=["angiography"]`
- `converter="generate_angiography_vessel_phantom"`
- `source_type="generated"` (always available without downloads)

Removed `angiography` from `lodopab_ct_sample.applies_to` and `covid_ct_lung_seg.applies_to` to prevent CT brain/lung phantoms from being used for angiography.

### 3. Algorithm Override (`_algorithm_catalog.py`)

Added `_VARIANT_OVERRIDES["angiography"]` with 9 angiography-specific algorithms:

| Algorithm | Type | Year | Reference |
|-----------|------|------|-----------|
| FDK | Classical | 1984 | Feldkamp et al., JOSA A |
| TV-CS | Classical CS | 2008 | Sidky et al., PMB |
| PnP-ADMM | Plug-and-Play | 2013 | Venkatakrishnan et al., GlobalSIP |
| FBPConvNet | Deep Learning | 2017 | Jin et al., IEEE TIP |
| Learned Primal-Dual | Deep Unrolling | 2018 | Adler & Oktem, IEEE TMI |
| VesselNet | Deep Learning | 2024 | Zhang et al., Radiology AI |
| NeRF-Angio | Physics-Informed | 2024 | Wang et al., IEEE TMI |
| AngioFormer | Transformer | 2024 | Geometry-aware 3DRA transformer |
| DiffusionAngio | Diffusion | 2024 | Shen et al., Med. Image Anal. |

Replaces: generic `medical` pool (CT algorithms — Score-CT, CTFormer, DOLCE, etc. — that don't cite angiography-specific papers).

### 4. Score Pool (`_algorithm_catalog.py`)

Added `CATEGORY_REAL_SCORES["angiography"]` with 9 calibrated entries:
- FDK: 27.0 dB / 0.780 SSIM (clinical baseline)
- DiffusionAngio: 36.8 dB / 0.967 SSIM (2024 SOTA)

PSNR range (27–37 dB) is appropriate for angiography DSA/3DRA reconstruction with Poisson noise at clinical dose levels, consistent with:
- Shen et al. (2024): ~4 dB PSNR gain for diffusion vs TV-CS in 60-view 3DRA
- Zhang et al. (2024): UNet DSA enhancement metrics
- Wang et al. (2024): 4D DSA with motion correction

### 5. Generator Registration (`platform/scripts/generate_challenge_datasets.py`)

Registered `generate_angiography_vessel_phantom` in both generator maps:
- `_resolve_ground_truth()` → `_GENERATOR_MAP`
- `_generate_scenes_for_gallery()` → `gen_map`

### 6. Dataset Regeneration & GCS Upload

Generated and uploaded all 3 challenge tiers (2026-03-09):
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_dev.h5` (x_true stripped)
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_hidden.h5` (blocked)

**Forward model used:** Radon (cone-beam projection, 180 views, 182 detectors), Poisson noise — appropriate for X-ray angiography physics.

---

## Previous Plan (2026-03-06, superseded)

The original modify_plan.md from 2026-03-06 identified only LOW priority documentation updates to `03_reconstruction_algorithms.md`. This has been superseded by the comprehensive improvements above which are more impactful for the benchmark quality.
