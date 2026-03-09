# Modify Plan: ceus (Contrast-Enhanced Ultrasound)

**Updated:** 2026-03-09
**Status:** DONE — microbubble phantom, algorithm overrides, and GCS datasets added

## Changes Made (2026-03-09)

### 1. Phantom Generator (`benchmarks/datasets/downloaders.py`)
- Added `generate_ceus_phantom()` — liver vasculature phantom with portal vein, branching hepatic vessels, Rayleigh speckle tissue background, and sparse microbubble harmonic signal (Gaussian-smoothed).
- Ground truth: vessel perfusion map `x_true` (128×128).
- Measurement: combined B-mode + contrast image `y` (tissue speckle + 2× microbubble signal + Gaussian noise).
- Metadata: `n_branches`, `n_bubbles`, `contrast_agent` (SonoVue).
- References: Errico et al., Nature 2015; Lowerison et al., Nat. Commun. 2022.

### 2. Registry Entry (`benchmarks/datasets/registry.py`)
- Added `"ceus_microbubble_generated"` DatasetEntry with `applies_to=["ceus"]`.

### 3. Algorithm Overrides (`_algorithm_catalog.py` — `_VARIANT_OVERRIDES`)
- Added `"ceus"` with 9 hand-crafted algorithms spanning Classical → Diffusion:
  - Pulse-Inversion (Simpson 1999), AM-CEUS (Mor-Avi 2002)
  - CNN-Bubble (Youn 2020), ULM-Net (Christensen-Jeffries 2020), DeepULM (van Sloun 2021)
  - PINN-CEUS (Lowerison 2022), CEUSF-Transformer (Huang 2023), SUPER-ULM (Rigo 2023)
  - DiffusionCEUS (Chen 2024)

### 4. Benchmark Scores (`_algorithm_catalog.py` — `CATEGORY_REAL_SCORES`)
- Added `"ceus"` with 9 PSNR/SSIM entries (24.1–39.6 dB, 0.751–0.962 SSIM).
- Monotonically increasing with method sophistication.

### 5. Runner Routing (`generate_challenge_datasets.py`)
- Added `"ceus": "identity"` to `_VARIANT_TO_RUNNER`.
- Added `generate_ceus_phantom` to both import blocks and both generator maps.

### 6. Converter Maps (`benchmarks/datasets/downloaders.py`)
- Added `generate_ceus_phantom` to both `_generated_converters` and `converter_map` dicts.

### 7. GCS Datasets
- Generated and uploaded all 3 tiers to GCS:
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_hidden.h5`

## Verdict

DONE. CEUS now has dedicated per-variant algorithm overrides (replacing generic medical_ultrasound pool) and a purpose-built microbubble liver phantom. Challenge datasets live on GCS. All syntax validated.
