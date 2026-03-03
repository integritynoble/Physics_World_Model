# Modify Plan: passive_microwave

## Current State
- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Algorithms:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

Passive microwave radiometry measures naturally emitted microwave radiation (brightness temperature) to retrieve geophysical parameters like soil moisture, sea surface temperature, or atmospheric water vapor. It is fundamentally different from SAR (Synthetic Aperture Radar), which is an active coherent imaging system.

The current algorithms are **entirely SAR-specific** (Matched Filter for SAR focusing, SAR-BM3D for speckle denoising, SAR-DRN, SAR-CAM). Passive microwave does not have SAR focusing or speckle. The reconstruction problem is an **aperture synthesis / deconvolution** problem (antenna pattern deconvolution) or a **retrieval** problem (brightness temperature to geophysical parameter inversion).

The check.md confirms the leaderboard shows SAR-specific algorithms, which is incorrect. The carrier routing `("remote_sensing", "RF") -> remote_sensing` keeps the SAR pool, but passive microwave should NOT use SAR algorithms.

**Appropriate passive microwave algorithms would be:**
- Backus-Gilbert (Classical) -- Backus & Gilbert, 1970 (aperture synthesis deconvolution)
- Tikhonov regularization (Classical) -- Standard regularized inversion
- PnP-BM3D (PnP) -- Generic denoising prior
- CNN-retrieval (Deep Learning) -- e.g., Turk et al., IEEE TGRS 2022

## Required Changes

Add a variant override or new carrier routing for `passive_microwave` in `_algorithm_catalog.py`. Since "RF" carrier maps to SAR algorithms by default but passive microwave is not SAR, the cleanest approach is to add `_VARIANT_OVERRIDES["passive_microwave"]` with aperture synthesis / radiometric inversion algorithms.

### Files to modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- add variant override for `passive_microwave` with radiometry-appropriate algorithms
