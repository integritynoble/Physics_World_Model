# Modify Plan: Weather / Doppler Radar

**Created:** 2026-03-03
**Status:** Algorithms are a significant mismatch -- SAR methods assigned to weather radar

## Assessment

Weather radar falls under `remote_sensing` category with carrier `RF`. It receives:

- Matched Filter (Classical) -- "Standard SAR focusing"
- SAR-BM3D (PnP) -- SAR speckle denoising (Parrilli et al., IEEE TGRS 2012)
- SAR-DRN (Deep Learning) -- SAR despeckling (Zhang et al., RS 2018)
- SAR-CAM (Transformer) -- cross-attention SAR (2024)

### Issue

All four algorithms are **SAR-specific** (Synthetic Aperture Radar). Weather/Doppler radar is fundamentally different:

- SAR produces high-resolution ground images via aperture synthesis from a moving platform
- Weather radar measures volumetric precipitation reflectivity, Doppler velocity, and spectrum width from a rotating antenna

Weather radar reconstruction tasks include:
- Clutter filtering (ground clutter, AP removal)
- Velocity dealiasing (Doppler ambiguity resolution)
- Reflectivity mosaicing and QPE (quantitative precipitation estimation)
- Dual-pol variable estimation (ZDR, KDP, rho_HV)
- Attenuation correction

Appropriate algorithms would be:
- Classical: Pulse-pair processing / FFT-based Doppler estimation
- PnP/Regularized: Variational clutter suppression
- Deep Learning: RainNet (Ayzel et al., 2020), U-Net for radar echo extrapolation
- Transformer: Nowcasting transformers (e.g., Earthformer, Bi et al., NeurIPS 2023)

SAR-BM3D and SAR-DRN address SAR speckle, which has different statistical properties (multiplicative Rayleigh/K-distribution noise) from weather radar noise (thermal noise + clutter + range sidelobes).

### Decision

Weather radar should not share the SAR algorithm pool. It needs either a variant override or carrier-based routing to a weather radar / Doppler processing pool.

## Deferred Items

1. **HIGH PRIORITY**: Add `weather_radar` to `_VARIANT_OVERRIDES` with weather-radar-appropriate algorithms:
   - Classical: Pulse-pair Doppler estimator or CLEAN-AP clutter filter
   - PnP: Variational reflectivity restoration
   - Deep Learning: RainNet or U-Net radar denoiser
   - Transformer: Earthformer or radar nowcasting transformer
2. **Score key**: Would need a `weather_radar` entry in `CATEGORY_REAL_SCORES` since SAR PSNR/SSIM values do not transfer.

No code changes made in this pass, but this is a significant domain mismatch.
