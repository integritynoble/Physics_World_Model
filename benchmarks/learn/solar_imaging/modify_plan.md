# Modify Plan: solar_imaging

## Current State
- **Category:** astronomy
- **Carrier:** Photon/EUV
- **Score key:** astronomy
- **Algorithms:**
  1. CLEAN (Classical) -- Hogbom, A&AS 1974
  2. AIRI (PnP) -- Terris et al., MNRAS 2022
  3. R2D2 (Deep Learning) -- Aghabiglou et al., ApJS 2024
  4. PRIMO (Deep Learning) -- Medeiros et al., ApJL 2023

## Assessment

**Problem:** The astronomy pool contains radio interferometry algorithms (CLEAN, AIRI, R2D2, PRIMO) designed for aperture synthesis from visibility data (u-v plane). Solar EUV/X-ray imaging uses direct imaging with grazing-incidence telescopes (SDO/AIA, SOHO/EIT, Hinode/XRT), not radio interferometry. The reconstruction problem is image deconvolution/enhancement from direct CCD images, not visibility-to-image inversion.

- **CLEAN** is for VLBI/radio interferometric imaging -- not applicable to solar EUV/X-ray direct imaging.
- **AIRI** is a PnP method for radio interferometric imaging -- same issue.
- **R2D2** is a learned residual-to-residual method for radio imaging -- same issue.
- **PRIMO** reconstructed the M87 black hole from EHT radio data -- same issue.

**Appropriate algorithms for solar EUV/X-ray imaging:**
1. Richardson-Lucy (Classical) -- standard deconvolution for solar telescope PSF, used in SDO pipeline
2. Pixon (Classical/Regularized) -- Pina & Puetter, 1993; used extensively in solar X-ray reconstruction (RHESSI)
3. MEM-Sato / Maximum Entropy (Classical) -- Sato et al., used in solar hard X-ray imaging
4. Deep image denoising (e.g., CARE or Noise2Clean adapted for solar) -- emerging DL approaches

## Required Changes

Add a `_CARRIER_ROUTING` entry or a `_VARIANT_OVERRIDES` entry for `solar_imaging` in `_algorithm_catalog.py` to route away from the radio-interferometry astronomy pool. Options:

1. **Option A (preferred):** Add `solar_imaging` to `_VARIANT_OVERRIDES` with domain-appropriate algorithms:
   - Richardson-Lucy (Classical) -- PSF deconvolution baseline
   - Pixon (Regularized) -- Pina & Puetter, PASP 1993
   - PnP-RED (PnP) -- generic PnP denoiser, applicable to direct imaging
   - SolarNet or CARE-Solar (Deep Learning) -- DL deconvolution for solar imaging

2. **Option B:** Add carrier routing `("astronomy", "Photon/EUV"): "computational"` to map to generic deconvolution pool (Tikhonov/PnP-RED/DIP/SwinIR), which is more appropriate than radio methods but still not solar-specific.

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add override or routing entry
