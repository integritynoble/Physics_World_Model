# Modify Plan -- radio_interferometry

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Routing:** Variant override -> astronomy algorithms (CLEAN, AIRI, R2D2, PRIMO)
- **Score key:** astronomy (via override)
- **Algorithms served (4):**
  1. CLEAN (Classical) -- Hogbom, A&AS 1974
  2. AIRI (PnP) -- Terris et al., MNRAS 2022
  3. R2D2 (Deep Learning) -- Aghabiglou et al., ApJS 2024
  4. PRIMO (Deep Learning) -- Medeiros et al., ApJL 2023

## Assessment

**Correct.** Radio interferometry was previously getting SAR algorithms (Matched Filter,
SAR-BM3D, SAR-DRN, SAR-CAM) via the `remote_sensing` category default. This was
fundamentally wrong -- VLBI reconstructs sky images from sparse Fourier-plane visibility
measurements, not SAR range-Doppler data.

The variant-level override was the correct fix approach because:
- `("remote_sensing", "RF")` carrier routing should still map to SAR for actual SAR modalities
- A variant override is more surgical and avoids side effects
- The astronomy algorithms (CLEAN, AIRI, R2D2, PRIMO) are the correct domain-specific methods

## Verdict

**PASS -- no code changes needed.** The variant override correctly provides
astronomy-specific algorithms for radio interferometry.

## Recommended Changes

None required. The fix has already been implemented and verified.
