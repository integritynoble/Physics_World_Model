# Modify Plan -- minflux

## Current State (Updated 2026-03-03)

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["minflux"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. MLE Localization (Classical) -- Balzarotti et al., Science 2017
  2. SPARCOM (PnP) -- Solomon et al., SIAM J. Imaging Sci. 2019
  3. DECODE (Deep Learning) -- Speiser et al., Nat. Methods 2021
  4. ANNA-PALM (Deep Learning) -- Ouyang et al., Nat. Biotechnol. 2018

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic microscopy pool (Richardson-Lucy,
PnP-FISTA, CARE, Restormer) with single-molecule localization methods.
MINFLUX is a localization nanoscopy technique, not a deconvolution microscopy
method. The previous algorithms solved a fundamentally different problem
(image deconvolution) from what MINFLUX requires (molecule position estimation
from photon count statistics).

The `minflux` override shares the localization microscopy philosophy with
`palm_storm` and `dna_paint` overrides, but uses MLE Localization (specific
to the MINFLUX excitation pattern) instead of ThunderSTORM.

## Changes Applied

- Added `_VARIANT_OVERRIDES["minflux"]` with four localization-microscopy algorithms
- MLE Localization: maximum likelihood estimation from MINFLUX photon counts
- SPARCOM: sparsity-based super-resolution from emitter distributions
- DECODE: deep learning single-molecule localization (Nature Methods 2021)
- ANNA-PALM: neural network accelerated localization microscopy

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
