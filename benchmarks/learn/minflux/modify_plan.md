# Modify Plan — minflux

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms (from catalog):**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022
- **Leaderboard (live):** Richardson-Lucy, PnP-FISTA, CARE, Restormer (4 entries)

## Assessment

The algorithms are **partially inappropriate** for MINFLUX nanoscopy.

MINFLUX is a single-molecule localization nanoscopy technique (Balzarotti et al., Science 2017). Its reconstruction problem is fundamentally different from widefield/confocal microscopy deconvolution:

- **Richardson-Lucy** is a PSF deconvolution algorithm. MINFLUX does not need PSF deconvolution -- it directly estimates molecule positions from photon counts in a patterned excitation beam. The reconstruction is a **localization** problem (estimate x,y,z coordinates from photon ratios), not a deconvolution problem.
- **PnP-FISTA** assumes a linear forward model with image-domain priors. MINFLUX localization is a nonlinear parameter estimation problem.
- **CARE** is for denoising/restoring conventional microscopy images. MINFLUX data is a sparse set of photon counts per excitation position, not an image to denoise.
- **Restormer** is an image restoration transformer -- same issue as CARE.

The microscopy category algorithms are designed for widefield/confocal/lightsheet image reconstruction. MINFLUX is a localization microscopy method, more akin to PALM/STORM. The check.md file confirms the live page shows DECODE, ANNA-PALM, SPARCOM, and SOFI -- these are from the leaderboard generator's per-variant name randomization but still point to localization methods.

Wait -- looking at the check.md more carefully: "Methods: DECODE + gradient, ANNA-PALM + gradient, SPARCOM + gradient, SOFI + gradient". But the catalog returns Richardson-Lucy, PnP-FISTA, CARE, Restormer. This suggests the live page may be using different method name generation than the catalog algorithms. Let me verify...

Actually, the check.md was generated at a specific date and the leaderboard DB confirms Richardson-Lucy, PnP-FISTA, CARE, Restormer are what the system actually uses. The check.md methods may reflect an earlier state.

## Recommended Changes

1. **Add a variant override** for `minflux` (and potentially other localization microscopy: `palm_storm`, `dna_paint`) in `_algorithm_catalog.py`:
   - Classical: **MLE Localization** (Thompson et al., Biophys. J. 2002) -- maximum likelihood estimation of emitter positions
   - PnP: **SPARCOM** (Solomon et al., SIAM 2019) -- sparsity-based super-resolution
   - Deep Learning: **DECODE** (Speiser et al., Nat. Methods 2021) -- deep learning for single-molecule localization
   - Transformer: **ANNA-PALM** (Ouyang et al., Nat. Biotechnol. 2018) -- deep learning assisted localization microscopy

2. Alternatively, create a `localization_microscopy` sub-category.

## Verdict

**Changes recommended** -- MINFLUX is a localization nanoscopy technique, not a deconvolution microscopy method. The current Richardson-Lucy/CARE/Restormer algorithms solve a fundamentally different problem (image deconvolution) than what MINFLUX requires (single-molecule localization from photon statistics).
