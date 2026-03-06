# Modify Plan: fluoroscopy

**Date:** 2026-03-06

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical (CT-like pool, no carrier routing override for X-ray)
- **Algorithms assigned:**
  1. FBP (Classical) -- Analytical baseline
  2. TV-ADMM (Compressed Sensing) -- Rudin et al., Physica D 60, 259 (1992) + ADMM
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 26, 4509 (2017)
  4. RED-CNN (Deep Learning) -- Chen et al., IEEE TMI 36, 2524 (2017)

## Assessment

**Appropriate: YES**

Fluoroscopy is real-time 2D X-ray projection imaging. It shares the same X-ray projection physics as CT (Beer-Lambert attenuation along ray paths). The algorithms assigned are standard X-ray/CT reconstruction/enhancement methods:

- **FBP:** Analytical baseline for X-ray denoising/deconvolution. Acceptable as baseline.
- **TV-ADMM:** Total variation denoising applicable directly to low-dose fluoroscopic frames. GOOD FIT.
- **FBPConvNet:** Jin et al., IEEE TIP 2017 is a real CT reconstruction paper. Applicable to fluoroscopy as a 2D post-processing CNN.
- **RED-CNN:** Chen et al., IEEE TMI 2017 designed specifically for low-dose X-ray imaging (CT). Directly applicable to low-dose fluoroscopy enhancement.

The carrier routing does not override medical + X-ray, so it falls through to the default "medical" pool — perfectly appropriate for fluoroscopy.

RED-CNN (Chen et al., 2017) is a better choice than Learned Primal-Dual for fluoroscopy because RED-CNN was specifically developed for low-dose X-ray (single 2D image denoising), while Learned Primal-Dual is a projection-to-volume network best suited for 3D CT reconstruction.

## Code Changes Needed

No code changes needed.

**Priority:** NONE — no changes needed.
