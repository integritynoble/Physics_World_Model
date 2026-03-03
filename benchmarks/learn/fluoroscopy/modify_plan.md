# Modify Plan: fluoroscopy

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical (CT-like pool, no carrier routing override for X-ray)
- **Algorithms assigned:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

**Appropriate: YES**

Fluoroscopy is real-time 2D X-ray projection imaging. It shares the same X-ray
projection physics as CT (line integrals through tissue, Beer-Lambert law). The
algorithms assigned -- FBP, PnP-ADMM, FBPConvNet, Learned Primal-Dual -- are
all standard X-ray/CT reconstruction methods that apply directly to fluoroscopy
image reconstruction (e.g., sparse-view or low-dose fluoroscopic frame
reconstruction). These are well-cited, real algorithms used in published X-ray
imaging benchmarks.

The carrier routing does not override medical + X-ray, so it falls through to
the default "medical" pool which is CT-centric -- perfectly appropriate for
fluoroscopy.

## Code Changes Needed

No code changes needed.
