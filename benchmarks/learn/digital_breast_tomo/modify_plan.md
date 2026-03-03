# Modify Plan: digital_breast_tomo (Digital Breast Tomosynthesis)

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical (no carrier routing override for X-ray)
- **Algorithms served:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

Good match. Digital breast tomosynthesis is a limited-angle X-ray CT modality
that acquires projection data over a narrow angular range (typically 15-50
degrees). The reconstruction problem is fundamentally the same as sparse-view /
limited-angle CT:

- FBP is the standard analytical baseline (with limited-angle artifacts).
- PnP-ADMM applies plug-and-play denoising to iterative CT reconstruction.
- FBPConvNet post-processes FBP with a CNN to remove artifacts.
- Learned Primal-Dual jointly learns data-domain and image-domain updates.

All four algorithms are directly applicable to DBT. The only nuance is that
DBT-specific algorithms might additionally handle the limited-angle geometry
more explicitly (e.g., directional regularization), but the CT pool captures
the core reconstruction challenge well.

## Verdict

No code changes needed.
