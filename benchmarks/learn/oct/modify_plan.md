# Modify Plan: OCT (Optical Coherence Tomography)

**Created:** 2026-03-03
**Status:** Done (fixed via carrier-based routing)

## Changes

OCT was previously getting CT algorithms (FBP, FBPConvNet) via the generic "medical" category.
Fixed by carrier-based routing: (medical, Photon) → clinical_optics pool.

Now correctly shows: FFT-OCT, BM4D, Speckle-DenoiseNet, OCTA-Net.

No additional changes needed.
