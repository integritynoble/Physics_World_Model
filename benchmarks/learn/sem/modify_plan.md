# Modify Plan: SEM (Scanning Electron Microscopy)

**Created:** 2026-03-03
**Status:** Done (fixed via carrier-based routing)

## Changes

SEM was previously getting cryo-EM particle reconstruction algorithms (RELION, cryoSPARC)
via the "electron_microscopy" category. These are wrong for SEM which needs image
denoising/restoration, not single-particle reconstruction.

Fixed by special EM routing: non-cryo EM variants → em_generic pool.

Now correctly shows: Wiener Filter, BM3D, Noise2Void, SwinIR.

No additional changes needed.
