# Modify Plan: fib_sem

## Current Assignment
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic (not in `_CRYO_EM_VARIANTS`)
- **Algorithms:** Wiener Filter (Classical), BM3D (PnP), Noise2Void (Deep Learning), SwinIR (Transformer)

## Assessment

The algorithm assignment is **acceptable**. FIB-SEM (Focused Ion Beam Scanning
Electron Microscopy) produces serial-section image stacks by alternating ion
beam milling and SEM imaging. The primary reconstruction challenges are:

1. **Denoising** individual SEM frames (shot noise, charging artifacts)
2. **Slice-to-slice alignment** for 3D volume reconstruction
3. **Isotropic resolution recovery** (axial resolution is limited by slice
   thickness)

The em_generic pool provides appropriate denoising algorithms:

- **Wiener Filter** is a reasonable classical denoising baseline.
- **BM3D** is widely used for SEM image denoising.
- **Noise2Void** (Krull et al., CVPR 2019) is directly applicable to EM
  denoising where paired training data is unavailable.
- **SwinIR** is a strong general-purpose restoration transformer.

**Minor concern:** The check.md shows RELION/cryoSPARC/IsoNet/CryoTransformer
on the live leaderboard, suggesting the deployed code may differ from the
current codebase. The current code correctly routes to em_generic, which is
the better assignment. One could argue for adding FIB-SEM-specific tools like
IsoNet (Liu et al., Nat. Commun. 2022) for missing-wedge compensation, but
the current generic EM pool is defensible for the denoising task.

## Verdict

No code changes needed. The current em_generic pool is appropriate for the
FIB-SEM denoising/restoration task. If desired, a future enhancement could add
IsoNet to the pool as a domain-specific deep learning method.
