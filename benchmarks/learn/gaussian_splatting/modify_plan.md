# Modify Plan: gaussian_splatting

## Current State

- **Category:** neural_rendering
- **Carrier:** Photon
- **Score key:** neural_rendering
- **Algorithms assigned:**
  1. COLMAP+MVS (Classical) -- Schonberger & Frahm, CVPR 2016
  2. Mip-NeRF 360 (PnP) -- Barron et al., CVPR 2022
  3. Instant-NGP (Deep Learning) -- Muller et al., SIGGRAPH 2022
  4. 3D-GS (Transformer) -- Kerbl et al., SIGGRAPH 2023

## Assessment

**Appropriate: YES**

3D Gaussian Splatting is a neural rendering method. The neural_rendering pool
contains exactly the right set of algorithms for comparison:

- **COLMAP+MVS**: The standard classical multi-view stereo baseline that all
  neural rendering methods compare against.
- **Mip-NeRF 360**: A top NeRF variant, the direct predecessor/competitor to
  3DGS. Correctly cited (Barron et al., CVPR 2022).
- **Instant-NGP**: The hash-grid accelerated NeRF from NVIDIA. Another key
  competitor/baseline (Muller et al., SIGGRAPH 2022).
- **3D-GS**: The original 3D Gaussian Splatting paper itself (Kerbl et al.,
  SIGGRAPH 2023).

All four are the exact algorithms used in published 3DGS benchmark comparisons.
The "type" labels (Classical, PnP, Deep Learning, Transformer) are approximate
for this domain but the algorithm selections are spot-on.

## Code Changes Needed

No code changes needed.
