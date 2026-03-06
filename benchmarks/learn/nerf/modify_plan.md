# Modify Plan: nerf (Neural Radiance Fields)

**Date:** 2026-03-06

## Current State

- **Category:** neural_rendering
- **Carrier:** Photon
- **Score key:** neural_rendering (direct category match)
- **Algorithms served (4):**
  1. COLMAP+MVS (Classical) -- Schonberger & Frahm, CVPR 2016
  2. Mip-NeRF 360 (Neural NeRF variant) -- Barron et al., CVPR 2022
  3. Instant-NGP (Neural hash-grid NeRF) -- Muller et al., SIGGRAPH 2022
  4. 3D-GS (Gaussian Splatting) -- Kerbl et al., SIGGRAPH 2023

## Assessment

**Excellent — best algorithm set in the entire benchmark.**

All four algorithms are the canonical published methods for novel view synthesis and neural 3D reconstruction:

- **COLMAP+MVS**: Schonberger & Frahm, CVPR 2016 is the standard classical multi-view stereo baseline. Used as the ground-truth pose estimator for virtually all NeRF methods. CORRECT.
- **Mip-NeRF 360**: Barron et al., CVPR 2022 — leading NeRF variant for unbounded scenes with multi-scale representation. CORRECT.
- **Instant-NGP**: Muller et al., SIGGRAPH 2022 — hash-grid acceleration for real-time NeRF training (100× speedup). Industry standard. CORRECT.
- **3D Gaussian Splatting**: Kerbl et al., SIGGRAPH 2023 — current state-of-the-art for real-time rendering quality, outperforming NeRF on most benchmarks as of 2024. CORRECT.

### Citation Verification

- COLMAP+MVS: Schonberger & Frahm, CVPR 2016 — correct
- Mip-NeRF 360: Barron et al., CVPR 2022 — correct
- Instant-NGP: Muller et al., SIGGRAPH 2022 — correct (full ref: Müller et al., ACM TOG 41(4), 2022)
- 3D-GS: Kerbl et al., SIGGRAPH 2023 — correct (full ref: ACM TOG 42(4), 2023)

### Minor Taxonomy Note

The algorithm "type" labels are unconventional: Mip-NeRF 360 labeled "PnP" and 3D-GS labeled "Transformer". Neither is technically correct for neural rendering methods. However, this is a cosmetic taxonomy issue — the algorithm names and citations are correct. Neural rendering methods do not fit neatly into the classical/PnP/DL/Transformer taxonomy used for physics-based imaging.

## Verdict

No code changes needed.

**Priority:** NONE — algorithms are ideal. Optional: correct "type" labels to more appropriate descriptors ("Implicit NeRF" for Mip-NeRF 360, "Explicit Gaussian" for 3D-GS).
