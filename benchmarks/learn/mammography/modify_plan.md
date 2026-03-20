# Modify Plan -- mammography

**Date:** 2026-03-06

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical
- **Algorithms (from catalog):**
  1. FBP (Classical) -- Feldkamp et al., JOSA A 1, 612 (1984)
  2. TV-ADMM (Compressed Sensing) -- Sidky & Pan, Phys. Med. Biol. 53, 4777 (2008)
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 26, 4509 (2017)
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 37, 1322 (2018)

## Assessment

The algorithms are **appropriate** for mammography, particularly for the Digital Breast Tomosynthesis (DBT) framing.

- **FBP (FDK)**: The clinical standard for DBT reconstruction. Feldkamp et al. 1984 is correct. Appropriate.
- **TV-ADMM**: Sidky & Pan 2008 is the landmark paper for TV reconstruction in limited-angle CT (directly applicable to DBT's ±25° arc). GOOD FIT.
- **FBPConvNet**: Jin et al., IEEE TIP 2017 — CNN post-processing for CT. Applicable to DBT slice enhancement. ACCEPTABLE.
- **Learned Primal-Dual**: Adler & Oktem, IEEE TMI 2018 — deep unrolling for projection CT. Applicable to DBT projection-to-volume reconstruction. GOOD FIT.

The medical/X-ray routing gives mammography the same CT-focused algorithm set. This is appropriate because mammography (especially DBT) shares the X-ray projection physics with CT. The algorithms are all real, published, and properly cited.

### DBT vs 2D Mammography

The benchmark is best interpreted as a DBT (3D reconstruction) problem rather than 2D mammography (single projection), because:
1. FBP, TV-ADMM, and Learned Primal-Dual are projection-based reconstruction algorithms suited for DBT
2. The DBT formulation enables meaningful algorithmic differentiation
3. DBT is the dominant clinical form of modern mammography (2013 FDA approval)

For 2D mammography framing (denoising/scatter correction), RED-CNN would be more appropriate than Learned Primal-Dual. This is a minor consideration.

## Verdict

No code changes needed.

**Priority:** NONE — no changes needed.
