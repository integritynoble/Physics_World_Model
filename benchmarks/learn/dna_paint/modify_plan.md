# Modify Plan: dna_paint

## Current State (After Fix)

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** smlm (via `_VARIANT_SCORE_ALIASES`)
- **Algorithms:** ThunderSTORM, FALCON, Deep-STORM, DECODE (SMLM-specific override)
- **Runner type:** psf
- **Signal shape:** [256, 256]

## Assessment

Algorithms are now domain-appropriate (SMLM localization methods, not deconvolution).
Dataset uses PSF forward model which is acceptable for the benchmark.

## Verdict

No further code changes needed. Algorithm override implemented in previous session.
