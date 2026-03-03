# Modify Plan: spectral_ct (Spectral CT / Photon-Counting CT)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical (default CT pool)
- **Algorithms served (4):**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

**Acceptable.** Spectral CT gets the generic medical/CT pool. While these are
conventional CT reconstruction algorithms that do not exploit cross-energy
information, they are valid for the tomographic inverse-problem framework:

- FBP is the standard baseline for any CT problem (applicable per energy bin)
- PnP-ADMM is a general-purpose regularized reconstruction framework
- FBPConvNet and Learned Primal-Dual are proven CT methods
- The algorithms are not wrong -- they miss the opportunity for spectral-specific
  material decomposition but correctly solve the per-bin reconstruction problem

Spectral-specific algorithms (One-Step Spectral CT, Butterfly-Net, DECT-MULTRA)
would better represent the field but are not required for benchmark correctness.

## Verdict

**PASS -- no code changes needed.** The medical/CT pool is acceptable for spectral
CT. The algorithms correctly test tomographic reconstruction, even though they do
not exploit cross-energy correlations.

## Recommended Changes

None required. Optional future enhancement: add a `_VARIANT_OVERRIDES["spectral_ct"]`
entry with material-decomposition-aware algorithms for improved domain specificity.
