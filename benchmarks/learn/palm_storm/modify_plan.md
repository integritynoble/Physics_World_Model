# Modify Plan: palm_storm (PALM/STORM Single-Molecule Localization)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy (with SMLM variant override)
- **Algorithms served (4):**
  1. ThunderSTORM (Classical) -- Ovesny et al., Bioinformatics 2014
  2. FALCON (PnP) -- Min et al., Sci. Rep. 2014
  3. Deep-STORM (Deep Learning) -- Nehme et al., Optica 2018
  4. DECODE (Deep Learning) -- Speiser et al., Nat. Methods 2021

## Assessment

**Correct.** The SMLM-specific override provides domain-appropriate single-molecule
localization algorithms. The previous assignment (generic microscopy deconvolution:
Richardson-Lucy, PnP-FISTA, CARE, Restormer) was inappropriate because PALM/STORM
requires localization of individual fluorophore blinking events, not image deconvolution.

The current algorithms are all domain-appropriate:
- ThunderSTORM is the gold-standard classical SMLM localization tool
- FALCON provides fast localization with deconvolution-based prior
- Deep-STORM is a CNN for dense emitter localization
- DECODE is the state-of-the-art probabilistic SMLM method

## Verdict

**PASS -- no code changes needed.** The SMLM override correctly provides
localization-specific algorithms for PALM/STORM.

## Recommended Changes

None required. Optional future additions: ANNA-PALM (accelerated reconstruction)
or FP-INR (implicit neural representation) as additional algorithm entries.
