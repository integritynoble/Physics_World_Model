# Modify Plan: PET (Positron Emission Tomography)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** medical
- **Carrier:** Gamma
- **Score key:** particle_imaging (routed via `_CARRIER_ROUTING[("medical", "Gamma")]`)
- **Algorithms served (4):**
  1. OSEM (Classical) -- Hudson & Larkin, IEEE TMI 1994
  2. MAPEM-RDP (PnP) -- Nuyts et al., Phys. Med. Biol. 2002
  3. DeepPET (Deep Learning) -- Haggstrom et al., Med. Image Anal. 2019
  4. TransEM (Transformer) -- Xie et al., 2023

## Assessment

**Correct.** PET was previously getting CT algorithms (FBP, FBPConvNet) via the
generic "medical" category. This was fixed by carrier-based routing:
`(medical, Gamma) -> particle_imaging` pool.

The current algorithms are all domain-appropriate:
- OSEM is the universal clinical PET reconstruction standard
- MAPEM-RDP is the basis of commercial PET (GE Q.Clear/BSREM)
- DeepPET is an end-to-end DL PET reconstruction method
- TransEM applies transformers to emission tomography

## Verdict

**PASS -- no code changes needed.** The carrier routing correctly sends PET to the
particle_imaging pool with domain-appropriate algorithms.

## Recommended Changes

None required. The fix has already been implemented and verified.
