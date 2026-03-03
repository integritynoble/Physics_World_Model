# Modify Plan: stm (Scanning Tunneling Microscopy)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** scanning_probe
- **Carrier:** Electron
- **Score key:** scanning_probe
- **Algorithms served (4):**
  1. BTR (Classical) -- Villarrubia, JRNIST 1997
  2. Reg-Deconv (PnP) -- Dongmo et al., 2000
  3. DeepSPM (Deep Learning) -- Alldritt et al., Commun. Phys. 2020
  4. E2E-BTR (Deep Learning) -- Kossler et al., Sci. Rep. 2022

## Assessment

**Acceptable.** The scanning_probe pool is designed primarily for AFM tip
deconvolution, but the algorithms are applicable to STM:

- Both AFM and STM share the tip-artifact convolution problem
- BTR (Blind Tip Reconstruction) addresses geometric tip broadening, which
  occurs in both AFM (dominant) and STM (less dominant but present)
- DeepSPM (Alldritt et al., 2020) is specifically an STM algorithm for
  molecular identification from STM images -- already correctly included
- E2E-BTR provides an end-to-end learned approach for tip deconvolution
- Reg-Deconv provides regularized deconvolution as a PnP baseline

While STM has unique challenges (LDOS convolution, electronic tip structure)
not fully captured by the AFM-oriented pool, the shared tip-artifact framework
makes this assignment acceptable. The presence of DeepSPM provides STM-specific
coverage.

## Verdict

**PASS -- no code changes needed.** The scanning_probe pool provides acceptable
algorithms for STM. The tip deconvolution framework is shared between AFM and STM,
and DeepSPM provides STM-specific domain coverage.

## Recommended Changes

None required. Optional future enhancement: add an STM-specific override with
drift correction and STS-deconv algorithms, but this is not necessary for correct
benchmark operation.
