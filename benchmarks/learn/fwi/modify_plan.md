# Modify Plan: fwi

## Current State (Updated 2026-03-03)

- **Category:** experimental_science
- **Carrier:** Seismic/Acoustic
- **Score key:** experimental_science
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["fwi"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. L-BFGS FWI (Classical) -- Virieux & Operto, Geophysics 2009
  2. TV-Reg FWI (Classical) -- Esser et al., Geophysics 2018
  3. InversionNet (Deep Learning) -- Wu & Lin, JGR 2019
  4. VelocityGAN (Deep Learning) -- Zhang & Lin, JGR 2020

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic experimental_science pool (Tikhonov,
PnP-RED, ResUNet, SwinIR) with FWI-specific algorithms. All four methods are
well-cited, domain-appropriate, and cover the classical-to-deep-learning
spectrum for seismic velocity inversion.

## Changes Applied

- Added `_VARIANT_OVERRIDES["fwi"]` with four FWI-specific algorithms
- L-BFGS FWI: standard gradient-based waveform misfit optimizer
- TV-Reg FWI: total-variation regularized FWI for sharp boundaries
- InversionNet: direct CNN mapping from seismograms to velocity
- VelocityGAN: adversarial training for velocity estimation

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
