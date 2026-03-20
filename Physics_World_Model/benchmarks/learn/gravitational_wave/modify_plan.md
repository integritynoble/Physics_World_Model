# Modify Plan: gravitational_wave

## Current State (Updated 2026-03-03)

- **Category:** experimental_science
- **Carrier:** Gravitational
- **Score key:** experimental_science
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["gravitational_wave"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Matched Filter (Classical) -- Allen et al., Phys. Rev. D 2012
  2. BayesWave (PnP) -- Cornish & Littenberg, CQG 2015
  3. GW-CNN (Deep Learning) -- George & Huerta, Phys. Rev. D 2018
  4. WaveFormer (Transformer) -- GW detection transformer, 2024

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the completely inappropriate generic
experimental_science pool (Tikhonov, PnP-RED, ResUNet, SwinIR) with GW-specific
algorithms. The previous set treated 1D time-series strain data as 2D images,
which was fundamentally wrong. All four replacements are standard GW detection
and signal extraction methods.

## Changes Applied

- Added `_VARIANT_OVERRIDES["gravitational_wave"]` with four GW-specific algorithms
- Matched Filter: LIGO/Virgo standard template-based detection pipeline
- BayesWave: Bayesian wavelet-based unmodeled transient analysis
- GW-CNN: deep learning detection from raw strain data
- WaveFormer: transformer-based GW detection and parameter estimation

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
