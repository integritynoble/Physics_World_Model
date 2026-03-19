# Modify Plan -- integral

## Current State (Updated 2026-03-03)

- **Category:** computational
- **Carrier:** Photon
- **Score key:** computational
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["integral"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Shift-and-Add (Classical) -- Ng et al., Stanford Tech Report 2005
  2. PnP-LF (PnP) -- PnP-ADMM with LF prior
  3. LFAttNet (Deep Learning) -- Tsai et al., IEEE TIP 2020
  4. DistgSSR (Transformer) -- Wang et al., CVPR 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic computational pool (Tikhonov,
PnP-RED, DIP, SwinIR) with light-field-specific algorithms that exploit the
4D plenoptic structure of integral imaging data. The `integral` and
`light_field` overrides share similar algorithms (both are plenoptic systems)
with minor differences (LFAttNet vs. LFNet).

## Changes Applied

- Added `_VARIANT_OVERRIDES["integral"]` with four light-field-specific algorithms
- Shift-and-Add: fundamental plenoptic refocusing baseline
- PnP-LF: plug-and-play with angular/disparity priors
- LFAttNet: attention-based light field depth estimation and SR
- DistgSSR: disentangled spatial-angular super-resolution transformer

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
