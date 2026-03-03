# Modify Plan -- light_field

## Current State (Updated 2026-03-03)

- **Category:** computational
- **Carrier:** Photon
- **Score key:** computational
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["light_field"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Shift-and-Sum (Classical) -- Ng et al., Stanford Tech Report 2005
  2. PnP-LF (PnP) -- PnP-ADMM with angular prior
  3. LFNet (Deep Learning) -- Wang et al., IEEE TPAMI 2020
  4. DistgSSR (Transformer) -- Wang et al., CVPR 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic computational pool (Tikhonov,
PnP-RED, DIP, SwinIR) with light-field-specific algorithms. Both `light_field`
and `integral` share the same plenoptic physics and use similar algorithm sets:
`light_field` uses LFNet (TPAMI 2020) while `integral` uses LFAttNet (TIP
2020). Both share DistgSSR as the transformer method.

## Changes Applied

- Added `_VARIANT_OVERRIDES["light_field"]` with four light-field-specific algorithms
- Shift-and-Sum: fundamental light field refocusing baseline
- PnP-LF: plug-and-play with angular consistency priors
- LFNet: deep learning for light field view synthesis and angular SR
- DistgSSR: disentangled spatial-angular super-resolution transformer

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
