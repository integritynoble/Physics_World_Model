# Modify Plan -- hyperspectral_remote

## Current State (Updated 2026-03-03)

- **Category:** remote_sensing
- **Carrier:** Photon
- **Score key:** computational (routed via `_CARRIER_ROUTING[("remote_sensing", "Photon")]` -> `"computational"`)
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["hyperspectral_remote"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. CNMF (Classical) -- Yokoya et al., IEEE TGRS 2012
  2. PnP-LTTR (PnP) -- He et al., IEEE TGRS 2020
  3. DBIN (Deep Learning) -- Dong et al., CVPR 2021
  4. MST++ (Transformer) -- Cai et al., CVPRW 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic computational pool (Tikhonov,
PnP-RED, DIP, SwinIR) with hyperspectral-specific algorithms. The previous
set had no spectral awareness; the new set includes spectral unmixing (CNMF),
tensor-based reconstruction (PnP-LTTR), spectral CNN (DBIN), and the NTIRE
2022 challenge winner (MST++).

## Changes Applied

- Added `_VARIANT_OVERRIDES["hyperspectral_remote"]` with four spectral-specific algorithms
- CNMF: coupled nonnegative matrix factorization for spectral fusion
- PnP-LTTR: low-tensor-train-rank with plug-and-play priors
- DBIN: deep blind image network for spectral reconstruction
- MST++: mask-guided spectral-wise transformer (NTIRE 2022 winner)

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
