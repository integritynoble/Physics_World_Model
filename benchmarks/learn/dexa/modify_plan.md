# Modify Plan: dexa

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Routing:** No carrier routing override for (medical, X-ray), so falls through to `_CATEGORY_ALGORITHMS["medical"]`.
- **Score key:** medical
- **Algorithms served:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Problem

The `medical` pool contains CT tomographic reconstruction algorithms. DEXA (Dual-Energy X-ray Absorptiometry) is NOT a tomographic technique -- it is a 2D projection-based method that measures bone mineral density by comparing X-ray attenuation at two different energies. The reconstruction problem is material decomposition from dual-energy projections, not volumetric tomographic reconstruction.

- **FBP (Filtered Back-Projection):** Tomographic reconstruction from sinograms. DEXA does not produce sinograms; it acquires 2D projections at two energies. WRONG.
- **PnP-ADMM:** Used here in a CT reconstruction context. The framework could apply to DEXA but the CT-specific implementation is wrong.
- **FBPConvNet:** Post-processing network for FBP-reconstructed CT images. DEXA has no FBP step. WRONG.
- **Learned Primal-Dual:** Unrolled CT reconstruction network operating on sinograms. WRONG.

## Recommended Algorithms

DEXA reconstruction involves dual-energy material decomposition: separating soft tissue and bone contributions from two energy-dependent projection images.

| Slot | Algorithm | Type | Reference | Rationale |
|------|-----------|------|-----------|-----------|
| Classical | Dual-Energy Subtraction (DES) | Classical | Lehmann et al., Med. Phys. 1981 | Standard log-subtraction method for dual-energy decomposition -- the foundational DEXA algorithm |
| PnP | PnP-ADMM (decomposition) | PnP | Adapted from Venkatakrishnan et al., 2013 | Plug-and-play with material decomposition forward model and image-domain denoising prior |
| Deep Learning | Butterfly-Net | Deep Learning | Long et al., Phys. Med. Biol. 2021 | Deep learning for dual-energy material decomposition with noise suppression |
| Deep Unrolling | DECT-MULTRA | Deep Unrolling | Gong et al., IEEE TMI 2020 | Model-based deep learning for multi-material decomposition from dual-energy data |

## Required Code Changes

1. **`_algorithm_catalog.py`:** Add `dexa` to `_VARIANT_OVERRIDES` with dual-energy decomposition algorithms.
2. **`_algorithm_catalog.py`:** Add DEXA-specific real scores to `CATEGORY_REAL_SCORES`.
3. **Consider:** Adding a carrier routing rule `("medical", "X-ray"): "medical"` would not help here since the issue is that DEXA is lumped with CT. A variant override is the correct approach.
