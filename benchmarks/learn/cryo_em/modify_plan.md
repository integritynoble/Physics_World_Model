# Modify Plan: cryo_em

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** Electron
- **Routing:** No carrier routing match for (scientific_instrumentation, Electron). Falls through to `_CATEGORY_ALGORITHMS["scientific_instrumentation"]`.
- **Score key:** scientific_instrumentation
- **Algorithms served:**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024

## Problem

cryo_em is listed in `_CRYO_EM_VARIANTS = {"cryo_em", "cryo_et", "electron_tomography", "electron_diffraction"}`, but this set is only checked when `category == "electron_microscopy"`. Since cryo_em has `category: "scientific_instrumentation"` in the modality catalog, the cryo-EM routing is never triggered.

The `scientific_instrumentation` pool contains generic instrument calibration algorithms (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) that are inappropriate for cryo-EM single particle analysis:

- **Deconv:** Generic deconvolution. Too vague -- cryo-EM uses CTF correction (Wiener filtering), not generic deconvolution. POOR FIT.
- **PnP-BM3D:** Generic denoiser. Not a cryo-EM method. POOR FIT.
- **ResNet-Calib:** Instrument calibration CNN. Not relevant to cryo-EM reconstruction. WRONG.
- **CalibFormer:** Instrument calibration transformer. Not relevant. WRONG.

The correct pool (electron_microscopy: RELION, cryoSPARC, cryoDRGN, CryoTransformer) exists but is unreachable due to the category mismatch.

## Root Cause

The modality catalog assigns `cryo_em` to `category: "scientific_instrumentation"` instead of `category: "electron_microscopy"`. This prevents the `_CRYO_EM_VARIANTS` routing from activating.

## Required Code Changes

**Option A (preferred -- fix category):**
1. **`_modality_catalog.py`:** Change `cryo_em` category from `"scientific_instrumentation"` to `"electron_microscopy"`.
   - This makes the `_CRYO_EM_VARIANTS` check in `get_algorithms()` activate correctly.
   - cryo_em will then get: RELION, cryoSPARC, cryoDRGN, CryoTransformer.

**Option B (fix routing):**
1. **`_algorithm_catalog.py`:** Add `("scientific_instrumentation", "Electron")` to `_CARRIER_ROUTING` pointing to `"electron_microscopy"`.
   - OR: Extend the `_CRYO_EM_VARIANTS` check to also work when category is `"scientific_instrumentation"`.

**Option A is simpler and more correct** -- cryo-EM SPA is unambiguously an electron microscopy technique.
