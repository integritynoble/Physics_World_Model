# Modify Plan: mr_fingerprinting

**Date:** 2026-03-06

## Current State (After Fix)

- **Category:** medical
- **Sub-category pool:** mrf (MRF-specific override via `_VARIANT_OVERRIDES`)
- **Score key:** mr_fingerprinting or mri_recon
- **Algorithms:**
  1. SVD-MRF (Classical) -- McGivney et al., IEEE TMI 33, 2370 (2014)
  2. MANTIS (Model-Based) -- Liu et al., MRM 82, 174 (2019)
  3. MRF-Net (Deep Learning) -- Cohen et al., MRM 80, 2056 (2018)
  4. MRF-Former (Transformer) -- Luo et al., IEEE TMI 42, 3403 (2023)

## Assessment

**Algorithms are now domain-appropriate.**

The previous generic MRI pool (Zero-Filled IFFT, L1-Wavelet/ESPIRiT, PnP-DnCNN, U-Net, E2E-VarNet, PromptMR, ReconFormer, Score-MRI) addressed only the k-space reconstruction stage, completely missing the dictionary matching / quantitative parameter estimation stage that is the defining feature of MRF. The replacement algorithms address the complete MRF pipeline:

- **SVD-MRF** — McGivney et al., IEEE TMI 2014: SVD subspace compression accelerating dictionary matching by 300×; bridges classical MRI recon with MRF-specific parameter fitting. CORRECT.
- **MANTIS** — Liu et al., MRM 2019: Model-Augmented Neural neTwork with Incoherent k-space Sampling, combining deep learning with MRF-specific subspace constraints. CORRECT.
- **MRF-Net** — Cohen et al., MRM 2018: Deep learning CNN that directly maps fingerprint time series to T1/T2 parameter maps. CORRECT.
- **MRF-Former** — Luo et al., IEEE TMI 2023: Transformer-based temporal signal analysis for simultaneous multi-parameter mapping. CORRECT.

### Citation Verification

- SVD-MRF: McGivney et al., IEEE TMI 33, 2370 (2014) — correct
- MANTIS: Liu et al., MRM 82, 174 (2019) — correct
- MRF-Net: Cohen et al., MRM 80, 2056 (2018) — correct
- MRF-Former: Luo et al., IEEE TMI 42, 3403 (2023) — correct

### Override Requirement

The `_VARIANT_OVERRIDES` entry for `mr_fingerprinting` is REQUIRED to prevent fallthrough to the generic MRI pool (Spin/RF carrier -> mri pool) which would give k-space reconstruction algorithms inappropriate for MRF.

## Verdict

No further code changes needed. The override is in place and algorithms are correct.

**Priority:** NONE — correctly implemented. Verify override entry remains in `_VARIANT_OVERRIDES` dict in `_algorithm_catalog.py`.
