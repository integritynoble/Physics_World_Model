# Modify Plan: ct_fluorescence

## Current State

- **Category:** multi_modal_fusion
- **Carrier:** X-ray
- **Routing:** Direct to `multi_modal_fusion` pool (no carrier routing override)
- **Score key:** multi_modal_fusion
- **Algorithms served:**
  1. MLAA (Classical) -- Rezaei et al., IEEE TMI 2012
  2. MR-Guided (PnP) -- Ehrhardt et al., SIIS 2015
  3. FBSEM-Net (Deep Learning) -- Mehranian & Reader, IEEE TMI 2020
  4. PPMF-Net (Transformer) -- Li et al., 2024

## Assessment

The multi_modal_fusion pool is designed for PET-CT/PET-MR fusion. CT + Fluorescence (FLIT) is a different fusion problem combining X-ray CT with fluorescence molecular imaging.

- **MLAA (Maximum Likelihood Activity and Attenuation):** Specifically for joint PET activity + CT attenuation estimation. Not applicable to CT+fluorescence. WRONG.
- **MR-Guided:** MR-guided PET reconstruction. Not applicable. WRONG.
- **FBSEM-Net:** Forward-Backward Stochastic EM for PET. Not applicable. WRONG.
- **PPMF-Net:** PET-MR fusion network. Not applicable. WRONG.

All four algorithms are PET/SPECT fusion methods, not CT+fluorescence methods.

## Recommended Algorithms

CT+fluorescence (fluorescence-guided imaging / FLIT) involves combining anatomical CT with fluorescence molecular tomography (FMT). The reconstruction problem is joint CT reconstruction + diffuse optical tomography for fluorophore distribution.

| Slot | Algorithm | Type | Reference | Rationale |
|------|-----------|------|-----------|-----------|
| Classical | Born/Rytov + FBP | Classical | Ntziachristos et al., Nat. Med. 2010 | Standard: FBP for CT anatomy, Born approximation for fluorescence diffuse tomography |
| PnP | PnP-ADMM (joint) | PnP | Venkatakrishnan et al., 2013 | Joint CT-FMT reconstruction with plug-and-play prior using CT-derived structural guidance |
| Deep Learning | FDot-Net | Deep Learning | Gao et al., IEEE TMI 2021 | Deep learning for fluorescence diffuse optical tomography with CT structural prior |
| Transformer | Cross-Modal Transformer | Transformer | Generic cross-modal fusion, 2024 | Transformer architecture for joint CT+fluorescence feature fusion and reconstruction |

## Required Code Changes

1. **`_algorithm_catalog.py`:** Add `ct_fluorescence` to `_VARIANT_OVERRIDES` with CT+fluorescence-specific algorithms.
2. **`_algorithm_catalog.py`:** Add CT+fluorescence real scores to `CATEGORY_REAL_SCORES` if published data is available.
