# Modify Plan: spc

## Current State
- **Category:** compressive
- **Carrier:** Photon
- **Score key:** compressive
- **Algorithms:**
  1. GAP-TV (Classical) -- Yuan et al., 2016
  2. PnP-FFDNet (PnP) -- Zhang et al., 2017
  3. EfficientSCI (Deep Learning) -- Wang et al., 2023
  4. MST-L (Transformer) -- Cai et al., CVPR 2022

## Assessment

The algorithms are appropriate for SPC (Single-Pixel Camera). SPC is a compressive sensing modality that uses random modulation patterns and a single-pixel detector to acquire compressed measurements.

- **GAP-TV** (Generalized Alternating Projection with Total Variation) is a standard compressive imaging reconstruction algorithm, applicable to SPC.
- **PnP-FFDNet** is a plug-and-play method that works well with compressive sensing forward models.
- **EfficientSCI** is designed for snapshot compressive imaging; while it targets video SCI specifically, the compressive sensing framework is the same.
- **MST-L** (Mask-guided Spectral-wise Transformer) is designed for spectral SCI (CASSI), not single-pixel cameras specifically.

**Minor concern:** EfficientSCI and MST-L are primarily designed for snapshot compressive imaging (video or spectral), not single-pixel cameras. SPC-specific algorithms include TVAL3 (Li et al., 2013), D-AMP (Metzler et al., NeurIPS 2016), and SPC-specific deep networks. However, the compressive sensing mathematical framework is shared, and these algorithms can be applied to SPC problems. The existing hand-crafted variants `spc_block` and `spc_kronecker` already have InverseNet-validated overrides, so the generic `spc` entry using the compressive pool is a reasonable fallback.

No code changes needed.

## Files to Modify
None.
