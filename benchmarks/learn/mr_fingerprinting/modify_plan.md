# Modify Plan — mr_fingerprinting

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via _CARRIER_ROUTING: medical + Spin/RF -> mri)
- **Algorithms (from catalog, 8 total):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022
- **Leaderboard (live):** All 8 algorithms (8 entries)

## Assessment

The algorithms are **partially appropriate** for MR Fingerprinting (MRF).

MRF (Ma et al., Nature 2013) is a quantitative MRI technique with a distinctive two-stage pipeline:
1. **Image reconstruction** from highly undersampled k-space (spiral/radial trajectories) per time frame
2. **Dictionary matching / parameter mapping** to estimate T1, T2, and other tissue properties by comparing temporal signal evolution to a precomputed dictionary

The MRI pool algorithms address Stage 1 (k-space reconstruction), which is valid. However, MRF has its own specialized algorithms:
- **Dictionary Matching** (Ma et al., Nature 2013) -- the original MRF method
- **SVD Compression** (McGivney et al., IEEE TMI 2014) -- accelerated dictionary matching
- **MRF-Net / DRONE** (Cohen et al., MRM 2018) -- deep learning for direct parameter mapping
- **SCQ-MRF** (Fang et al., NeurIPS 2019) -- subspace-constrained quantification

The standard MRI reconstruction algorithms (VarNet, PromptMR, etc.) solve the spatial image reconstruction sub-problem but miss the temporal dictionary matching / quantitative parameter estimation that makes MRF unique.

## Recommended Changes (Optional)

If improving specificity:
1. Add a variant override for `mr_fingerprinting`:
   - Classical: **Dictionary Matching** -- Ma et al., Nature 2013
   - Compressed Sensing: **SVD-MRF** -- McGivney et al., IEEE TMI 2014
   - PnP: **PnP-DnCNN** (keep)
   - Deep Learning: **DRONE** -- Cohen et al., MRM 2018
   - Deep Unrolling: **E2E-VarNet** (keep for k-space stage)
   - Transformer: **ReconFormer** (keep)

## Verdict

No code changes needed. The MRI pool algorithms correctly target the k-space image reconstruction stage, which is a valid and important inverse problem in MRF. The dictionary matching stage is a downstream task. However, a variant override would better represent the full MRF reconstruction pipeline.
