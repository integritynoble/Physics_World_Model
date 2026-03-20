# Modify Plan: US/MRI Fusion

**Created:** 2026-03-03
**Status:** Algorithms are a mismatch -- PET-specific methods assigned to US/MRI fusion

## Assessment

US/MRI fusion falls under `multi_modal_fusion` category with carrier `Acoustic`. It receives:

- MLAA (Classical) -- Maximum Likelihood Activity and Attenuation (Rezaei et al., IEEE TMI 2012)
- MR-Guided (PnP) -- MR-guided PET reconstruction (Ehrhardt et al., SIIS 2015)
- FBSEM-Net (Deep Learning) -- Forward-Backward Stochastic EM for PET (Mehranian & Reader, IEEE TMI 2020)
- PPMF-Net (Transformer) -- Li et al., 2024

### Issue

MLAA and FBSEM-Net are **PET-specific** algorithms. MLAA jointly estimates PET activity and attenuation maps from PET emission data. FBSEM-Net is an EM-based PET reconstruction network. Neither applies to ultrasound/MRI fusion.

US/MRI fusion is used in image-guided interventions (e.g., prostate biopsy) where real-time ultrasound is registered to pre-operative MRI. Appropriate methods include:

- Classical: Rigid/deformable registration (e.g., B-spline or Demons-based)
- Deep Learning: VoxelMorph (Balakrishnan et al., IEEE TMI 2019), Label-reg
- Real-time fusion: CNN-based US-to-MRI registration networks

MR-Guided is labeled as PET reconstruction with MR priors (Ehrhardt et al.), not US/MRI fusion. PPMF-Net is the most generic of the four.

### Decision

The `multi_modal_fusion` category is a shared pool for PET-CT, PET-MR, SPECT-CT, and US-MRI. The pool is dominated by nuclear medicine fusion methods (MLAA, FBSEM-Net) which do not transfer to US/MRI. However, creating a separate US/MRI fusion pool would need a variant override or carrier routing.

## Deferred Items

1. **HIGH PRIORITY**: Add `_CARRIER_ROUTING` entry `("multi_modal_fusion", "Acoustic")` pointing to a new `us_mri_fusion` pool, or add `us_mri` to `_VARIANT_OVERRIDES` with registration-based algorithms:
   - Classical: Demons or B-spline registration
   - PnP: Regularized deformable registration
   - Deep Learning: VoxelMorph (Balakrishnan et al., IEEE TMI 2019)
   - Transformer: TransMorph (Chen et al., Med. Image Anal. 2022)
2. **Score key**: Would also need a `us_mri_fusion` entry in `CATEGORY_REAL_SCORES`.

No code changes made in this pass, but this is one of the more significant mismatches among the 14 modalities reviewed.
