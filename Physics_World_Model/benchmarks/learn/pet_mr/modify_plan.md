# Modify Plan: pet_mr

## Current State
- **Category:** multi_modal_fusion
- **Carrier:** Gamma
- **Score key:** multi_modal_fusion
- **Algorithms:**
  1. MLAA (Classical) -- Rezaei et al., IEEE TMI 2012
  2. MR-Guided (PnP) -- Ehrhardt et al., SIIS 2015
  3. FBSEM-Net (Deep Learning) -- Mehranian & Reader, IEEE TMI 2020
  4. PPMF-Net (Transformer) -- Li et al., 2024

## Assessment

PET/MR fusion is a multi-modal medical imaging technique combining PET (nuclear) with MRI (structural/functional). The category `multi_modal_fusion` is correct. All algorithms are highly appropriate:

- **MLAA** -- joint activity and attenuation estimation, standard for PET without CT-based attenuation maps (crucial for PET/MR since MR does not directly provide attenuation). Correct.
- **MR-Guided** -- MR-guided PET reconstruction (Ehrhardt et al., SIIS 2015). This paper is specifically about using MR anatomical priors to guide PET reconstruction. Perfect match for PET/MR.
- **FBSEM-Net** -- PET reconstruction network (Mehranian & Reader, IEEE TMI 2020). Correct.
- **PPMF-Net** -- transformer for PET/MR fusion. Correct.

## Required Changes

No code changes needed. The algorithms are perfectly matched for PET/MR fusion.

## 2026-03-06 Comprehensive Check Update

- Physics: Poisson OSEM (PET) + undersampled k-space MRI with MRAC coupling
- Key mismatch: MRAC bone error (5-15% SUV bias), simultaneous motion, k-undersampling, B0 inhomogeneity
- GCS datasets: 3 tiers confirmed in challenge-data/v1.0/
- Algorithm pool: PASS — MLAA (MR-specific joint AC) + MR-Guided PnP + FBSEM-Net + transformers are all ideal for PET-MR
- Note: PET-MR is the best-matched modality in the entire multi_modal_fusion pool
