# Modify Plan: pet_ct (PET-CT Fusion)

## Current State

- **Category:** multi_modal_fusion
- **Carrier:** X-ray (CT component) + Gamma (PET component)
- **Score key:** multi_modal_fusion
- **Algorithms served (11 in full catalog, 4 highlighted):**
  1. MLAA (Classical) -- Rezaei et al., IEEE TMI 2012
  2. Image Registration (Classical) -- Rigid/deformable registration baseline
  3. MR-Guided (PnP) -- Ehrhardt et al., SIIS 2015
  4. Guided Reconstruction (PnP) -- Structural guidance from auxiliary modality
  5. FBSEM-Net (Deep Learning) -- Mehranian & Reader, IEEE TMI 2020
  6. Fusion-U-Net (Deep Learning) -- Dual-input U-Net for fusion
  7. PPMF-Net (Vision Transformer) -- Li et al., 2024
  8. CrossModal-ViT (Vision Transformer) -- Cross-modal attention transformer, 2024
  9. MultiModal-Fusion-Former (Vision Transformer) -- Multi-modal fusion transformer, 2024
  10. DiffusionFusion (Diffusion) -- Zhang et al., 2024
  11. ScoreFusion (Score-based) -- Wei et al., 2025

## Assessment

**Appropriate.** PET/CT fusion is the standard clinical combined modality scanner. The algorithms are well-matched:

- **MLAA** (Maximum-Likelihood Activity and Attenuation estimation) is the reference joint reconstruction algorithm for PET-CT. Direct.
- **MR-Guided**: the name refers to PnP structural guidance from an anatomical modality — applicable to CT guidance in PET-CT with the same mathematical framework.
- **FBSEM-Net**: deep learning PET reconstruction with anatomical side information (Mehranian & Reader 2020). Highly appropriate.
- **CrossModal-ViT** and **PPMF-Net**: cross-modal transformers for PET-CT joint functional-anatomical reconstruction. State-of-the-art.
- **DiffusionFusion**: score-based diffusion conditioned on CT for PET reconstruction. Current frontier.

The "MR-Guided" name is slightly misleading for PET-CT context (references MR-guided paper), but the underlying PnP anatomically-guided approach applies equally to CT guidance.

## 2026-03-06 Comprehensive Check Update

- Physics: dual-modality Beer-Lambert (CT) + Poisson OSEM (PET) with ACF coupling
- Key mismatch: CT-to-511-keV scaling, patient motion, scatter fraction, TOF kernel
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — MLAA + PnP guidance + deep learning + transformers cover full paradigm space

## Verdict

No code changes needed. Algorithm naming note: "MR-Guided" could be renamed "CT-Guided" for PET-CT clarity, but the mathematical approach is identical.
