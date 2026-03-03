# Modify Plan: pet_ct

## Current State
- **Category:** multi_modal_fusion
- **Carrier:** X-ray
- **Score key:** multi_modal_fusion
- **Algorithms:**
  1. MLAA (Classical) -- Rezaei et al., IEEE TMI 2012
  2. MR-Guided (PnP) -- Ehrhardt et al., SIIS 2015
  3. FBSEM-Net (Deep Learning) -- Mehranian & Reader, IEEE TMI 2020
  4. PPMF-Net (Transformer) -- Li et al., 2024

## Assessment

PET/CT fusion is a multi-modal medical imaging technique. The category `multi_modal_fusion` is correct. The algorithms are appropriate:

- **MLAA** (Maximum-Likelihood Activity and Attenuation estimation) -- a standard joint PET/CT reconstruction algorithm (Rezaei et al., IEEE TMI 2012). Correct.
- **MR-Guided** -- while the name says "MR-Guided" (from Ehrhardt et al.), the concept of anatomically-guided PET reconstruction applies equally to CT-guided PET. The name could be more specific ("CT-Guided PET" or "Anatomically-Guided"), but the underlying PnP approach is appropriate.
- **FBSEM-Net** -- a deep learning PET reconstruction network (Mehranian & Reader, IEEE TMI 2020). Correct.
- **PPMF-Net** -- a transformer for PET/MR fusion (Li et al., 2024). Applicable to PET/CT as well.

The leaderboard (PPMF-Net, FuseNet, MR-Guided, OSEM+AC) shows domain-appropriate fusion methods.

## Required Changes

No code changes needed. The multi_modal_fusion algorithms are appropriate for PET/CT. The "MR-Guided" name is slightly misleading for PET/CT context (it references an MR-guided paper), but the underlying method class (anatomically-guided PnP reconstruction) is valid.
