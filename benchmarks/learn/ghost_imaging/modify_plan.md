# Modify Plan: ghost_imaging

## Current State (updated 2026-03-06)

- **Category:** quantum
- **Carrier:** Photon
- **Score key:** quantum
- **Algorithms (10 total from quantum pool):**
  1. G(2)-Corr (Classical) -- Pittman et al., PRA 1995
  2. Photon Counting (Classical) -- Classical baseline
  3. CS-TVAL3 (PnP) -- Li et al., 2014
  4. Bayesian CS (PnP) -- Bayesian compressed sensing
  5. DRU-Net (Deep Learning) -- Wang et al., Sci. Rep. 2020
  6. Quantum-CNN (Deep Learning) -- Quantum imaging CNN
  7. Ghost-ViT (Vision Transformer) -- Zhu et al., 2025
  8. Quantum-ViT (Vision Transformer) -- Quantum imaging transformer, 2024
  9. DiffusionQuantum (Diffusion) -- Zhang et al., 2024
  10. ScoreQuantum (Score-based) -- Wei et al., 2025

**Status:** PASS — check.md written 2026-03-06

## Assessment

**Appropriate: YES**

Ghost imaging is a quantum/computational imaging technique that reconstructs
images from intensity correlations. The quantum pool has algorithms specifically
chosen for ghost imaging:

- **G(2)-Corr**: The fundamental second-order intensity correlation method
  from Pittman et al. (1995) -- this IS the original ghost imaging algorithm.
- **CS-TVAL3**: Compressed sensing with total-variation regularization, widely
  used in computational ghost imaging (Li et al., 2014). Correctly placed as
  the optimization-based method.
- **DRU-Net**: A deep learning approach for ghost image reconstruction (Wang
  et al., Sci. Rep. 2020). Appropriate for the domain.
- **Ghost-ViT**: A vision transformer for ghost imaging. Forward-looking but
  reasonable.

All algorithms are directly relevant to ghost imaging reconstruction.

## Code Changes Needed

No code changes needed.
