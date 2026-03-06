# Modify Plan: entangled_photon

## Current Assignment (updated 2026-03-06)
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

The algorithm assignment is appropriate. Entangled photon microscopy / imaging
uses photon-pair correlations and coincidence detection, which falls squarely
in the quantum imaging category:

- **G(2)-Corr** (Pittman et al., PRA 1995) is the foundational second-order
  correlation measurement used in ghost imaging and entangled photon setups.
- **CS-TVAL3** (Li et al., 2014) is a compressed-sensing reconstruction widely
  used in computational ghost imaging with few measurements.
- **DRU-Net** (Wang et al., Sci. Rep. 2020) is a deep learning approach for
  ghost image recovery from sparse coincidence data.
- **Ghost-ViT** (Zhu et al., 2025) is a vision transformer adapted for
  quantum/ghost imaging reconstruction.

The quantum category score ranges and mismatch descriptions (SLM/DMD pattern
fidelity, detector timing jitter, dark count rate) are appropriate for
entangled photon microscopy.

## Verdict

No code changes needed.
