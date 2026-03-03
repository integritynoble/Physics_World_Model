# Modify Plan -- quantum_illumination

## Current State

- **Category:** quantum
- **Carrier:** Photon
- **Routing:** No carrier routing for `("quantum", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["quantum"]`
- **Score key:** quantum
- **Algorithms assigned:**
  1. G(2)-Corr (Classical) -- Pittman et al., PRA 1995
  2. CS-TVAL3 (PnP) -- Li et al., 2014
  3. DRU-Net (Deep Learning) -- Wang et al., Sci. Rep. 2020
  4. Ghost-ViT (Transformer) -- Zhu et al., 2025

## Assessment

**Appropriate: YES.**

Quantum illumination uses entangled photon pairs to detect objects in noisy/lossy environments. The reconstruction problem involves recovering a target scene from correlation measurements of signal and idler photons. This is closely related to ghost imaging and quantum correlation imaging.

- **G(2)-Corr**: Second-order correlation measurement is the foundational reconstruction method for quantum imaging. Directly applicable -- quantum illumination detects targets by measuring correlations between signal and idler beams.
- **CS-TVAL3**: Compressed sensing with total variation is widely used in quantum ghost imaging to reduce the number of measurements needed. Appropriate for quantum illumination where measurement budgets are limited.
- **DRU-Net**: Deep learning denoising/reconstruction from correlated photon measurements. Published for ghost imaging recovery.
- **Ghost-ViT**: Transformer-based quantum imaging reconstruction. Cutting-edge approach for this domain.

All four algorithms are from the quantum/ghost imaging literature and are well-suited.

## Plan

No code changes needed.
