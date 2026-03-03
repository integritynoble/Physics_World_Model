# Modify Plan: entangled_photon

## Current Assignment
- **Category:** quantum
- **Carrier:** Photon
- **Score key:** quantum
- **Algorithms:** G(2)-Corr (Classical), CS-TVAL3 (PnP), DRU-Net (Deep Learning), Ghost-ViT (Transformer)

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
