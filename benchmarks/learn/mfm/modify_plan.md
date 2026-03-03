# Modify Plan — mfm

## Current State

- **Category:** scanning_probe
- **Carrier:** Magnetic
- **Score key:** scanning_probe
- **Algorithms (from catalog):**
  1. BTR (Classical) -- Villarrubia, JRNIST 1997
  2. Reg-Deconv (PnP) -- Dongmo et al., 2000
  3. DeepSPM (Deep Learning) -- Alldritt et al., Commun. Phys. 2020
  4. E2E-BTR (Deep Learning) -- Kossler et al., Sci. Rep. 2022
- **Leaderboard (live):** BTR, Reg-Deconv, DeepSPM, E2E-BTR (4 entries)

## Assessment

The algorithms are **appropriate** for Magnetic Force Microscopy (MFM).

- **BTR (Blind Tip Reconstruction)** by Villarrubia is the foundational algorithm for scanning probe tip deconvolution. While originally designed for AFM, the same tip-sample convolution artifact affects MFM, making this directly applicable.
- **Reg-Deconv (Regularized Deconvolution)** by Dongmo et al. is a regularized inverse approach for scanning probe data -- applicable to MFM tip deconvolution and transfer function inversion.
- **DeepSPM** by Alldritt et al. is a deep learning method for scanning probe microscopy image analysis. Applicable to MFM.
- **E2E-BTR (End-to-End Blind Tip Reconstruction)** by Kossler et al. is a deep learning approach for tip reconstruction in scanning probe microscopy. Directly relevant.

All four algorithms are real, published, properly cited, and come from the scanning probe microscopy domain. The scanning_probe category is well-curated for MFM.

## Verdict

No code changes needed.
