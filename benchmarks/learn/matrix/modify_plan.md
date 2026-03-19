# Modify Plan — matrix

## Current State

- **Category:** compressive
- **Carrier:** Photon
- **Score key:** compressive
- **Algorithms (from catalog):**
  1. GAP-TV (Classical) -- Yuan et al., 2016
  2. PnP-FFDNet (PnP) -- Zhang et al., 2017
  3. EfficientSCI (Deep Learning) -- Wang et al., 2023
  4. MST-L (Transformer) -- Cai et al., CVPR 2022
- **Leaderboard (live):** GAP-TV, PnP-FFDNet, EfficientSCI, MST-L (4 entries)

## Assessment

The algorithms are **partially appropriate** for Generic Matrix Sensing.

- **GAP-TV** (Generalized Alternating Projection with Total Variation) was developed for snapshot compressive imaging. For generic matrix sensing (y = Phi * x), TV regularization is a valid classical approach, so this is acceptable.
- **PnP-FFDNet** is a general PnP method -- acceptable for any linear inverse problem including matrix sensing.
- **EfficientSCI** (Wang et al., 2023) is specifically designed for Snapshot Compressive Imaging (SCI) with coded aperture masks. It assumes a specific SCI forward model structure (temporal compressive sensing with masks), which is more specialized than generic matrix sensing. Still acceptable since matrix sensing is in the "compressive" category.
- **MST-L** (Mask-guided Spectral-wise Transformer) is designed for spectral compressive imaging (CASSI-type). The "mask-guided" aspect is specific to coded aperture spectral imaging, not generic matrix sensing.

The "compressive" category algorithms are tuned for snapshot compressive imaging (SCI/CASSI), which is a specific subset of compressive sensing. Generic matrix sensing (y = Phi * x) is broader and could benefit from more general compressive sensing algorithms like ISTA/FISTA, LISTA, or AMP.

However, the 03_reconstruction_algorithms.md learning material already lists LISTA and Diffusion Posterior Sampling as modality-specific solvers, which are more appropriate. The catalog algorithms represent the benchmark leaderboard baselines rather than the internal solvers.

## Recommended Changes (Optional)

If improving specificity for the benchmark leaderboard:
1. Add a variant override for `matrix`:
   - Classical: **FISTA-L1** (Beck & Teboulle, SIAM J. Imaging Sci. 2009)
   - PnP: **PnP-FFDNet** (keep)
   - Deep Learning: **LISTA** (Gregor & LeCun, ICML 2010)
   - Transformer: **Transformer-CS** or DPS (Song et al., ICLR 2023)

## Verdict

No code changes needed (the compressive category algorithms are close enough). Optional: add a variant override with FISTA-L1 and LISTA for better alignment with generic matrix sensing.
