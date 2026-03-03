# Modify Plan: holography

## Current State

- **Category:** coherent
- **Carrier:** Photon
- **Score key:** coherent
- **Algorithms assigned:**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 1982
  2. prDeep (PnP) -- Metzler et al., ICML 2018
  3. PhaseNet (Deep Learning) -- Rivenson et al., LSA 2018
  4. LRGS (Deep Unrolling) -- Choi et al., 2023

## Assessment

**Appropriate: YES**

Digital holographic microscopy is a coherent imaging technique requiring phase
retrieval / holographic reconstruction. The "coherent" pool is exactly right:

- **GS/HIO (Gerchberg-Saxton / Hybrid Input-Output)**: The foundational
  iterative phase retrieval algorithm. Fienup 1982 is the canonical citation.
  Directly applicable to holographic reconstruction.
- **prDeep**: A PnP phase retrieval method (Metzler et al., ICML 2018).
  Combines a deep denoiser with phase retrieval iterations. Directly relevant.
- **PhaseNet**: Rivenson et al., Light: Science & Applications 2018. A deep
  learning method specifically designed for holographic reconstruction.
  This is THE landmark DL paper for digital holography.
- **LRGS (Learned Regularization for Generalized Scattering)**: A deep
  unrolling approach for coherent imaging. Appropriate.

All four algorithms are real, published, and directly applicable to digital
holographic microscopy. The learning materials (03_reconstruction_algorithms.md)
also list PhaseNet and Angular Spectrum, which are consistent with this pool.

## Code Changes Needed

No code changes needed.
