# Modify Plan: phase_retrieval

## Current State
- **Category:** coherent
- **Carrier:** Photon/Electron
- **Score key:** coherent
- **Algorithms:**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 1982
  2. prDeep (PnP) -- Metzler et al., ICML 2018
  3. PhaseNet (Deep Learning) -- Rivenson et al., LSA 2018
  4. LRGS (Deep Unrolling) -- Choi et al., 2023

## Assessment

Phase retrieval / coherent diffractive imaging (CDI) recovers the complex wavefield from intensity-only diffraction measurements. The category `coherent` is correct. The algorithms are excellent domain-specific choices:

- **GS/HIO** (Gerchberg-Saxton / Hybrid Input-Output) -- the foundational phase retrieval algorithms (Fienup 1982). Perfect.
- **prDeep** -- PnP phase retrieval with deep denoiser prior (Metzler et al., ICML 2018). Domain-specific. Perfect.
- **PhaseNet** -- deep learning phase retrieval (Rivenson et al., LSA 2018). Domain-specific. Perfect.
- **LRGS** -- deep unrolling for phase retrieval (Choi et al., 2023). Domain-specific. Perfect.

All four algorithms are specifically designed for phase retrieval problems.

## Required Changes

No code changes needed. The coherent category algorithms are perfectly matched for phase retrieval.
