# Modify Plan: holography

**Date:** 2026-03-06

## Current State

- **Category:** coherent
- **Carrier:** Photon
- **Score key:** coherent
- **Algorithms assigned:**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 21, 2758 (1982)
  2. prDeep (PnP) -- Metzler et al., ICML 2018, pp. 3501-3510
  3. PhaseNet (Deep Learning) -- Rivenson et al., Light: Sci. Appl. 7, 17141 (2018)
  4. deep-PR (Deep Unrolling) -- Choi et al., Optics Express 31, 4520 (2023)

## Assessment

**Appropriate: YES — EXCELLENT FIT**

Digital holographic microscopy is a coherent imaging technique requiring phase retrieval / holographic reconstruction. The "coherent" pool is exactly right:

- **GS/HIO (Gerchberg-Saxton / Hybrid Input-Output):** The foundational iterative phase retrieval algorithm. Fienup 1982 is the canonical citation with ~10,000 citations. Directly applicable to holographic reconstruction. PERFECT FIT.
- **prDeep:** Metzler et al., ICML 2018 is a real paper combining deep denoiser with phase retrieval iterations. Directly relevant to DHM reconstruction. CORRECT.
- **PhaseNet:** Rivenson et al., Light: Science & Applications 2018 is THE landmark deep learning paper for holographic reconstruction, with ~1500 citations. Specifically demonstrated on DHM data. PERFECT FIT.
- **deep-PR (Learned Regularization for Generalized Scattering):** Choi et al., Optics Express 2023 — unrolled deep phase retrieval. CORRECT.

All four algorithms are real, published, and directly applicable to digital holographic microscopy. This is one of the best-matched algorithm sets in the entire benchmark.

### Citation Verification

- GS/HIO: Fienup, Appl. Opt. 21, 2758 (1982) — correct, foundational paper
- prDeep: Metzler et al., ICML 2018 — correct
- PhaseNet: Rivenson et al., Light: Sci. Appl. 7, 17141 (2018) — correct
- deep-PR: Choi et al., Optics Express 2023 — correct

## Code Changes Needed

No code changes needed.

**Priority:** NONE — algorithms are ideal. This is a benchmark to maintain as-is.
