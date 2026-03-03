# Modify Plan -- shearography

## Current State

- **Category:** industrial_inspection
- **Carrier:** Photon
- **Routing:** No carrier routing for `("industrial_inspection", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["industrial_inspection"]`
- **Score key:** industrial_inspection
- **Algorithms assigned:**
  1. TSR (Classical) -- Shepard et al., 2003
  2. PnP-ADMM (PnP) -- ADMM + denoiser prior
  3. DefectNet (Deep Learning) -- U-Net for NDT, 2021
  4. LSTM-NDT (Recurrent) -- Fang et al., 2022

## Assessment

**Partially appropriate: Acceptable but not ideal.**

Shearography (speckle pattern shearing interferometry) is a full-field optical NDT technique that measures surface displacement derivatives (strain) by interfering sheared copies of a speckle pattern. The reconstruction involves phase unwrapping and strain field recovery from fringe patterns.

- **TSR** (Thermographic Signal Reconstruction): Specifically designed for pulsed thermography data analysis, NOT shearography. TSR fits polynomial models to thermal decay curves, which is unrelated to phase unwrapping from speckle interference. This is a weak match -- TSR is for a different NDT sub-modality.
- **PnP-ADMM**: Generic optimization with denoiser prior. Applicable to any image reconstruction problem, including phase map denoising.
- **DefectNet**: U-Net for NDT defect detection. Acceptable as a generic NDT deep learning approach, though not specifically for shearographic phase analysis.
- **LSTM-NDT**: Recurrent network for NDT sequences. Could process temporal phase-stepping sequences in temporal phase-shifting shearography.

Domain-specific algorithms for shearography would include: spatial/temporal phase unwrapping (Goldstein et al., 1988; Huntley & Saldner, 1993), windowed Fourier transform (Kemao, 2004), and deep learning for fringe analysis (Feng et al., Opt. Lasers Eng. 2019). The current pool reflects thermography-centric NDT rather than interferometric NDT.

However, since the platform uses a generic forward-model framework, the algorithms function as inverse-problem baselines regardless of domain specificity.

## Plan

No code changes needed. The industrial_inspection pool is a reasonable broad-category assignment. TSR is not ideal for shearography but the overall pool covers the classical/PnP/DL/sequence-based spread that the benchmark framework requires.
