# Modify Plan: dic (Differential Interference Contrast)

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms served:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

The generic microscopy pool is **acceptable but not ideal** for DIC. DIC is a
phase-gradient imaging modality where the measurement encodes the optical path
length gradient via shearing interferometry. The forward model involves PSF
convolution plus a directional gradient operation, so deconvolution algorithms
(Richardson-Lucy, CARE) are broadly applicable to the denoising/restoration
aspect.

However, the distinguishing reconstruction problem in DIC is **quantitative
phase retrieval** from the gradient-contrast image -- algorithms like
Transport-of-Intensity Equation (TIE) solvers, Wiener deconvolution with
phase-gradient kernels, or DIC-specific deep learning (e.g., PhaseStain,
Ounkomol et al. 2018) would be more domain-appropriate.

The current algorithms are not wrong (they address the image restoration
component), but they miss the phase-recovery aspect that makes DIC distinct.

## Verdict

No code changes needed. The generic microscopy pool is a reasonable fit since
DIC produces intensity images that benefit from standard deconvolution/denoising.
The phase-gradient specificity is captured by the modality's forward model and
mismatch parameters (shear_amount, bias_retardation, prism_orientation) rather
than by the algorithm pool. A future improvement could add a DIC-specific
sub-category with TIE/phase-gradient solvers.
