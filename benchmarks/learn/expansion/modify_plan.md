# Modify Plan: expansion

## Current Assignment
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms:** Richardson-Lucy (Classical), PnP-FISTA (PnP), CARE (Deep Learning), Restormer (Transformer)

## Assessment

The algorithm assignment is appropriate. Expansion microscopy (ExM) physically
expands the specimen by ~4x using a swellable polymer, then images the expanded
sample with standard fluorescence microscopy (widefield, confocal, or
light-sheet). The reconstruction task is therefore standard fluorescence
microscopy image restoration:

- **Richardson-Lucy** is the standard deconvolution algorithm for fluorescence
  microscopy and the appropriate classical baseline for ExM.
- **PnP-FISTA** is a plug-and-play proximal method well-suited for
  microscopy deconvolution with learned priors.
- **CARE** (Weigert et al., Nat. Methods 2018) is the most widely-used deep
  learning method for fluorescence microscopy restoration and was specifically
  validated on confocal/widefield data similar to ExM.
- **Restormer** (Zamir et al., CVPR 2022) is a strong general-purpose
  image restoration transformer.

The microscopy category score ranges and mismatch descriptions (PSF
aberrations, refractive index, coverslip thickness) are all relevant to
expansion microscopy.

## Verdict

No code changes needed.
