# Modify Plan: eels

## Current State (After Fix)
- **Category:** electron_microscopy
- **Sub-category pool:** em_analytical (EELS-specific spectral deconvolution)
- **Algorithms:** [Fourier-Ratio, RL-EELS, NMF-EELS, EELS-Net]

## Assessment
Algorithms are now domain-appropriate.

The previous generic EM denoising pool (Wiener Filter, BM3D, Noise2Void, SwinIR) addressed spatial image denoising but missed the spectral deconvolution problem that is central to EELS. The replacement algorithms target EELS specifically:
- **Fourier-Ratio** — the canonical ZLP deconvolution method, divides the measured spectrum by the ZLP in Fourier space (Egerton, EELS in the Electron Microscope, 2011)
- **RL-EELS** — Richardson-Lucy iterative deconvolution using the ZLP as the PSF (Gloter et al., Ultramicroscopy 2003)
- **NMF-EELS** — Non-negative Matrix Factorization for spectral unmixing of overlapping EELS edges (de la Pena et al., HyperSpy 2016)
- **EELS-Net** — deep learning CNN for end-to-end ZLP deconvolution and background subtraction (Schwartz et al., npj Comput. Mater. 2022)

## Verdict
No further code changes needed.
