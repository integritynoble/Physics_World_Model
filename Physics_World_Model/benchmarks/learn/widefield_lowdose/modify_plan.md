# Modify Plan: Low-Dose Widefield Microscopy

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

Low-dose widefield microscopy falls under `microscopy` category with carrier `Photon`. It receives:

- Richardson-Lucy (Classical) -- standard fluorescence deconvolution
- PnP-FISTA (PnP) -- plug-and-play with FISTA (Bai et al., 2020)
- CARE (Deep Learning) -- content-aware restoration (Weigert et al., Nat. Methods 2018)
- Restormer (Transformer) -- restoration transformer (Zamir et al., CVPR 2022)

Low-dose widefield is the same optical system as standard widefield but with reduced photon budget (fewer photons per pixel). The reconstruction task emphasizes denoising in the Poisson-noise-limited regime. All four algorithms are appropriate:

- Richardson-Lucy handles Poisson noise natively
- CARE was specifically designed for low-SNR microscopy restoration
- PnP-FISTA and Restormer provide regularized denoising

Score key `microscopy` is correct.

No algorithm or citation changes required.
