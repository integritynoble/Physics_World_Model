# Modify Plan: Three-Photon Microscopy

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

Three-photon microscopy falls under `microscopy` category with carrier `Photon`. It receives:

- Richardson-Lucy (Classical) -- standard deconvolution for fluorescence microscopy
- PnP-FISTA (PnP) -- plug-and-play with FISTA optimizer (Bai et al., 2020)
- CARE (Deep Learning) -- content-aware image restoration for microscopy (Weigert et al., Nat. Methods 2018)
- Restormer (Transformer) -- general restoration transformer (Zamir et al., CVPR 2022)

Three-photon microscopy produces fluorescence images with deep tissue penetration. The reconstruction task is deconvolution and denoising, identical to other fluorescence microscopy modalities. Richardson-Lucy and CARE are standard tools in fluorescence microscopy. Score key `microscopy` is correct.

No algorithm or citation changes required.
