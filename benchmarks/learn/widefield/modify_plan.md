# Modify Plan: Widefield Fluorescence Microscopy

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

Widefield fluorescence microscopy falls under `microscopy` category with carrier `Photon`. It receives:

- Richardson-Lucy (Classical) -- the standard deconvolution method for widefield fluorescence
- PnP-FISTA (PnP) -- plug-and-play with FISTA (Bai et al., 2020)
- CARE (Deep Learning) -- content-aware restoration (Weigert et al., Nat. Methods 2018)
- Restormer (Transformer) -- restoration transformer (Zamir et al., CVPR 2022)

Richardson-Lucy deconvolution is THE canonical algorithm for widefield microscopy deconvolution -- it was originally designed for this exact use case. CARE was also demonstrated on widefield data in the original paper. This is an excellent algorithm match. Score key `microscopy` is correct.

No algorithm or citation changes required.
