# Modify Plan: TIRF Microscopy (Total Internal Reflection Fluorescence)

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

TIRF microscopy falls under `microscopy` category with carrier `Photon`. It receives:

- Richardson-Lucy (Classical) -- standard fluorescence deconvolution
- PnP-FISTA (PnP) -- plug-and-play with FISTA (Bai et al., 2020)
- CARE (Deep Learning) -- content-aware restoration (Weigert et al., Nat. Methods 2018)
- Restormer (Transformer) -- restoration transformer (Zamir et al., CVPR 2022)

TIRF produces thin optical-section fluorescence images. The primary reconstruction task is deconvolution and denoising of fluorescence data. Richardson-Lucy and CARE are widely used in TIRF image processing. Score key `microscopy` is correct.

No algorithm or citation changes required.
