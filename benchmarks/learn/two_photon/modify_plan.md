# Modify Plan: Two-Photon / Multiphoton Microscopy

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

Two-photon microscopy falls under `microscopy` category with carrier `Photon`. It receives:

- Richardson-Lucy (Classical) -- standard fluorescence deconvolution
- PnP-FISTA (PnP) -- plug-and-play with FISTA (Bai et al., 2020)
- CARE (Deep Learning) -- content-aware restoration (Weigert et al., Nat. Methods 2018)
- Restormer (Transformer) -- restoration transformer (Zamir et al., CVPR 2022)

Two-photon microscopy produces deep-tissue fluorescence images. The CARE paper (Weigert et al.) specifically demonstrates results on two-photon data, making it an ideal benchmark algorithm. Richardson-Lucy deconvolution is also standard for two-photon image restoration. Score key `microscopy` is correct.

No algorithm or citation changes required.
