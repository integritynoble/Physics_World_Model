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

## 2026-03-06 Comprehensive Check Update

- Physics: F_3PM ~ C(r) * I_exc^3 * PSF_3; scattering attenuation exp(-3*z/l_s); 1300-1700 nm excitation (third bio window)
- Key mismatch: scattering-induced PSF broadening at depth, pulse chirp/duration, depth-dependent signal attenuation, photodamage threshold
- GCS datasets: 3 tiers confirmed in challenge-data/v1.0/
- Algorithm pool: PASS — RL (baseline), PnP-FISTA (critical for very low photon counts ~5-50/pixel), CARE (validated on multiphoton deep-tissue data), Restormer (state-of-the-art)
- Note: 3PM is the most photon-limited microscopy modality in the catalog; PnP-FISTA's noise robustness is especially important here
