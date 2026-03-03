# Modify Plan: stem

## Current State
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic (non-cryo EM routing)
- **Algorithms:**
  1. Wiener Filter (Classical) -- Analytical baseline
  2. BM3D (PnP) -- Dabov et al., IEEE TIP 2007
  3. Noise2Void (Deep Learning) -- Krull et al., CVPR 2019
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

The algorithms are reasonable but could be more STEM-specific. The EM generic pool provides denoising algorithms, which is one of the main tasks in STEM image processing (STEM images are inherently noisy due to electron dose limitations). However, STEM also involves specific reconstruction tasks depending on the imaging mode:

- **Wiener Filter** is a valid denoising/deconvolution baseline for STEM.
- **BM3D** is widely used for STEM denoising (e.g., Mevenkamp et al., Adv. Struct. Chem. Imaging, 2015).
- **Noise2Void** has been specifically applied to electron microscopy denoising (Buchholz et al., 2019), making it highly appropriate.
- **SwinIR** is a general restoration method applicable to STEM.

For HAADF-STEM, the primary challenge is denoising at low electron dose. For ptychographic STEM (4D-STEM), phase retrieval methods would be needed, but the generic STEM modality likely refers to conventional STEM imaging.

The current algorithm selection is acceptable for conventional STEM denoising/restoration. More STEM-specific alternatives could include:
- Non-local means adapted for STEM (Kondo et al., Microscopy 2016)
- STEM-specific DL denoising (e.g., STEM-Net)

Overall, the generic EM denoising pool is a reasonable fit.

No code changes needed.

## Files to Modify
None.
