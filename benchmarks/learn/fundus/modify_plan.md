# Modify Plan: fundus

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Score key:** clinical_optics (routed via `("medical", "Photon") -> "clinical_optics"`)
- **Algorithms assigned:**
  1. FFT-OCT (Classical) -- Analytical baseline
  2. BM4D (PnP) -- Maggioni et al., IEEE TIP 2013
  3. Speckle-DenoiseNet (Deep Learning) -- Devalla et al., BOE 2019
  4. OCTA-Net (Transformer) -- Hybrid U-Net+Transformer, 2023

## Assessment

**Problematic -- algorithms are OCT-specific, not fundus-specific**

Fundus photography is a simple optical imaging modality (white-light
reflectance/fluorescence of the retina). It does NOT involve OCT, Fourier-domain
reconstruction, or speckle. The current "clinical_optics" pool was designed
for OCT and contains:

- **FFT-OCT**: This is an OCT-specific algorithm (Fourier-domain OCT
  reconstruction). Fundus cameras do not produce interferograms.
- **BM4D**: A generic 3D denoiser -- acceptable but not fundus-specific.
- **Speckle-DenoiseNet**: OCT speckle denoising. Fundus images do not have
  OCT-type speckle noise.
- **OCTA-Net**: OCT angiography network. Not applicable to fundus photography.

The fundus camera inverse problem is essentially image deblurring/denoising
through ocular optics (cornea + lens PSF). Better algorithms would be:

- **Wiener Deconvolution** (classical)
- **Richardson-Lucy** (classical iterative)
- **Retinal image enhancement networks** (e.g., cofe-Net, I-SECRET)
- **Fundus-specific denoising/super-resolution** (Swin-Retina, etc.)

## Code Changes Needed

**Add fundus-specific variant override in `_algorithm_catalog.py`:**

```python
"fundus": [
    {"name": "Richardson-Lucy",    "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Richardson 1972 / Lucy 1974"},
    {"name": "PnP-BM3D",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
    {"name": "cofe-Net",           "type": "Deep Learning", "mask_aware": False, "params": "5M",   "source": "Shen et al., IEEE TMI 2020"},
    {"name": "Swin-Fundus",       "type": "Transformer",   "mask_aware": True,  "params": "15M",  "source": "SwinIR-based retinal enhancement, 2023"},
],
```

**Alternatively**, add a new carrier routing for fundus specifically, but a
variant override is cleaner since fundus is the only fundus-photography
modality.
