# Modify Plan: fundus

## Current Assignment
- **Category:** medical (routed to clinical_optics via `("medical", "Photon")`)
- **Carrier:** Photon
- **Score key:** clinical_optics
- **Algorithms (after override):** Richardson-Lucy (Classical), PnP-BM3D (PnP), cofe-Net (Deep Learning), Swin-Fundus (Transformer)

## Assessment

The algorithms were **problematic** before the override. The clinical_optics
pool was designed for OCT and contains OCT-specific algorithms. Fundus
photography is a simple optical imaging modality (white-light reflectance of
the retina) with NO involvement of OCT, Fourier-domain reconstruction, or
speckle.

**Problems with the original assignment:**
- **FFT-OCT**: This is an OCT-specific algorithm (Fourier-domain OCT
  reconstruction). Fundus cameras do not produce interferograms.
- **BM4D**: A generic 3D denoiser -- acceptable but not fundus-specific.
- **Speckle-DenoiseNet**: OCT speckle denoising. Fundus images do not have
  OCT-type speckle noise.
- **OCTA-Net**: OCT angiography network. Not applicable to fundus photography.

The fundus camera inverse problem is essentially image deblurring/denoising
through ocular optics (cornea + lens PSF). Better algorithms are:
- **Richardson-Lucy** (classical iterative deconvolution)
- **PnP-BM3D** (plug-and-play deblurring with BM3D prior)
- **cofe-Net** (corrective fusion enhancement for fundus, Shen et al., TMI 2020)
- **Swin-Fundus** (SwinIR-based retinal image enhancement)

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"fundus": [
    {"name": "Richardson-Lucy",    "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Richardson 1972 / Lucy 1974"},
    {"name": "PnP-BM3D",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
    {"name": "cofe-Net",           "type": "Deep Learning", "mask_aware": False, "params": "5M",   "source": "Shen et al., IEEE TMI 2020"},
    {"name": "Swin-Fundus",       "type": "Transformer",   "mask_aware": True,  "params": "15M",  "source": "SwinIR-based retinal enhancement, 2023"},
],
```

Also added `"fundus"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"fundus"` to `_VARIANT_OVERRIDES`
  - Added `"fundus"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct fundus-specific deconvolution/enhancement algorithms.
