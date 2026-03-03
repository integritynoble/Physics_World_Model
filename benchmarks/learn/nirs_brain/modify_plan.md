# Modify Plan: nirs_brain (Functional Near-Infrared Spectroscopy / fNIRS)

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Score key:** clinical_optics (routed via `_CARRIER_ROUTING[("medical", "Photon")]`)
- **Algorithms served (4):**
  1. FFT-OCT (Classical) -- Analytical baseline
  2. BM4D (PnP) -- Maggioni et al., IEEE TIP 2013
  3. Speckle-DenoiseNet (Deep Learning) -- Devalla et al., BOE 2019
  4. OCTA-Net (Transformer) -- Hybrid U-Net+Transformer, 2023

## Assessment

**Inappropriate.** The clinical_optics pool is designed for OCT/OCTA (Optical Coherence
Tomography), which is a fundamentally different modality from fNIRS:

- **fNIRS** measures diffuse near-infrared light to reconstruct brain hemodynamic
  activity (oxygenated/deoxygenated hemoglobin concentrations). The forward model
  involves diffuse optical tomography (DOT) with the diffusion equation.
- **OCT** measures interferometric backscattered light for structural cross-sectional
  imaging. Completely different physics.

Problems with current algorithms:
- "FFT-OCT" is specific to OCT interferogram processing -- it has no meaning in fNIRS.
- "Speckle-DenoiseNet" is an OCT speckle denoiser -- fNIRS does not have speckle.
- "OCTA-Net" is an OCT angiography segmentation network -- irrelevant to fNIRS.
- "BM4D" is a generic denoiser that could apply, but is not fNIRS-specific.

Appropriate fNIRS algorithms would include:
- Modified Beer-Lambert Law (MBLL) -- standard classical baseline
- Tikhonov-regularized DOT inversion
- PnP-ADMM with DOT forward model
- ReconNet-DOT or DL-DOT (learned diffuse optical tomography)

## Recommended Changes

**Recommended:** Add a variant override for `nirs_brain` in `_VARIANT_OVERRIDES`:
```python
"nirs_brain": [
    {"name": "MBLL",        "type": "Classical",     ...},
    {"name": "Tikhonov-DOT","type": "Classical",     ...},
    {"name": "PnP-DOT",    "type": "PnP",           ...},
    {"name": "DL-DOT",     "type": "Deep Learning",  ...},
]
```
Plus corresponding entry in `CATEGORY_REAL_SCORES`.

Alternatively, route `nirs_brain` to the `computational` pool (Tikhonov, PnP-RED,
Deep Image Prior, SwinIR), which is at least not OCT-specific.

## Verdict

Changes recommended. The current OCT-specific algorithms (FFT-OCT, Speckle-DenoiseNet,
OCTA-Net) are clearly wrong for a diffuse optical tomography modality. This is one
of the more significant mismatches in the catalog.
