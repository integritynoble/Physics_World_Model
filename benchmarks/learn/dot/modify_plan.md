# Modify Plan: dot (Diffuse Optical Tomography)

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Score key:** clinical_optics (routed via carrier)
- **Algorithms served:**
  1. FFT-OCT (Classical) -- Analytical baseline
  2. BM4D (PnP) -- Maggioni et al., IEEE TIP 2013
  3. Speckle-DenoiseNet (Deep Learning) -- Devalla et al., BOE 2019
  4. OCTA-Net (Transformer) -- Hybrid U-Net+Transformer, 2023

## Assessment

**Significant mismatch.** The carrier routing `("medical", "Photon") ->
"clinical_optics"` sends DOT to the OCT/fundus/endoscopy algorithm pool. This
is incorrect because DOT and OCT are fundamentally different modalities:

- **OCT** is an interferometric imaging technique that uses FFT to recover depth
  profiles from spectral interference patterns. It produces cross-sectional
  images directly.
- **DOT** is a diffuse tomographic modality that solves an ill-posed inverse
  problem governed by the radiative transfer equation (or its diffusion
  approximation). It reconstructs 3D maps of absorption and scattering
  coefficients from boundary measurements.

The correct algorithms for DOT are:
- Born/Rytov approximation + Tikhonov regularization (Classical)
- L-BFGS with TV regularization (Schweiger & Arridge, PMB 2005)
- PnP with diffusion-model prior
- DOT-specific neural networks (e.g., DeepDOT, Yoo et al. Optica 2020)

The current algorithms (FFT-OCT, Speckle-DenoiseNet, OCTA-Net) are entirely
OCT-specific and have no relevance to DOT.

## Recommended Changes

DOT needs either:
1. A variant-level override in `_VARIANT_OVERRIDES`:
```python
"dot": [
    {"name": "Tikhonov-Born",  "type": "Classical",     ...},
    {"name": "L-BFGS-TV",      "type": "Classical",     ...},
    {"name": "PnP-Diffusion",  "type": "PnP",           ...},
    {"name": "DeepDOT",        "type": "Deep Learning", ...},
]
```
2. Or a more targeted carrier routing that distinguishes OCT (interferometric)
   from DOT (diffuse tomography). DOT could route to "computational" instead,
   which has Tikhonov/PnP-RED/DIP/SwinIR -- a better generic fit.

**File to modify:** `/home/spiritai/pwm/Physics_World_Model/platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`

## Verdict

Code changes recommended. The OCT algorithm pool (FFT-OCT, Speckle-DenoiseNet,
OCTA-Net) is completely inappropriate for diffuse optical tomography. DOT is a
tomographic inverse problem, not an interferometric imaging technique.
