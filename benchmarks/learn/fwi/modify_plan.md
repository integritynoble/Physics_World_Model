# Modify Plan: fwi

## Current State

- **Category:** experimental_science
- **Carrier:** Seismic/Acoustic
- **Score key:** experimental_science
- **Algorithms assigned:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Partially appropriate -- generic experimental_science pool; FWI-specific
algorithms exist and would be better**

Full-Waveform Inversion is a major geophysics inverse problem with a rich
published algorithm landscape. The current generic experimental_science pool
(Tikhonov, PnP-RED, ResUNet, SwinIR) is not wrong but is generic and misses
domain-specific methods:

- **Tikhonov**: Acceptable as a classical baseline but FWI typically uses
  L-BFGS gradient descent on the waveform misfit functional, not Tikhonov.
- **PnP-RED**: Generic, not widely used in FWI literature.
- **ResUNet**: Generic image-domain network.
- **SwinIR**: An image restoration transformer, not FWI-specific.

Better FWI-specific algorithms:
- **L-BFGS FWI** (classical gradient-based, the standard)
- **TV-regularized FWI** (Anagaw & Sacchi, 2012)
- **InversionNet** (Wu & Lin, 2019) -- CNN-based velocity inversion
- **VelocityGAN** (Zhang & Alkhalifah, 2022) -- GAN-based FWI
- **WISE** (Huang et al., 2024) -- Wavefield-Informed Seismic Estimator

## Code Changes Needed

**Add FWI-specific variant override in `_algorithm_catalog.py`:**

```python
"fwi": [
    {"name": "L-BFGS FWI",     "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Pratt et al., Geophysics 1998"},
    {"name": "TV-Reg FWI",     "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Anagaw & Sacchi, Geophysics 2012"},
    {"name": "InversionNet",   "type": "Deep Learning", "mask_aware": False, "params": "10M",  "source": "Wu & Lin, IEEE TGRS 2019"},
    {"name": "VelocityGAN",    "type": "Transformer",   "mask_aware": True,  "params": "20M",  "source": "Zhang & Alkhalifah, IEEE TGRS 2022"},
],
```
