# Modify Plan: ocean_color (Ocean Color Remote Sensing)

## Current State

- **Category:** remote_sensing
- **Carrier:** Photon
- **Score key:** computational (routed via `_CARRIER_ROUTING[("remote_sensing", "Photon")]`)
- **Algorithms served (4):**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. Deep Image Prior (Deep Learning) -- Ulyanov et al., CVPR 2018
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Acceptable.** Ocean color remote sensing retrieves water-leaving radiance and
derived products (chlorophyll-a, CDOM, TSS) from satellite multispectral measurements.
The inverse problem involves atmospheric correction and bio-optical inversion.

The routing correctly avoids the SAR-specific algorithms (Matched Filter, SAR-BM3D, etc.)
by using the `computational` pool for Photon-carrier remote sensing. The generic
algorithms are reasonable:

- **Tikhonov** is a standard regularization for spectral inversion problems.
- **PnP-RED** applies well to image restoration with spectral priors.
- **Deep Image Prior** and **SwinIR** are generic but applicable to spatial-spectral
  reconstruction and super-resolution.

More domain-specific algorithms would include:
- OC-SMART (Fan et al., RSE 2021) -- neural network atmospheric correction
- QAA (Lee et al., AO 2002) -- quasi-analytical bio-optical inversion
- SeaDAS L2gen -- NASA's standard ocean color processing

But the computational pool is a reasonable generic approximation.

## Verdict

No code changes needed.
