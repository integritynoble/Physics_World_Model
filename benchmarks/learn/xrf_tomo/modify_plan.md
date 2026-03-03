# Modify Plan: xrf_tomo

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** X-ray
- **Score key:** scientific_instrumentation
- **Algorithms assigned:**

| Name          | Type           | Source                           |
|---------------|----------------|----------------------------------|
| Deconv        | Classical      | Analytical baseline              |
| PnP-BM3D     | PnP            | Danielyan et al., 2012           |
| ResNet-Calib  | Deep Learning  | ResNet for calibration, 2022     |
| CalibFormer   | Transformer    | Transformer calibration, 2024    |

## Assessment

**Partially appropriate -- could be improved but acceptable.**

XRF tomography (X-ray fluorescence computed tomography) reconstructs 3D elemental concentration maps from angle-dependent XRF emission spectra. The `scientific_instrumentation` pool provides generic instrument-science algorithms:

1. **Deconv** (Deconvolution) -- Acceptable as a generic classical baseline. In XRF-CT, the classical method is typically filtered backprojection with self-absorption correction, which is a form of deconvolution. Reasonable but imprecise naming.
2. **PnP-BM3D** -- A solid generic PnP approach. BM3D denoising is applicable to any imaging modality. Good fit.
3. **ResNet-Calib** -- Described as "ResNet for calibration, 2022". While XRF-CT does require careful calibration (self-absorption, fluorescence yield, detector dead time), this is a vague reference. Passable but not ideal.
4. **CalibFormer** -- "Transformer calibration, 2024" is similarly vague. Acceptable as a placeholder.

The algorithms are not wrong per se, but they are generic scientific-instrument algorithms rather than XRF-CT-specific ones. Better alternatives would reference the XRF-CT literature directly:
- FBP with self-absorption correction (Schroer, 2001)
- SIRT/ART iterative reconstruction (standard for limited-angle XRF-CT)
- Deep-learning-based self-absorption correction networks
- Total-variation regularized reconstruction for sparse-angle XRF-CT

## Proposed Changes

No code changes needed.

The current assignment is acceptable. The algorithms are generic but not wrong -- they represent valid classical, PnP, DL, and transformer approaches to an instrument-science reconstruction problem. The `scientific_instrumentation` pool is a reasonable home for XRF-CT, which is indeed a synchrotron/beamline scientific instrument.

If greater domain specificity is desired in the future, consider adding a variant override with XRF-CT-specific algorithms:

```python
# Optional future improvement (not required now):
_VARIANT_OVERRIDES["xrf_tomo"] = [
    {"name": "FBP-SA",        "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "FBP + self-absorption correction, Schroer 2001"},
    {"name": "PnP-BM3D",      "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
    {"name": "XRF-UNet",      "type": "Deep Learning", "mask_aware": False, "params": "4M",   "source": "U-Net for XRF-CT, 2023"},
    {"name": "SA-Net",         "type": "Deep Unrolling","mask_aware": True,  "params": "6M",   "source": "Self-absorption correction network, 2024"},
]
```
