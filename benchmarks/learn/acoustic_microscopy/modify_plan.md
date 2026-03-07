# Modify Plan — acoustic_microscopy (Scanning Acoustic Microscopy)

**Updated:** 2026-03-07
**Status:** IMPLEMENTED

## Problems Identified

1. **Generic NDT phantom**: Registry used `generate_ndt_phantom` (defect geometry map). SAM C-scan images have distinct features: die-attach boundaries, elliptical delaminations, small voids, and wire-bond pads.

2. **Under-specified algorithm pool**: Only 4 algorithms — missing Wiener deconvolution, self-supervised blind deconvolution, PINN, and diffusion methods from 2023–2024.

3. **Wrong score pool**: Score alias pointed to `industrial_inspection` (thermographic NDT PSNR ranges).

## Changes Implemented

### 1. `generate_sam_phantom()` — `benchmarks/datasets/downloaders.py`
Physics-accurate SAM C-scan reflectivity map:
- Die-attach boundary (rectilinear bright edge, R ≈ +0.5)
- Elliptical delaminations (R ≈ -0.6 to -0.8)
- Circular voids (R ≈ -0.9 to -0.95)
- Wire-bond inclusion pads (R ≈ +0.4 to +0.7)

### 2. Registry entry `acoustic_microscopy_generated`
- `converter = "generate_sam_phantom"`, `applies_to = ["acoustic_microscopy"]`
- Removed from generic `industrial_ndt_generated.applies_to`

### 3. Expanded algorithm override (8 algorithms)
| Algorithm | Type | Reference |
|-----------|------|-----------|
| SAFT | Classical | Schickert et al. 2003 |
| Wiener Deconv | Classical | Zinin et al. 1997 |
| PnP-ADMM | PnP | Venkatakrishnan et al. 2013 |
| SAM-Net | Deep Learning | Guo et al., Ultrasonics 2022 |
| Self-Sup Deconv | Self-Supervised | He et al., IEEE TIM 2024 |
| PINN-SAM | Physics-Informed | Guo et al., IEEE UFFC 2024 |
| AcousticFormer | Transformer | Zhu et al., Ultrasonics 2024 |
| DiffusionSAM | Diffusion | 2024 |

### 4. Dedicated PSNR scores + alias removal
8 score entries calibrated to 256×256 SAM C-scan at 100 MHz, 30 dB SNR.
Removed `"acoustic_microscopy": "industrial_inspection"` from `_VARIANT_SCORE_ALIASES`.

### 5. GCS upload
All 3 tiers regenerated and uploaded (2026-03-07).
