# Modify Plan: xrf_imaging

## Current State

- **Category:** industrial_inspection
- **Carrier:** X-ray
- **Score key:** industrial_inspection
- **Algorithms assigned:**

| Name       | Type           | Source                  |
|------------|----------------|-------------------------|
| TSR        | Classical      | Shepard et al., 2003    |
| PnP-ADMM  | PnP            | ADMM + denoiser prior   |
| DefectNet  | Deep Learning  | U-Net for NDT, 2021     |
| LSTM-NDT   | Recurrent      | Fang et al., 2022       |

## Assessment

**Inappropriate -- needs changes.**

XRF imaging is an elemental-mapping technique based on characteristic X-ray fluorescence emission. It is fundamentally a spectral deconvolution and spatial mapping problem, completely different from thermal NDT. The current algorithms are wrong:

1. **TSR** (Thermographic Signal Reconstruction) is specific to pulsed thermography temporal analysis. It has no connection to XRF spectral processing. Wrong.
2. **PnP-ADMM** is a generic optimization framework -- acceptable for any inverse problem.
3. **DefectNet** is a defect-detection CNN for visual/thermal NDT imagery. XRF imaging reconstructs elemental concentration maps from fluorescence spectra, not defect images. Wrong.
4. **LSTM-NDT** is for thermographic time-series analysis. Irrelevant to XRF. Wrong.

XRF imaging should use algorithms from the XRF/spectroscopy/elemental-mapping literature:
- **Classical:** Least-squares spectral fitting (PyMCA-style, Sole et al., 2007) or fundamental-parameters (FP) method
- **PnP:** PnP with spectral denoiser (PnP-BM3D or PnP-DnCNN on spectral channels)
- **Deep Learning:** XRF-Net or similar CNN for elemental quantification (e.g., Falkenberg et al.)
- **Domain-specific:** Physics-informed spectral unmixing networks

Note: `xrf_imaging` has the same mismatch as `xray_ndt` because both are in the `industrial_inspection` category and share the same carrier (`X-ray`), so they get the identical thermal-NDT pool. A carrier-routing fix for `("industrial_inspection", "X-ray")` would affect both.

## Proposed Changes

Add a variant override in `_algorithm_catalog.py`:

```python
_VARIANT_OVERRIDES["xrf_imaging"] = [
    {"name": "FP-Quantify",   "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Fundamental parameters method, Sole et al. 2007 (PyMCA)"},
    {"name": "PnP-BM3D",      "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
    {"name": "XRF-UNet",      "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "U-Net for elemental mapping, 2022"},
    {"name": "SpectraFormer",  "type": "Transformer",   "mask_aware": True,  "params": "6M",   "source": "Transformer for spectral unmixing, 2024"},
]
```

Alternatively, since the `spectroscopy` pool (SG-ALS, PnP-DnCNN, CDAE, Cascade-UNet) would be a much better generic fit than industrial_inspection, a carrier-routing rule could redirect:

```python
("industrial_inspection", "X-ray"): "spectroscopy",  # or a new xrf-specific pool
```

However, this would also affect `xray_ndt`, so individual variant overrides are safer.

### Files to modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- add variant override for `xrf_imaging`

### Risk
- Low. Only changes the leaderboard algorithm names for this single modality.
