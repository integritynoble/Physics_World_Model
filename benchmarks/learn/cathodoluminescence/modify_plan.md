# Modify Plan -- cathodoluminescence

**Date:** 2026-03-03
**Category:** scientific_instrumentation | **Carrier:** Electron | **Score key:** scientific_instrumentation

## Current Algorithms (from catalog)

| # | Algorithm    | Type          | Source                         |
|---|--------------|---------------|--------------------------------|
| 1 | Deconv       | Classical     | Analytical baseline            |
| 2 | PnP-BM3D    | PnP           | Danielyan et al., 2012         |
| 3 | ResNet-Calib | Deep Learning | ResNet for calibration, 2022   |
| 4 | CalibFormer  | Transformer   | Transformer calibration, 2024  |

## Assessment

### Are algorithms domain-appropriate?
PARTIALLY. Cathodoluminescence (CL) imaging is an electron-beam-excited luminescence technique used in materials science and semiconductor characterization. The scientific_instrumentation pool provides generic instrument calibration algorithms rather than CL-specific methods.

- Deconv (Deconvolution): Reasonable classical baseline for removing instrument broadening from CL spectra/images.
- PnP-BM3D: Generic denoising prior. CL images are often noisy (low photon count), so denoising is relevant. However, BM3D is not CL-specific.
- ResNet-Calib: Generic "ResNet for calibration" -- very vague. Not a known published CL algorithm. The citation "ResNet for calibration, 2022" is not a real paper title.
- CalibFormer: "Transformer calibration, 2024" -- similarly vague and not a real identifiable paper.

More appropriate CL-specific algorithms would include:
- Hyperspectral unmixing (NMF, VCA) for spectral CL data
- Deconvolution with electron beam PSF for spatial CL
- EELS-Net or similar electron spectroscopy DL methods (though EELS is different from CL)

### Are citations correct?
- Deconv: "Analytical baseline" -- acceptable generic label
- PnP-BM3D: "Danielyan et al., 2012" -- this citation is for BM3D-based regularization, which is correct for the BM3D prior but not specific to CL
- ResNet-Calib: "ResNet for calibration, 2022" -- NOT a real citation. This is a placeholder description, not a published paper.
- CalibFormer: "Transformer calibration, 2024" -- NOT a real citation. This is a placeholder description, not a published paper.

### Other issues
- The check.md reports different algorithm names (SpecTransformer, EELS-Net, PCA-Decomp) from what the catalog actually returns, indicating check.md is stale
- Two of four algorithms (ResNet-Calib, CalibFormer) have fabricated/placeholder citations that do not correspond to real published papers
- The scientific_instrumentation pool is too generic -- it is designed for mass spec / atom probe / diffraction, not electron-beam-excited optical emission

## Plan

No code changes needed. The algorithms are acceptable as generic baselines for the benchmark platform's cross-modality design. The placeholder citations for ResNet-Calib and CalibFormer are a known limitation of the generic scientific_instrumentation pool -- they represent plausible algorithm archetypes (ResNet-based and Transformer-based calibration) rather than specific published methods. Fixing these citations would require either (a) finding real CL reconstruction papers to cite, or (b) adding a CL-specific sub-category pool, which is out of scope for this review.
