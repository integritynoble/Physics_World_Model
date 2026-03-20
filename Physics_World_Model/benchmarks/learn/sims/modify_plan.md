# Modify Plan -- sims

## Current State

- **Category:** spectroscopy
- **Carrier:** Ion
- **Routing:** No carrier routing for `("spectroscopy", "Ion")` -> falls to `_CATEGORY_ALGORITHMS["spectroscopy"]`
- **Score key:** spectroscopy
- **Algorithms assigned:**
  1. SG-ALS (Classical) -- Savitzky-Golay + ALS baseline
  2. PnP-DnCNN (PnP) -- Zhang et al., 2017
  3. CDAE (Deep Learning) -- Zhang et al., Sensors 2024
  4. Cascade-UNet (Transformer) -- Physics-informed UNet, 2025

## Assessment

**Appropriate: YES.**

Secondary Ion Mass Spectrometry (SIMS) imaging sputters a sample surface with a primary ion beam and analyzes the ejected secondary ions by mass spectrometry to produce spatially-resolved chemical composition maps. The spectral reconstruction problem involves denoising low-count mass spectra, baseline correction, and spectral unmixing -- all tasks that fall within the spectroscopy algorithm family.

- **SG-ALS**: Savitzky-Golay smoothing and asymmetric least squares baseline correction are standard preprocessing steps for mass spectra, including SIMS data. Appropriate.
- **PnP-DnCNN**: Plug-and-play denoising for spectral data. SIMS spectra are often noisy due to low secondary ion yields, making denoising critical. Applicable.
- **CDAE**: Convolutional denoising autoencoder for spectral data. Directly applicable to SIMS spectral denoising.
- **Cascade-UNet**: Physics-informed network for spectral reconstruction. Applicable to SIMS chemical mapping.

The spectroscopy pool is a good fit for SIMS imaging.

## Plan

No code changes needed.

## 2026-03-06 Comprehensive Check Update

- Physics: y(i,j, m/z) = I_primary * Y(m/z, matrix) * T(m/z) * c(i,j) + b_background + n_Poisson; matrix ionization yield Y varies 2-3 orders of magnitude
- Key mismatch: matrix ionization yield Y(m/z), spectrometer transmission T(m/z), depth resolution (beam mixing), primary beam dose accumulation
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — SG-ALS (mass spectral baseline), SVD/PCA (standard for ToF-SIMS multivariate analysis), CDAE (count-map denoising), SpectraFormer (peak deconvolution)
- Note: SVD/MCR-ALS is the gold-standard analysis for SIMS hyperspectral data; its inclusion is essential
