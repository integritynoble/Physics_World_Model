# Modify Plan -- brillouin

## Algorithm Catalog Review

**Category:** spectroscopy | **Carrier:** Photon | **Score key:** spectroscopy

| Algorithm | Type | Source |
|-----------|------|--------|
| SG-ALS | Classical | Savitzky-Golay + ALS baseline |
| PnP-DnCNN | PnP | Zhang et al., 2017 |
| CDAE | Deep Learning | Zhang et al., Sensors 2024 |
| Cascade-UNet | Transformer | Physics-informed UNet, 2025 |

### Domain Appropriateness

**Good fit.** Brillouin microscopy measures inelastic light scattering spectra to map viscoelastic properties. The spectroscopy pool provides spectral processing algorithms that are applicable:

- **SG-ALS (Savitzky-Golay + Asymmetric Least Squares)** -- Standard spectral smoothing + baseline correction. Widely used for Raman/Brillouin spectral preprocessing. Appropriate.
- **PnP-DnCNN** -- Zhang et al., 2017 is the DnCNN paper. Applicable as a generic denoising prior for spectral data. Citation is real.
- **CDAE** -- Zhang et al., Sensors 2024. Convolutional Denoising Autoencoder for spectral data. Real venue. Appropriate.
- **Cascade-UNet** -- "Physics-informed UNet, 2025" is a vague source without authors or venue.

Brillouin-specific reconstruction involves Lorentzian peak fitting to extract Brillouin frequency shift and linewidth. The spectroscopy methods address the spectral denoising/baseline correction step, which is a valid preprocessing task before peak fitting.

**Issues:**
1. **Cascade-UNet source vague** -- "Physics-informed UNet, 2025" is not a citable reference. Needs real authors and venue.
2. **Cascade-UNet labeled as "Transformer"** -- A UNet is not a transformer architecture. The type label is misleading.
3. **PnP-DnCNN citation** -- "Zhang et al., 2017" should specify the venue (CVPR 2017 for DnCNN, or IEEE TIP 2017).
4. **No Brillouin-specific methods** -- Lorentzian fitting, VIPA spectrometer calibration methods, and Brillouin-specific deep learning (e.g., Remer & Bhatt, Biomed. Opt. Express 2020) are absent.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Adjoint" and "PnP-ADMM" which do not match the leaderboard (SG-ALS, PnP-DnCNN, CDAE, Cascade-UNet).

## Proposed Changes

1. **`_algorithm_catalog.py`**: Fix Cascade-UNet source to a real citation with authors and venue.
2. **`_algorithm_catalog.py`**: Fix Cascade-UNet type from "Transformer" to "Deep Learning" or "Hybrid", since a UNet is not a transformer.
3. **`_algorithm_catalog.py`**: Add venue to PnP-DnCNN citation (e.g., "Zhang et al., IEEE TIP 2017").
4. **`03_reconstruction_algorithms.md`**: Update to match leaderboard algorithms.

**Priority:** MEDIUM -- algorithms are reasonable for spectroscopy; Cascade-UNet has a vague citation and misleading type label.
