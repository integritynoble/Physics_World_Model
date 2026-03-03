# Modify Plan -- atom_probe

## Algorithm Catalog Review

**Category:** scientific_instrumentation | **Carrier:** Ion | **Score key:** scientific_instrumentation

| Algorithm | Type | Source |
|-----------|------|--------|
| Deconv | Classical | Analytical baseline |
| PnP-BM3D | PnP | Danielyan et al., 2012 |
| ResNet-Calib | Deep Learning | ResNet for calibration, 2022 |
| CalibFormer | Transformer | Transformer calibration, 2024 |

### Domain Appropriateness

**Moderate fit.** The scientific_instrumentation pool is a catch-all for mass spectrometry, atom probe, diffraction, etc. The algorithms are generic calibration/deconvolution methods, not APT-specific.

Atom Probe Tomography (APT) reconstruction involves:
- 3D spatial reconstruction from detector hit positions + time-of-flight data
- Bas et al. (1995) / Geiser et al. (2007) trajectory reconstruction protocols
- Local electrode corrections and trajectory aberration correction

**Issues:**
1. **Deconv is too generic** -- APT reconstruction is not a deconvolution problem. The classical method should be the Bas protocol or Geiser protocol (standard APT reconstruction).
2. **PnP-BM3D** -- BM3D is an image denoising method. APT data is 3D point cloud + mass spectrum, not a 2D image. Citation (Danielyan et al., 2012) is real but domain-inappropriate.
3. **ResNet-Calib source vague** -- "ResNet for calibration, 2022" is not a citable reference. No authors or venue.
4. **CalibFormer source vague** -- "Transformer calibration, 2024" is not a citable reference. No authors or venue.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Adjoint" and "PnP-ADMM" which also do not match the leaderboard (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) or the APT domain.

## Proposed Changes

1. **`_algorithm_catalog.py`**: Consider adding a variant override for `atom_probe` with APT-specific algorithms: Bas protocol (Classical), regularized reconstruction (PnP), APT-ML (Wei et al., Ultramicroscopy 2019) (Deep Learning), and a transformer variant. Alternatively, add carrier routing for `("scientific_instrumentation", "Ion")`.
2. **`_algorithm_catalog.py`**: At minimum, fix the vague sources for ResNet-Calib and CalibFormer with real citations.
3. **`03_reconstruction_algorithms.md`**: Update to match leaderboard algorithms.

**Priority:** MEDIUM -- algorithms are functional placeholders but not domain-specific to atom probe tomography. The vague citations for ResNet-Calib and CalibFormer are a quality concern.
