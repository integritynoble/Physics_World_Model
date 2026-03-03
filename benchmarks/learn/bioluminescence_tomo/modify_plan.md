# Modify Plan -- bioluminescence_tomo

## Algorithm Catalog Review

**Category:** experimental_science | **Carrier:** Photon | **Score key:** experimental_science

| Algorithm | Type | Source |
|-----------|------|--------|
| Tikhonov | Classical | Analytical baseline |
| PnP-RED | PnP | Romano et al., IEEE TIP 2017 |
| ResUNet | Deep Learning | Residual U-Net baseline |
| SwinIR | Transformer | Liang et al., ICCVW 2021 |

### Domain Appropriateness

**Moderate fit.** Bioluminescence Tomography (BLT) is a diffuse optical inverse problem -- reconstructing a 3D bioluminescent source distribution from surface photon measurements. The `experimental_science` category gives generic inverse problem algorithms.

- **Tikhonov** -- Valid classical baseline for any ill-posed linear inverse problem, including BLT. Appropriate.
- **PnP-RED** -- Romano et al., IEEE TIP 2017 is a real citation. PnP methods are applicable to BLT. Appropriate.
- **ResUNet** -- Generic deep learning architecture. Applicable but not BLT-specific. Source is vague ("Residual U-Net baseline" is not a real citation).
- **SwinIR** -- Image restoration transformer. BLT outputs are 3D volumetric source maps, not 2D images, so SwinIR is a questionable fit. Citation is real (Liang et al., ICCVW 2021) but the method is for 2D image super-resolution.

BLT-specific algorithms would include:
- Tikhonov with diffusion equation system matrix (appropriate, already present conceptually)
- Adaptive FEM-based reconstruction (Lv et al., PMB 2006)
- Source-permissible region constrained methods (Han et al., Opt. Express 2006)
- Deep learning for BLT (Gao et al., 2018)

**Issues:**
1. **ResUNet source vague** -- "Residual U-Net baseline" needs a real citation.
2. **SwinIR is 2D** -- BLT is a 3D volumetric reconstruction; using a 2D image restoration transformer is technically mismatched.
3. **No BLT-specific methods** -- All algorithms are generic; none reference diffuse optical tomography or bioluminescence.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Adjoint" and "PnP-ADMM" which do not match the leaderboard (Tikhonov, PnP-RED, ResUNet, SwinIR).

## Proposed Changes

1. **`_algorithm_catalog.py`**: Fix ResUNet source to a real citation (e.g., Diakogiannis et al., ISPRS J. 2020 for ResUNet, or a BLT-specific DL reference).
2. **`03_reconstruction_algorithms.md`**: Update to match leaderboard algorithms.
3. **Consider** adding a variant override or sub-category for optical tomography modalities to provide more domain-specific algorithms.

**Priority:** MEDIUM -- Tikhonov and PnP-RED are defensible; ResUNet source is vague; SwinIR is a 2D/3D mismatch concern.
