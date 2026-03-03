# Modify Plan -- adaptive_optics

## Algorithm Catalog Review

**Category:** experimental_science | **Carrier:** Photon | **Score key:** experimental_science

| Algorithm | Type | Source |
|-----------|------|--------|
| Tikhonov | Classical | Analytical baseline |
| PnP-RED | PnP | Romano et al., IEEE TIP 2017 |
| ResUNet | Deep Learning | Residual U-Net baseline |
| SwinIR | Transformer | Liang et al., ICCVW 2021 |

### Domain Appropriateness

**Poor fit.** Adaptive optics is a wavefront sensing and correction problem, not a generic experimental science signal recovery problem. The `experimental_science` category gives generic deconvolution/restoration algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR) that do not reflect the AO-specific reconstruction pipeline.

AO reconstruction involves:
- Wavefront estimation from Shack-Hartmann or pyramid WFS data
- PSF estimation/deconvolution from estimated wavefront
- Iterative blind deconvolution when PSF is uncertain

Appropriate algorithms would include:
- **Classical:** Wiener deconvolution with Kolmogorov PSF model, or modal wavefront reconstruction (Zernike least-squares)
- **PnP:** PnP-ADMM with atmospheric PSF prior
- **Deep Learning:** Deep wavefront sensing (Nishizaki et al., Opt. Express 2019), or PSF-aware deconvolution nets
- **Transformer:** Vision Transformer for PSF-blind deconvolution

**Issues:**
1. **All 4 algorithms are generic image restoration** -- none reference AO, wavefront sensing, or atmospheric turbulence.
2. **ResUNet source vague** -- "Residual U-Net baseline" is not a citable reference.
3. **SwinIR** is a general image restoration transformer, not AO-specific.
4. **check.md H1 flagged dataset domain mismatch** -- SEG/EAGE Salt Model citation suggests geophysics test data, not AO.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Adjoint" and "PnP-ADMM" as solvers -- neither matches the leaderboard algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR) or the AO domain.

## Proposed Changes

1. **`_algorithm_catalog.py`**: Either (a) add carrier routing for `("experimental_science", "Photon")` to a new `ao_optics` pool with AO-specific algorithms, or (b) create a dedicated `adaptive_optics` variant override with wavefront reconstruction methods.
2. **`_algorithm_catalog.py`**: Replace generic algorithms with AO-domain methods: Zernike least-squares (classical), PnP-ADMM with PSF prior (PnP), WFNet/Deep-WFS (deep learning), AO-ViT (transformer).
3. **`03_reconstruction_algorithms.md`**: Rewrite with AO-specific solver descriptions.
4. **Fix ResUNet source** to a real citation.

**Priority:** HIGH -- the algorithms are entirely generic and do not reflect the unique wavefront correction physics of adaptive optics. However, note that adding `("experimental_science", "Photon")` routing would also affect other photon-carrier experimental_science modalities, so a variant override may be safer.
