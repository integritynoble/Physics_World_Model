# Modify Plan -- seismic_tomo

## Current State

- **Category:** experimental_science
- **Carrier:** Seismic
- **Routing:** No carrier routing for `("experimental_science", "Seismic")` -> falls to `_CATEGORY_ALGORITHMS["experimental_science"]`
- **Score key:** experimental_science
- **Algorithms assigned:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Partially appropriate: Acceptable but not domain-optimal.**

Seismic tomography reconstructs subsurface velocity or attenuation models from seismic travel times or waveforms. The inverse problem is structurally a tomographic reconstruction from line integrals (travel-time tomography) or a nonlinear PDE-constrained inversion (full waveform inversion).

- **Tikhonov**: Regularized least squares is indeed the classical approach for seismic travel-time tomography. This is a correct and standard baseline (damped least squares inversion).
- **PnP-RED**: Generic plug-and-play regularization. Not commonly used in geophysics, but functionally applicable.
- **ResUNet**: U-Net architectures have been adapted for seismic inversion (e.g., Yang & Ma, Geophysics 2019). Acceptable.
- **SwinIR**: Image restoration transformer. Not standard in seismology, but applicable as a generic learned prior.

Domain-specific algorithms would include: LSQR (Paige & Saunders, 1982), conjugate gradient on normal equations, full waveform inversion (Tarantola, 1984; Virieux & Operto, 2009), and physics-informed neural networks for seismic inversion (e.g., InversionNet by Wu & Lin, 2020). However, the current assignment is functionally correct since Tikhonov IS the classical seismic tomography approach, and the others are generic but applicable inverse-problem solvers.

## Plan

No code changes needed. Tikhonov as the classical baseline is domain-correct for seismic travel-time tomography. The other algorithms are generic but functional.
