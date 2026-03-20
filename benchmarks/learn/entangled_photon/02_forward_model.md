# 02 — Forward Model: Entangled Photon Microscopy

## 1. The Forward Model

### 1.1 Model Type: `nonlinear_operator`

The forward model is **nonlinear**: y = f(x) + n, where f is a nonlinear mapping. This means superposition does not hold, and iterative linearisation (Newton, Gauss-Newton) or specialised algorithms are needed for reconstruction.

### 1.2 Signal Equation

```
y = PSF ⊛ x + noise  (⊛ = convolution)
```

### 1.3 Physics Engine

This modality uses the **`compressive_mask`** category module:
Coded aperture / compressive sensing.

---

## 2. Mismatch Parameters

The PWM benchmark introduces physics mismatch between the true acquisition
and what algorithms assume. This tests algorithm robustness.

### Mismatch Parameter Table

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Pair generation rate | 0.0 | 0.1 – 10.0 | - |
| Coincidence window | 1.0 | 0.1 – 10.0 | ns |
| Accidental coincidence rate | 0.0 | 0.0 – 20.0 | - |
| Photon loss (per arm) | 0.0 | 0.0 – 6.0 | dB |


### Mismatch Philosophy

- **Public tier**: mild mismatch — algorithms should perform well
- **Dev tier**: moderate mismatch — exposes fragility
- **Hidden tier**: severe mismatch — tests true robustness

The gap between public and hidden tier performance reveals how sensitive
an algorithm is to model errors.

---

## 3. Noise Model

The noise model combines:

1. **Signal-dependent noise**: Poisson (photon counting), speckle
   (coherent), or multiplicative noise
2. **Signal-independent noise**: Gaussian read noise, dark current,
   thermal noise
3. **Systematic errors**: background, fixed-pattern noise, calibration
   errors

The relative importance depends on the imaging regime:
- Low-signal: Poisson-dominated → shot noise is the bottleneck
- High-signal: calibration-dominated → mismatch is the bottleneck

---

## 4. Why Reconstruction is Challenging

The inverse problem y = A(x) + n is challenging because:

1. **Ill-conditioning**: small changes in y cause large changes in x
2. **Underdetermination**: fewer measurements than unknowns (compressed sensing)
3. **Nonlinearity**: the forward model may be nonlinear
4. **Model mismatch**: A_true ≠ A_assumed
5. **Noise amplification**: regularisation is needed to control noise

---

## 5. Connection to PWM Framework

The PWM benchmark for Entangled Photon Microscopy uses:

- **Forward model**: `compressive_mask` with `nonlinear_operator` operator
- **Default solver**: `coincidence_recon`
- **Metrics**: ['psnr', 'ssim']
- **Primary metric**: psnr

---

*Previous: [01 — Physics Fundamentals](01_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
