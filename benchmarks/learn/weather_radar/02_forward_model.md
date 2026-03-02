# 02 — Forward Model: Weather / Doppler Radar

## 1. The Forward Model

### 1.1 Model Type: `nonlinear_operator`

The forward model is **nonlinear**: y = f(x) + n, where f is a nonlinear mapping. This means superposition does not hold, and iterative linearisation (Newton, Gauss-Newton) or specialised algorithms are needed for reconstruction.

### 1.2 Signal Equation

```
s(t) = Σ_n  σ_n · exp(-j4πf_c R_n(t)/c) · rect(t/T)
```

### 1.3 Physics Engine

This modality uses the **`remote_sensing_sar`** category module:
Range-Doppler / synthetic aperture.

---

## 2. Mismatch Parameters

The PWM benchmark introduces physics mismatch between the true acquisition
and what algorithms assume. This tests algorithm robustness.

### Mismatch Parameter Table

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Calibration bias | 0.0 | -1.0 – 1.0 | dBZ |
| Beam elevation error | 0.0 | -0.2 – 0.2 | deg |
| Attenuation correction | 1.0 | 0.9 – 1.1 | - |
| Ground clutter leakage | 0.0 | 0.0 – 0.05 | - |


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

The PWM benchmark for Weather / Doppler Radar uses:

- **Forward model**: `remote_sensing_sar` with `nonlinear_operator` operator
- **Default solver**: `reflectivity_estimation`
- **Metrics**: ['psnr', 'ssim']
- **Primary metric**: psnr

---

*Previous: [01 — Physics Fundamentals](01_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
