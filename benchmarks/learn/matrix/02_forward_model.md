# 02 — Forward Model: Generic Matrix Sensing

## 1. The Forward Model

### 1.1 Model Type: `explicit_matrix`

The forward model uses an **explicit matrix**: y = Φx + n, where Φ is a known measurement matrix (e.g., random Gaussian, Hadamard). The matrix is stored and applied directly, enabling compressed sensing approaches.

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
| Matrix perturbation | 0.0 | 0.0 – 10.0 | A |
| Condition number change | 0.0 | 0 – 0 | - |


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

The PWM benchmark for Generic Matrix Sensing uses:

- **Forward model**: `compressive_mask` with `explicit_matrix` operator
- **Default solver**: `fista_l2`
- **Metrics**: ['psnr', 'ssim']
- **Primary metric**: psnr

---

*Previous: [01 — Physics Fundamentals](01_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
