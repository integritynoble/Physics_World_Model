# 02 — Forward Model: Portal Imaging (EPID)

## 1. The Forward Model

### 1.1 Model Type: `linear_operator`

The forward model is **linear**: y = Ax + n, where A is a linear operator (matrix or linear transform). This means superposition holds — doubling the input doubles the output. Many classical reconstruction algorithms (least-squares, CG, FISTA) exploit linearity.

### 1.2 Signal Equation

```
I = I₀ · exp(-∫ μ_compton(E, ρ) dl) + scatter
```

### 1.3 Physics Engine

This modality uses the **`medical_ct_radon`** category module:
Radon transform / projection-based sensing.

---

## 2. Mismatch Parameters

The PWM benchmark introduces physics mismatch between the true acquisition
and what algorithms assume. This tests algorithm robustness.

### Mismatch Parameter Table

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Isocenter shift | 0.0 | -2.0 – 2.0 | mm |
| Beam energy variation | 6.0 | 5.8 – 6.2 | MV |
| Detector sag | 0.0 | -1.0 – 1.0 | mm |
| Scatter kernel width | 5.0 | 3.0 – 7.0 | mm |


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

The PWM benchmark for Portal Imaging (EPID) uses:

- **Forward model**: `medical_ct_radon` with `linear_operator` operator
- **Default solver**: `back_projection`
- **Metrics**: ['psnr', 'ssim']
- **Primary metric**: psnr

---

*Previous: [01 — Physics Fundamentals](01_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
