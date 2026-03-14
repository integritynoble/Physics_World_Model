---
modality: lensless
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct a 2D image from a lensless camera measurement, given the blurred measurement and calibrated PSF.

# Plan

1. Load measurement y and calibrated PSF H
2. Compute FFT-domain operators: H, H*, |H|^2
3. Apply Wiener deconvolution as initialization: x0 = IFFT(H* Y / (|H|^2 + 1/SNR))
4. Refine with FISTA+TV (80 iterations, step=1/L, tv_weight=0.003)
5. Output reconstructed image

# Action

## Algorithm: FISTA + Total Variation

**Initialization**: Wiener deconvolution (closed-form, FFT-domain)

**Iterative refinement**:
```
for k = 1,...,80:
    gradient = H^T(Hx - y)           # FFT-domain gradient
    z = prox_TV(x - step * gradient)  # TV proximal step
    x = z + ((t_prev - 1) / t) * (z - z_prev)  # FISTA momentum
```

**Parameters**:
- `n_iterations`: 80
- `step_size`: 1/L where L = max(|H|^2) (Lipschitz constant)
- `tv_weight`: 0.003
- `initialization`: Wiener (SNR=200)

**Convergence guarantee**: FISTA converges at rate O(1/k^2) for convex problems with Lipschitz-continuous gradient. The PSF convolution operator is linear, so the data fidelity term is convex with L = max(|H|^2).

## Canonical Chain

$C \to D$ — Convolution (PSF) → Detect (intensity)

## Mismatch Handling

- **PSF calibration error**: addressed by using measured PSF directly
- **Depth-dependent PSF**: not corrected (single-plane reconstruction)

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (FISTA O(1/k^2) for convex)
- **expected_psnr_db**: ~8 (limited by PSF conditioning, not algorithm)
