---
modality: spectral_lensless
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct L=8 spectral bands from a single coded+dispersed lensless measurement, given the binary mask pattern, dispersion shifts, and calibrated diffuser PSF.

# Plan

1. Load measurement y, binary mask M, dispersion shifts {W_b}, and calibrated PSF H
2. Construct composite operator: A(x_1,...,x_L) = sum_b H * W_b(M . x_b)
3. Apply GAP-TV (Generalized Alternating Projection with TV) for spectral cube recovery
4. Output L=8 reconstructed spectral bands

# Action

## Algorithm: GAP-TV (Generalized Alternating Projection with TV)

**Initialization**: Back-projection: x_b^0 = M . W_b^T(H^T * y) / L

**Iterative refinement (GAP splitting)**:
```
for k = 1,...,60:
    # Forward projection
    y_est = sum_b H * W_b(M . x_b^k) / L

    # Residual back-projection (gap update)
    r = H^T * (y - y_est)
    for b = 1,...,L:
        v_b = x_b^k + M . W_b^T(r) / L

    # TV denoising per band
    for b = 1,...,L:
        x_b^{k+1} = denoise_tv(v_b, weight=tv_weight)
```

**Parameters**:
- `n_iterations`: 60
- `tv_weight`: 0.01
- `n_bands`: 8
- `initialization`: back-projection

**Convergence guarantee**: GAP converges for consistent linear systems under TV regularization. The composite operator is linear with well-defined adjoint A^T(y) = {M . W_b^T(H^T * y)}_b.

## Canonical Chain

$M \to W \to C \to \Sigma \to D$ -- Modulate (mask) -> Disperse (prism) -> Convolve (PSF) -> Accumulate (sum) -> Detect (intensity)

## Mismatch Handling

- **Mask misalignment**: corrected via cross-correlation registration of mask pattern
- **Dispersion calibration error**: corrected via spectral calibration with narrowband sources
- **PSF calibration error**: addressed by using measured PSF directly

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (GAP for convex problems)
- **expected_psnr_db**: ~17 (8:1 compression + dispersion cross-talk + diffuser ill-conditioning)
