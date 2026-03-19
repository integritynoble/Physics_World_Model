---
modality: lensless_3d
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct Nz=8 depth planes from a single 2D lensless measurement, given the calibrated depth-dependent PSFs.

# Plan

1. Load measurement y and calibrated per-depth PSFs {H_z}
2. Construct composite operator: A(x_1,...,x_Nz) = sum_z H_z * x_z
3. Apply GAP-TV for volumetric recovery
4. Output Nz=8 reconstructed depth planes

# Action

## Algorithm: GAP-TV (Generalized Alternating Projection with TV)

**Initialization**: Back-projection: x_z^0 = H_z^T * y / Nz

**Iterative refinement (GAP splitting)**:
```
for k = 1,...,60:
    # Forward projection
    y_est = sum_z H_z * x_z^k / Nz

    # Residual back-projection (gap update)
    for z = 1,...,Nz:
        v_z = x_z^k + H_z^T * (y - y_est) / Nz

    # TV denoising per depth plane
    for z = 1,...,Nz:
        x_z^{k+1} = denoise_tv(v_z, weight=tv_weight)
```

**Parameters**:
- `n_iterations`: 60
- `tv_weight`: 0.01
- `n_depth_planes`: 8
- `initialization`: back-projection

**Convergence guarantee**: GAP converges for consistent linear systems under TV regularization. The composite operator is linear with well-defined adjoint A^T(y) = {H_z^T * y}_z.

## Canonical Chain

$C \to \Sigma \to D$ -- Convolve (depth-dependent PSF) -> Accumulate (sum over depth) -> Detect (intensity)

## Mismatch Handling

- **PSF calibration error**: corrected by per-depth point source calibration
- **Depth discretization**: mitigated by using sufficient depth planes
- **Inter-plane crosstalk**: resolved by the diversity of depth-dependent PSFs

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (GAP for convex problems)
- **expected_psnr_db**: ~18 (8:1 depth compression, well-conditioned depth-dependent PSFs)
