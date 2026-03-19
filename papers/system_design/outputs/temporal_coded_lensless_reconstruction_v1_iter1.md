---
modality: temporal_coded_lensless
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct T=8 video frames from a single coded lensless measurement, given the temporal mask patterns and calibrated diffuser PSF.

# Plan

1. Load measurement y, T=8 temporal mask patterns m_t, and calibrated PSF H
2. Construct composite operator: A(x_1,...,x_T) = sum_t H * (m_t ⊙ x_t)
3. Apply GAP-TV (Generalized Alternating Projection with TV) for video recovery
4. Output T=8 reconstructed video frames

# Action

## Algorithm: GAP-TV (Generalized Alternating Projection with TV)

**Initialization**: Zero-filled back-projection: x_t^0 = m_t ⊙ (H^T * y) / T

**Iterative refinement (GAP splitting)**:
```
for k = 1,...,60:
    # Forward projection
    y_est = sum_t H * (m_t ⊙ x_t^k)

    # Residual back-projection (gap update)
    for t = 1,...,T:
        r_t = m_t ⊙ (H^T * (y - y_est))
        v_t = x_t^k + r_t / T

    # TV denoising per frame
    for t = 1,...,T:
        x_t^{k+1} = denoise_tv(v_t, weight=tv_weight)
```

**Parameters**:
- `n_iterations`: 60
- `tv_weight`: 0.01
- `n_frames`: 8
- `initialization`: zero-filled back-projection

**Convergence guarantee**: GAP converges for consistent linear systems under TV regularization. The composite operator is linear with well-defined adjoint A^T(y) = {m_t ⊙ (H^T * y)}_t.

## Canonical Chain

$M \to C \to \Sigma \to D$ — Modulate (temporal mask) → Convolve (PSF) → Accumulate (sum) → Detect (intensity)

## Mismatch Handling

- **Temporal jitter**: corrected via synchronization calibration
- **PSF calibration error**: addressed by using measured PSF directly
- **Mask pattern error**: corrected via direct measurement of actual patterns

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (GAP for convex problems)
- **expected_psnr_db**: ~20 (8:1 compression + diffuser ill-conditioning)
