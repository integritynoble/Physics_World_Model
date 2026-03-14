---
modality: coded_lensless
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct a 2D image from a coded lensless measurement, given the binary mask pattern and calibrated diffuser PSF.

# Plan

1. Load measurement y, binary mask M, and calibrated PSF H
2. Construct composite operator: A = H * diag(M)
3. Compute FFT-domain operators for efficient inversion
4. Apply Wiener deconvolution as initialization
5. Refine with FISTA+TV (80 iterations)
6. Output reconstructed image

# Action

## Algorithm: FISTA + Total Variation

**Initialization**: Wiener deconvolution of coded measurement

**Iterative refinement**:
```
# Composite forward operator: A(x) = H * (M ⊙ x)
# Adjoint: A^T(y) = M ⊙ (H^T * y)

for k = 1,...,80:
    gradient = M ⊙ (H^T * (H * (M ⊙ x) - y))  # gradient of ||A(x)-y||^2
    z = prox_TV(x - step * gradient, tv_weight)   # TV proximal step
    x = z + ((t_prev - 1) / t) * (z - z_prev)     # FISTA momentum
```

**Parameters**:
- `n_iterations`: 80
- `step_size`: 1/L where L = max(|H|^2) (Lipschitz constant, mask doesn't affect L since ||M||≤1)
- `tv_weight`: 0.005
- `initialization`: Wiener (SNR=150)

**Convergence guarantee**: FISTA converges at rate O(1/k^2) for convex problems. The composite operator A = H * diag(M) is linear, so data fidelity is convex with L = max(|H|^2).

## Canonical Chain

$M \to C \to D$ — Modulate (binary mask) → Convolve (PSF) → Detect (intensity)

## Mismatch Handling

- **Mask misalignment**: corrected via cross-correlation registration of mask pattern
- **PSF calibration error**: addressed by using measured PSF directly
- **Depth-dependent PSF**: not corrected (single-plane reconstruction)

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (FISTA O(1/k^2) for convex)
- **expected_psnr_db**: ~13 (coded mask improves conditioning over plain lensless ~8 dB)
