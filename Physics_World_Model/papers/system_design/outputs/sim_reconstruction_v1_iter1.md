---
modality: sim
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct a super-resolved 2D image from 9 SIM raw frames (3 orientations × 3 phases), given the system PSF.

# Plan

1. Load 9 raw frames and system PSF
2. Combine raw frames into effective measurement (sum or frequency-domain separation)
3. Apply Wiener-SIM deconvolution as initialization
4. Refine with PnP-ADMM + TV denoiser (40 iterations, rho=0.5)
5. Output super-resolved image

# Action

## Algorithm: PnP-ADMM with TV Denoiser

**Initialization**: Wiener-SIM (frequency-domain, SNR=50)

**Iterative refinement (ADMM splitting)**:
```
for k = 1,...,40:
    # x-update (FFT-domain closed-form)
    x = IFFT((H* Y + rho * FFT(z - u)) / (|H|^2 + rho))

    # z-update (TV denoising proximal step)
    z = denoise_tv(x + u, weight=tv_weight/rho)

    # dual update
    u = u + x - z
```

**Parameters**:
- `n_iterations`: 40
- `rho`: 0.5 (ADMM penalty parameter)
- `tv_weight`: 1.0 (TV denoising strength)
- `initialization`: Wiener-SIM (SNR=50)

**Convergence guarantee**: ADMM converges for convex problems when both subproblems are solved exactly. The x-update is closed-form (FFT-domain); the z-update uses TV denoising which is a convex proximal operator.

## Canonical Chain

$M \to C \to D$ — Modulation (sinusoidal patterns) → Convolution (PSF) → Detect (intensity)

## Mismatch Handling

- **Pattern frequency error**: mitigated by using measured effective OTF
- **Pattern phase error**: absorbed into sum-image approximation
- **Optical aberrations**: corrected through measured PSF

# Demands

- **feasibility**: yes
- **algorithm_convergence**: yes (ADMM for convex problems)
- **expected_psnr_db**: ~27 (SIM provides ~2× resolution gain)
