---
modality: ct
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct the original attenuation map from a sparse-view (60 angles), low-dose (I0=1e4) CT sinogram. The inverse problem is severely ill-posed due to angular undersampling and high noise. The sinogram is also corrupted by beam hardening and Compton scatter. Use TV-ADMM with mismatch corrections to regularize the reconstruction.

# Plan

1. Apply beam hardening polynomial correction to the sinogram
2. Apply scatter kernel subtraction to remove low-frequency bias
3. Initialize with Filtered Back-Projection (FBP, ramp filter)
4. Run TV-ADMM: alternate data fidelity gradient step and TV proximal step
5. Enforce non-negativity constraint at each iteration
6. Check convergence: ||x_{k+1} - x_k|| / ||x_k|| < 1e-4 or max 100 iterations

# Action

## Algorithm: TV-ADMM

**Type**: Variational

**References**:
  - Sidky & Pan, 'Image reconstruction in circular cone-beam CT by constrained TV minimization', Phys. Med. Biol. 53(17), 2008
  - Kak & Slaney, 'Principles of Computerized Tomographic Imaging', IEEE Press 1988

### Algorithm Steps

**Step 1: Mismatch Pre-Correction**

Apply beam hardening polynomial linearization and scatter subtraction to the raw sinogram before reconstruction.

$$
y_corr = a0 + a1*y + a2*y^2 - 0.1*G_sigma(y)
$$
Parameters:
  - `a0`: 0.0
  - `a1`: 1.0
  - `a2`: -0.05
  - `scatter_sigma`: 20.0

**Step 2: FBP Initialization**

Compute initial estimate via filtered back-projection with a Ram-Lak filter.

$$
x_0 = R^{-1}_{FBP}(y_corr)
$$
Parameters:
  - `filter`: ramp

**Step 3: Data Fidelity Gradient**

Compute the gradient of 0.5*||Rx - y_corr||^2 where R is the Radon operator.

$$
grad = R^T(Rx_k - y_corr)
$$

**Step 4: TV Proximal Step**

Apply the proximal operator of the isotropic total variation penalty. This preserves edges while denoising flat regions.

$$
x_{k+1} = prox_{lambda*TV}(x_k - eta * grad)
$$
Parameters:
  - `lambda_tv`: 0.01
  - `eta`: 0.005

**Step 5: Non-Negativity Projection**

Project onto the non-negative orthant since attenuation coefficients are non-negative.

$$
x_{k+1} = max(x_{k+1}, 0)
$$

### Mismatch Corrections

- `beam_hardening` [high]: Polychromatic cupping artifact from 80 kVp source
  Correction: 2nd-order polynomial linearization: y_corr = y - 0.05*y^2
- `scatter` [medium]: Compton scatter low-frequency bias (SPR ~0.3)
  Correction: Subtract 10% of Gaussian-blurred sinogram (sigma=20 px)

**Convergence**: ||x_{k+1} - x_k||_2 / ||x_k||_2 < 1e-4, or max 100 iterations

### Hyperparameters

- `lambda_tv`: 0.01
- `step_size`: 0.005
- `num_iterations`: 100
- `filter`: ramp

# Demands

- **feasibility**: yes
- **budget_feasible**: N/A
- **algorithm_convergence**: yes

**Comments**: TV-ADMM converges reliably for sparse-view CT. The TV penalty effectively suppresses streak artifacts from angular undersampling. Beam hardening and scatter pre-corrections are essential at this noise level.
