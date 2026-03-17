# Specification: CT Reconstruction (LoDoPaB)

## Domain
domain: 2D image grid
geometry: 128 x 128 pixels
pixel_size: 0.7 mm (approximate, from LoDoPaB-CT)
dimension: 2

## Equations
# Forward model (Radon transform):
#   y = A * x + eta
#   A = Radon transform (parallel beam, n_angles projections)
#   eta ~ Poisson noise (realistic clinical noise model)
#
# Inverse problem (reconstruction):
#   min_x 0.5 * ||A*x - y||_2^2 + lambda_1 * ||x||_TV + lambda_2 * ||x||_2^2
#   subject to: x >= 0 (non-negativity)
#
# TV = total variation = sum |nabla x|
# Solved via ADMM or proximal gradient

equations: |
  forward: y = Radon(x) + eta
  inverse: min_x 0.5 * ||Radon(x) - y||_2^2 + lambda_TV * TV(x) + lambda_L2 * ||x||_2^2
  constraint: x >= 0

parameters:
  n_angles: 128           # number of projection angles (0 to pi)
  n_detectors: 183        # detector elements per angle
  geometry: parallel_beam
  noise_model: Poisson    # realistic clinical noise
  lambda_TV: 1.0e-3       # TV regularization weight (auto-tuned via Morozov)
  lambda_L2: 1.0e-5       # Tikhonov regularization weight
  max_iterations: 500     # ADMM iterations
  image_size: [128, 128]

## Boundary Conditions
boundary: |
  non_negativity: x(i,j) >= 0 for all pixels
  support: x = 0 outside patient body contour (if known)

## Initial Conditions
initial: |
  x_0 = filtered back-projection (FBP) result (warm start)

## Observables
# 1. Reconstructed image x_hat on 128x128 grid
# 2. PSNR: 10 * log10(max(x)^2 / MSE)  (dB)
# 3. SSIM: structural similarity index
# 4. Relative error: ||x_hat - x_true|| / ||x_true||
# Validated against LoDoPaB-CT dataset (Leuschner et al., 2021)
# 200 real clinical cases from Siemens scanner
observables:
  - reconstructed_image: x_hat on 128x128 grid
  - PSNR: dB
  - SSIM: dimensionless [0, 1]
  - relative_error: dimensionless

## Tolerance
tolerance:
  PSNR_minimum: 30.0      # dB
  SSIM_minimum: 0.85
  metric: PSNR

## Primitives Required
primitives: [integrate, solve_linear, optimize, constrain, discretize]
# int (Radon transform = line integrals), L (normal equations),
# O (ADMM/proximal gradient for TV), B (non-negativity), G (pixel grid)

## Task Instance Variations
# Vary: n_angles (32-512), noise level (low to clinical), geometry
#        (parallel, fan-beam, cone-beam), regularization (TV, wavelet, learned),
#        image size (64x64 to 512x512)
# Public: LoDoPaB 128-angle with ground truth provided
# Dev: sparse-angle (64, 32), elevated noise
# Hidden: limited-angle (missing wedge), metal artifacts, truncated projections
