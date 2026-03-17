# Specification: Seismic Full-Waveform Inversion (Marmousi-2)

## Domain
domain: 2D subsurface velocity model
geometry: x in [0, 17000] m, z in [0, 3500] m
dimension: 2

## Equations
# Forward model (acoustic wave equation):
#   (1/c(x,z)^2) * d^2p/dt^2 = nabla^2 p + s(t) * delta(x - x_s)
#   c(x,z) = P-wave velocity field (unknown, to be inverted)
#   p(x, z, t) = pressure wavefield
#   s(t) = source wavelet
#
# Inverse problem (FWI):
#   min_{c} J(c) = 0.5 * sum_s sum_r ||p_obs(x_r, t; x_s) - p_syn(x_r, t; x_s; c)||_2^2
#                  + lambda * TV(c)
#
# Gradient via adjoint-state method:
#   dJ/dc = -2/c^3 * integral_0^T p_forward * p_adjoint dt
#
# Multi-scale frequency continuation: solve low-freq first, use as starting model for higher freq

equations: |
  forward: (1/c^2) * d^2p/dt^2 = laplacian(p) + s(t)*delta(x-x_s)
  objective: min_c 0.5 * sum ||p_obs - p_syn(c)||^2 + lambda * TV(c)
  adjoint: gradient via adjoint-state method
  continuation: frequency bands 2-5, 5-10, 10-15 Hz

parameters:
  model: Marmousi-2       # Martin et al., 2006
  n_sources: 240          # surface sources at 10m spacing
  n_receivers: 480        # surface receivers
  source_wavelet: Ricker  # peak frequency 10 Hz
  freq_bands: [[2,5], [5,10], [10,15]]  # Hz
  starting_model: 1D_gradient_smoothed  # 500m Gaussian smoothing of true model
  lambda_TV: 1.0e-3       # TV regularization
  dx: 10                  # m (spatial grid spacing)
  dt: 0.001               # s (time step)
  T_record: 4.0           # s (recording time)
  iterations_per_band: [50, 30, 20]  # L-BFGS iterations per frequency band
  grid_size: [1700, 350]  # nx, nz

## Boundary Conditions
boundary: |
  free_surface: z = 0 (pressure-free, p = 0)
  absorbing: PML (perfectly matched layer) on left, right, bottom
  PML_thickness: 20 grid points
  source_depth: z = 10 m (near-surface sources)

## Initial Conditions
initial: |
  starting_velocity: c_0(x,z) = smooth 1D gradient
    c_0(z) = 1500 + (4500-1500) * z/z_max, smoothed with 500m Gaussian
  no prior wavefield: p(x,z,0) = 0, dp/dt(x,z,0) = 0

## Observables
# 1. Inverted velocity model c_inv(x, z) on 1700 x 350 grid
# 2. Velocity PSNR: 10*log10(max(c_true)^2 / MSE(c_inv, c_true))
# 3. Data fit: ||p_obs - p_syn(c_inv)||^2 / ||p_obs||^2
# 4. Structural similarity: SSIM(c_inv, c_true)
# 5. Depth profiles: c(z) at x = 5000, 8500, 12000 m
# Validated against SEG/EAGE Marmousi community benchmark
observables:
  - velocity_model: c_inv(x, z) on 1700x350 grid  # m/s
  - velocity_PSNR: dB
  - data_misfit: relative L2 norm
  - depth_profiles: c(z) at 3 x-locations  # m/s
  - SSIM: dimensionless

## Tolerance
tolerance:
  velocity_PSNR_minimum: 25.0  # dB
  data_misfit_relative: 0.05   # 5% residual
  metric: velocity_PSNR

## Primitives Required
primitives: [integrate, solve_linear, optimize, transform, constrain, discretize]
# int (wave equation time integration), L (Helmholtz solver for frequency-domain FWI),
# O (L-BFGS optimization), F (FFT for frequency filtering), B (PML, free surface),
# G (finite-difference grid)

## Task Instance Variations
# Vary: model (Marmousi, Overthrust, BP, synthetic layered), frequency range,
#        source/receiver geometry (marine, land, OBS), noise level,
#        starting model quality, regularization type
# Public: Marmousi-2 with reference velocity, full-aperture acquisition
# Dev: limited-offset, noisy data, poor starting model
# Hidden: elastic (P+S waves), anisotropic (VTI), time-lapse (4D)
