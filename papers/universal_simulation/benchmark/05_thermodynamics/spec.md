# Specification: 2D Heat Conduction (Sanity Check)

## Domain
domain: unit_square
geometry: [0, 1] x [0, 1]
dimension: 2

## Equations
# Heat equation:
#   du/dt = alpha * (d^2u/dx^2 + d^2u/dy^2)
#
# Steady-state Poisson variant:
#   -alpha * nabla^2 u = f(x, y)

equations: |
  du/dt = alpha * laplacian(u)
  steady_state: -alpha * laplacian(u) = f(x,y)

parameters:
  alpha: 0.01             # m^2/s (thermal diffusivity)
  T_final: 1.0            # s (for transient)
  source_term: f(x,y) = 0  # homogeneous (varies per instance)

## Boundary Conditions
boundary: |
  u(0, y, t) = 0          # left wall
  u(1, y, t) = 0          # right wall
  u(x, 0, t) = 0          # bottom wall
  u(x, 1, t) = sin(pi * x)  # top wall (Dirichlet)

## Initial Conditions
initial: |
  u(x, y, 0) = 0          # zero everywhere

## Observables
# 1. Temperature field u(x, y, T_final) on 64x64 grid
# 2. Steady-state solution (for Poisson variant)
# 3. Heat flux at boundaries: q = -alpha * du/dn
# 4. Total thermal energy: E(t) = integral u(x,y,t) dA
observables:
  - temperature_field: u(x, y, T) on 64x64 grid
  - boundary_heat_flux: q(x) at y=1  # 64 points
  - total_energy: E(t) for 100 time snapshots

## Tolerance
tolerance:
  field_L2_relative: 1.0e-4
  metric: L2_relative_norm

## Primitives Required
primitives: [differentiate, evolve, solve_linear, constrain, discretize]

## Task Instance Variations
# Vary: alpha (1e-4 to 1), boundary conditions (all Dirichlet, mixed,
#        time-dependent), source terms (Gaussian, sinusoidal), domain shape
# Public: unit square, analytical solution available
# Dev: L-shaped domain, anisotropic diffusivity
# Hidden: discontinuous diffusivity, nonlinear radiation BC
