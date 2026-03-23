# Specification: Turbulent Backward-Facing Step (Re = 5100)

## Domain
domain: 2D channel with backward-facing step
geometry: |
  Inlet channel: x in [-5h, 0], y in [0, h]  (h = step height)
  Expansion region: x in [0, 30h], y in [-h, h]
  Step at x=0: sudden expansion from h to 2h
  Step height: h = 0.0127 m
dimension: 2 (spanwise-averaged statistics from 3D DNS)

## Equations
# Incompressible Navier-Stokes:
#   du/dt + (u . nabla) u = -1/rho * nabla p + nu * nabla^2 u
#   nabla . u = 0
#
# Reynolds number: Re = U_0 * h / nu = 5100
#   U_0 = centerline velocity at inlet
#   nu = kinematic viscosity

equations: |
  du/dt + (u . grad) u = -grad(p)/rho + nu * laplacian(u)
  div(u) = 0

parameters:
  Re: 5100
  h: 0.0127               # m (step height)
  U_0: 1.0                # m/s (reference velocity, normalized)
  nu: 1.961e-4            # m^2/s (for Re=5100 with U_0=1, h=0.0127)
  rho: 1.0                # kg/m^3 (normalized)
  expansion_ratio: 2.0    # channel height doubles at step
  T_simulation: 500       # non-dimensional time units (h/U_0)
  T_averaging: 300        # time units for statistics (after transient)

## Boundary Conditions
boundary: |
  inlet (x = -5h): fully developed turbulent channel profile
    U(y) = U_center * (1 - (2y/h - 1)^6)  (approximate)
    with superimposed turbulent fluctuations (synthetic turbulence generator)
  outlet (x = 30h): convective outflow dU/dt + U_c * dU/dx = 0
  top wall (y = h for x<0; y = h for x>0): no-slip, u = 0
  bottom wall (y = 0 for x<0; y = -h for x>0): no-slip, u = 0
  step face (x = 0, y in [-h, 0]): no-slip, u = 0

## Initial Conditions
initial: |
  u(x, y, t=0) = inlet profile extended uniformly + random perturbations
  p(x, y, t=0) = 0
  transient: discard first 200 time units before collecting statistics

## Observables
# Validated against JHU Turbulence Database DNS (Li et al., 2008)
# 1. Reattachment length: x_r / h (expected: ~6.1 for Re=5100)
# 2. Mean streamwise velocity: <U>(x, y) at x/h = 4, 6, 8, 10
# 3. Turbulent kinetic energy: TKE(x, y) = 0.5 * (<u'^2> + <v'^2>)
# 4. Reynolds shear stress: <u'v'>(x, y)
# 5. Skin friction coefficient: C_f(x) along bottom wall
observables:
  - reattachment_length: x_r / h  # dimensionless
  - mean_velocity_profiles: <U>(y) at x/h = [4, 6, 8, 10]  # 50 points each
  - turbulent_kinetic_energy: TKE(y) at x/h = [4, 6, 8, 10]  # normalized by U_0^2
  - reynolds_stress: <u_prime_v_prime>(y) at x/h = [4, 6, 8, 10]
  - skin_friction: C_f(x) for x/h in [0, 20]  # 100 points

## Tolerance
# Reattachment length: relative error <= 5%
# Mean velocity profiles: L2 relative error <= 5%
# TKE profiles: L2 relative error <= 10% (statistics are noisy)
tolerance:
  reattachment_relative: 0.05
  velocity_L2_relative: 0.05
  TKE_L2_relative: 0.10
  metric: L2_relative_norm

## Primitives Required
primitives: [differentiate, evolve, evaluate_nonlinear, couple, constrain, discretize, project]
# partial (spatial derivatives), E (time-stepping), N (convective nonlinearity),
# K (pressure-velocity coupling), B (no-slip walls, inlet/outlet),
# G (mesh with refinement near step), Pi (statistical averaging / POD)

## Task Instance Variations
# Vary: Re (1000-10000), expansion ratio (1.5-3.0), step geometry (BFS, double expansion,
#        ramp), inlet turbulence intensity, grid resolution
# Public: Re=5100 BFS with DNS reference data
# Dev: different Re, modified geometry
# Hidden: 3D effects, heated step, pulsating inlet
