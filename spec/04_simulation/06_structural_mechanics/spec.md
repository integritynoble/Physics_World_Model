# Specification: Topology Optimization of Aerospace Bracket

## Domain
domain: design_region
geometry: rectangular design domain [0, L] x [0, H] with load and support regions
L: 0.2       # m (length)
H: 0.1       # m (height)
dimension: 2

## Equations
# Linear elasticity (state equation):
#   -div(sigma) = f   in Omega
#   sigma = C(rho) : epsilon(u)
#   epsilon(u) = 0.5 * (grad u + grad u^T)
#
# SIMP (Solid Isotropic Material with Penalization):
#   C(rho) = rho^p * C_0
#   p = 3 (penalization exponent)
#
# Topology optimization:
#   min_{rho} compliance = integral f . u dOmega
#   subject to: volume constraint integral rho dOmega <= V_max
#               0 < rho_min <= rho(x) <= 1
#               K(rho) u = f  (equilibrium)

equations: |
  state: -div(C(rho) : epsilon(u)) = f
  SIMP: C(rho) = rho^p * C_0
  objective: min compliance = f^T u
  volume: integral(rho) <= V_frac * |Omega|

parameters:
  E_0: 210e9              # Pa (Young's modulus, steel)
  nu_poisson: 0.3         # Poisson's ratio
  rho_min: 1.0e-3         # minimum density (avoid singularity)
  p_simp: 3               # SIMP penalization exponent
  V_frac: 0.4             # volume fraction constraint (40%)
  filter_radius: 3.0      # mesh elements (density filter for manufacturability)
  max_iterations: 200     # optimization iterations

## Boundary Conditions
boundary: |
  fixed_support: left edge (x = 0), u = 0 (clamped)
  applied_load: point load F = [0, -1000] N at (L, H/2)  (mid-right edge)
  traction_free: all other boundaries

## Initial Conditions
initial: |
  rho(x, y) = V_frac  (uniform initial density)

## Observables
# 1. Optimal density field rho*(x, y) on design mesh
# 2. Final compliance value C*
# 3. Displacement field u*(x, y) under optimal topology
# 4. Von Mises stress field sigma_vm(x, y)
# 5. Volume fraction achieved (should be V_frac within tolerance)
observables:
  - optimal_density: rho_star(x, y) on 200x100 mesh
  - compliance: C_star  # N.m
  - displacement_field: u(x, y) on 200x100 mesh  # m
  - von_mises_stress: sigma_vm(x, y)  # Pa
  - volume_fraction: V_achieved / V_total

## Tolerance
# Compliance within 0.1% of reference (ANSYS/TopOpt benchmark)
# Volume constraint satisfied to 0.1%
tolerance:
  compliance_relative: 1.0e-3
  volume_constraint: 1.0e-3
  metric: relative_compliance_error

## Primitives Required
primitives: [differentiate, solve_linear, optimize, constrain, discretize]
# partial (strain-displacement), L (FEM equilibrium), O (MMA/OC optimizer),
# B (supports, loads, volume constraint), G (structured mesh)

## Task Instance Variations
# Vary: load position/direction, support configuration (cantilever, bridge, MBB beam),
#        volume fraction (0.2-0.6), multiple load cases, 3D extension
# Public: standard MBB beam and cantilever with TopOpt-88 reference solutions
# Dev: multi-load, asymmetric supports
# Hidden: stress-constrained, buckling, 3D bracket
