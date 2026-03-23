# Specification: Rectangular Waveguide Mode Analysis

## Domain
domain: rectangular_cross_section
geometry: [0, a] x [0, b]
a: 0.02286  # meters (WR-90, standard X-band)
b: 0.01016  # meters
dimension: 2 (cross-section); propagation along z

## Equations
# Helmholtz equation for TE modes (H_z component):
#   nabla_t^2 H_z + k_c^2 H_z = 0
#   where k_c^2 = omega^2 * mu * epsilon - beta^2
#
# For TM modes (E_z component):
#   nabla_t^2 E_z + k_c^2 E_z = 0
#
# Cutoff frequencies:
#   f_c(m,n) = (1 / 2*pi*sqrt(mu*epsilon)) * sqrt((m*pi/a)^2 + (n*pi/b)^2)
#
# Analytical solution (for validation):
#   TE_mn: H_z = H_0 * cos(m*pi*x/a) * cos(n*pi*y/b)
#   TM_mn: E_z = E_0 * sin(m*pi*x/a) * sin(n*pi*y/b)

equations: |
  nabla_t^2 psi + k_c^2 psi = 0
  k_c^2 = (m*pi/a)^2 + (n*pi/b)^2
  f_c = c / (2*pi) * k_c

parameters:
  mu: 1.2566370614e-6    # H/m (free space)
  epsilon: 8.854187817e-12  # F/m (free space)
  c: 299792458            # m/s
  frequency_range: [8.0e9, 12.0e9]  # Hz (X-band)

## Boundary Conditions
# PEC (perfect electric conductor) walls:
#   TE: dH_z/dn = 0 on walls (Neumann)
#   TM: E_z = 0 on walls (Dirichlet)
boundary: |
  TE_modes: Neumann (dH_z/dn = 0) on all four walls
  TM_modes: Dirichlet (E_z = 0) on all four walls

## Initial Conditions
initial: N/A (eigenvalue problem, no time dependence)

## Observables
# 1. Cutoff frequencies f_c(m,n) for first 10 modes
# 2. Mode field patterns H_z(x,y) or E_z(x,y) on 64x64 grid
# 3. Propagation constants beta(f) for f in [8, 12] GHz
# 4. Attenuation constant alpha(f) for lossy walls (sigma_w = 5.8e7 S/m, copper)
observables:
  - cutoff_frequencies: f_c for (m,n) in first 10 modes  # Hz
  - mode_field_patterns: psi(x,y) on 64x64 grid
  - propagation_constant: beta(f) for 100 frequency points  # rad/m
  - attenuation: alpha(f) for lossy-wall extension  # Np/m

## Tolerance
tolerance:
  cutoff_freq_relative: 1.0e-8
  field_pattern_L2: 1.0e-8
  metric: L2_relative_norm

## Primitives Required
primitives: [differentiate, solve_linear, transform, constrain, discretize]

## Task Instance Variations
# Vary: waveguide dimensions (WR-90 to WR-430), fill material (air, Teflon, alumina),
#        wall conductivity (PEC to lossy Cu/Al), frequency range, mode count
# Public: standard WR-90, analytical ground truth
# Dev: non-standard dimensions, dielectric-filled
# Hidden: partially filled waveguide (requires mode matching), lossy walls
