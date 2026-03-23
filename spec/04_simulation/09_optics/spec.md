# Specification: Fresnel Diffraction (Sanity Check)

## Domain
domain: observation_plane
geometry: aperture plane (x_a, y_a) -> observation plane (x, y) at distance z
aperture: circular, radius R = 0.5 mm
z: 0.1  # m (propagation distance)
dimension: 2

## Equations
# Fresnel diffraction integral:
#   U(x, y, z) = (exp(ikz) / (i*lambda*z)) *
#     integral integral U_0(x_a, y_a) *
#     exp(i*pi/(lambda*z) * ((x-x_a)^2 + (y-y_a)^2)) dx_a dy_a
#
# For circular aperture, reduces to Lommel functions (analytical)
# Fresnel number: N_F = R^2 / (lambda * z)

equations: |
  U(x,y,z) = (exp(ikz)/(i*lambda*z)) * FT{U_0 * exp(i*pi*(x_a^2+y_a^2)/(lambda*z))}
  I(x,y) = |U(x,y,z)|^2
  Fresnel_number: N_F = R^2 / (lambda * z)

parameters:
  lambda: 632.8e-9        # m (HeNe laser wavelength)
  R_aperture: 0.5e-3      # m (aperture radius)
  z: 0.1                  # m (propagation distance)
  k: 9.926e6              # 2*pi/lambda, rad/m
  N_F: 3.95               # Fresnel number
  grid_points: 512        # per axis

## Boundary Conditions
boundary: |
  aperture_function: U_0(x_a, y_a) = 1 inside circle, 0 outside
  radiation_condition: outgoing waves at infinity

## Initial Conditions
initial: |
  U_0 = plane wave (uniform amplitude and phase) incident on aperture

## Observables
# 1. Intensity pattern I(x, y) on 512x512 observation grid
# 2. On-axis intensity I(0, 0, z) vs z for z in [0.01, 1.0] m
# 3. Radial intensity profile I(r) at z = 0.1 m
# 4. Encircled energy: E(r) = integral_0^r I(r') 2*pi*r' dr'
observables:
  - intensity_pattern: I(x, y) on 512x512 grid  # W/m^2 (normalized)
  - on_axis_intensity: I(0, 0, z) for 100 z values  # W/m^2
  - radial_profile: I(r) for 256 radial points  # W/m^2
  - encircled_energy: E(r) for 256 radial points

## Tolerance
tolerance:
  intensity_L2_relative: 1.0e-5
  metric: L2_relative_norm

## Primitives Required
primitives: [differentiate, transform, solve_linear, constrain, discretize]
# partial (phase gradient), F (FFT for Fresnel propagation), L (if using angular spectrum),
# B (aperture boundary), G (observation grid)

## Task Instance Variations
# Vary: aperture shape (circle, rectangle, slit, annular), wavelength (400-800 nm),
#        propagation distance (near-field to far-field), coherence (partial),
#        aberrations (defocus, astigmatism)
# Public: circular aperture with Lommel function analytical reference
# Dev: rectangular aperture, Gaussian beam illumination
# Hidden: partially coherent source, phase-only aperture, near-field (N_F >> 1)
