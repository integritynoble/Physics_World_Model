# Specification: Helium Atom Ground State Energy (Configuration Interaction)

## Domain
domain: R^3 x R^3 (two-electron configuration space)
# Effective radial domain: r in [0, R_max], R_max = 50 a_0 (Bohr radii)
# Angular: full sphere (l_max expansion)
dimension: 6 (3 per electron)

## Equations
# Time-independent Schrodinger equation:
#   H psi(r1, r2) = E psi(r1, r2)
#
# Hamiltonian (atomic units, Z = 2):
#   H = -1/2 nabla_1^2 - 1/2 nabla_2^2 - Z/r1 - Z/r2 + 1/|r1 - r2|
#
# CI expansion:
#   psi = sum_{ij} c_{ij} phi_i(r1) phi_j(r2)  (antisymmetrized)
#   phi_i = R_{nl}(r) Y_lm(theta, phi)  (hydrogen-like orbitals or STO/GTO basis)
#
# Matrix eigenvalue problem:
#   H_CI c = E c
#   H_CI_{ij,kl} = <phi_i phi_j | H | phi_k phi_l>  (antisymmetrized)

equations: |
  H = T_1 + T_2 + V_ne(r1) + V_ne(r2) + V_ee(r1, r2)
  T_i = -1/2 * nabla_i^2
  V_ne(r) = -Z / r
  V_ee(r1, r2) = 1 / |r1 - r2|
  H_CI * c = E * c

parameters:
  Z: 2                    # nuclear charge (helium)
  basis: STO-6G           # Slater-type orbitals fitted by 6 Gaussians
  n_max: 4                # principal quantum number cutoff
  l_max: 3                # angular momentum cutoff (s, p, d, f)
  R_max: 50               # a_0 (Bohr radii)

## Boundary Conditions
boundary: |
  psi(r -> infinity) = 0  (bound state)
  psi antisymmetric under electron exchange (Pauli exclusion)
  normalization: <psi|psi> = 1

## Initial Conditions
initial: N/A (eigenvalue problem)

## Observables
# 1. Ground state energy E_0 (should be -2.9037 Ha; exact: -2.903724 Ha)
# 2. Ionization energy IE = E(He+) - E(He) (experimental: 24.587 eV)
# 3. Electron density rho(r) = integral |psi|^2 d(other electron)
# 4. <r1> expectation value (mean electron-nucleus distance)
# 5. Correlation energy: E_corr = E_CI - E_HF
observables:
  - ground_state_energy: E_0  # Hartree
  - ionization_energy: IE  # eV
  - electron_density: rho(r) on radial grid (100 points)  # a_0^-3
  - mean_radius: <r>  # a_0
  - correlation_energy: E_corr  # mHa

## Tolerance
# Absolute error in ground state energy: <= 1 mHa (milli-Hartree)
# This requires capturing electron correlation beyond Hartree-Fock
tolerance:
  energy_absolute: 1.0e-3  # Hartree (1 mHa)
  density_L2_relative: 1.0e-2
  metric: absolute_energy_error

## Primitives Required
primitives: [differentiate, solve_linear, project, constrain, discretize, optimize]
# partial (kinetic energy operator), L (CI matrix eigenvalue), Pi (basis truncation),
# B (antisymmetry, normalization), G (radial/angular grid), O (SCF optimization for HF starting point)

## Task Instance Variations
# Vary: atom (He, Li+, Be2+), basis set (STO-3G to cc-pVQZ), n_max (2-6),
#        l_max (1-5), method (HF, CISD, Full-CI)
# Public: He with STO-6G, analytical/NIST ground truth
# Dev: Li+ and Be2+ with larger basis sets
# Hidden: near-degenerate states, excited state requests, basis set extrapolation
