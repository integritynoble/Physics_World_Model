# Specification: GRI-Mech 3.0 Methane/Air Ignition Delay

## Domain
domain: 0D homogeneous reactor (constant volume or constant pressure)
dimension: 0 (spatially homogeneous; ODE system in species concentrations)

## Equations
# Species conservation (0D reactor):
#   dY_k/dt = W_k * omega_dot_k / rho    for k = 1, ..., K
#
# Energy equation (adiabatic, constant volume):
#   dT/dt = -1/(rho * c_v) * sum_k (h_k * W_k * omega_dot_k)
#
# GRI-Mech 3.0: 53 species, 325 reactions (reversible)
# Reaction rates via Arrhenius: k = A * T^b * exp(-E_a / (R * T))
#
# This is a stiff ODE system with stiffness ratio ~1e10

equations: |
  dY_k/dt = W_k * omega_dot_k(T, Y, p) / rho
  dT/dt = -sum_k(h_k * W_k * omega_dot_k) / (rho * c_v)
  omega_dot_k = sum_j(nu_kj * q_j)  (net production rate)
  q_j = k_fj * prod(C_k^nu_kj_forward) - k_rj * prod(C_k^nu_kj_reverse)
  k_j = A_j * T^(b_j) * exp(-E_aj / (R * T))

parameters:
  mechanism: GRI-Mech 3.0  # 53 species, 325 reactions
  fuel: CH4                # methane
  oxidizer: air            # 21% O2, 79% N2
  equivalence_ratio: 1.0   # stoichiometric
  T_initial: 1500          # K
  p_initial: 10            # atm
  reactor_type: constant_volume

## Boundary Conditions
# 0D reactor: no spatial boundaries
# Adiabatic walls (no heat loss)
# Constant volume (or constant pressure, varies per instance)
boundary: |
  adiabatic: dQ/dt = 0
  closed_system: dm/dt = 0
  constant_volume: dV/dt = 0  (for CV reactor)

## Initial Conditions
initial: |
  T(0) = T_initial  # K
  p(0) = p_initial   # atm
  Y_CH4(0) = fuel mass fraction (from equivalence ratio)
  Y_O2(0) = oxidizer mass fraction
  Y_N2(0) = nitrogen mass fraction
  Y_k(0) = 0 for all other species

## Observables
# 1. Ignition delay time tau_ign (defined as time of max dT/dt)
# 2. Temperature history T(t)
# 3. Peak temperature T_max
# 4. Major species histories: CH4(t), O2(t), CO(t), CO2(t), H2O(t), OH(t)
# 5. Heat release rate: dQ/dt(t)
# Validated against shock-tube ignition delay experiments (Smith et al., 1999)
observables:
  - ignition_delay: tau_ign  # seconds
  - temperature_history: T(t) for 1000 time points  # K
  - peak_temperature: T_max  # K
  - species_histories: Y_k(t) for CH4, O2, CO, CO2, H2O, OH  # mass fractions
  - heat_release_rate: dQ/dt(t)  # W/m^3

## Tolerance
# Ignition delay: relative error <= 10% compared to experimental shock-tube data
# Temperature history: L2 relative error <= 5%
tolerance:
  tau_ign_relative: 0.10
  T_history_L2_relative: 0.05
  metric: ignition_delay_relative_error

## Primitives Required
primitives: [differentiate, evolve, evaluate_nonlinear, couple, constrain]
# partial (temperature/species gradients for Jacobian), E (BDF implicit time-stepping),
# N (Arrhenius rate evaluation), K (species-temperature coupling), B (conservation)

## Task Instance Variations
# Vary: T_initial (800-2000 K), p_initial (1-50 atm), equivalence_ratio (0.5-2.0),
#        fuel (CH4, C2H6, C3H8), reactor type (CV, CP), dilution with Ar/He
# Public: standard conditions with experimental reference data
# Dev: lean/rich mixtures, elevated pressures
# Hidden: NTC regime (800-1000K), multi-stage ignition, extinction limits
