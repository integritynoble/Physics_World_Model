# Specification: COVID-19 County-Level Epidemic Dynamics (SEIR-D)

## Domain
domain: temporal ODE per county (no spatial structure)
geometry: t in [0, T_final] days
T_final: 180  # days (March - August 2020)
dimension: 0 (compartmental ODE; 8 counties modeled independently)

## Equations
# SEIR-D compartmental model:
#   dS/dt = -beta(t) * S * I / N
#   dE/dt = beta(t) * S * I / N - sigma * E
#   dI/dt = sigma * E - gamma * I - delta * I
#   dR/dt = gamma * I
#   dD/dt = delta * I
#
# Time-varying transmission rate:
#   beta(t) = beta_0 * (1 - sum_k alpha_k * sigmoid((t - t_k) / tau_k))
#   accounts for NPI (non-pharmaceutical intervention) effects
#
# N = S + E + I + R + D (total population, constant per county)

equations: |
  dS/dt = -beta(t) * S * I / N
  dE/dt = beta(t) * S * I / N - sigma * E
  dI/dt = sigma * E - gamma * I - delta * I
  dR/dt = gamma * I
  dD/dt = delta * I
  beta(t) = beta_0 * product_k(1 - alpha_k * sigmoid((t - t_k) / tau_k))

parameters:
  # County-specific (fitted via MLE on first 60% of data)
  counties:
    - name: Fulton County, GA
      N: 1063564
      beta_0: 0.35
      sigma: 0.2        # 1/incubation_period (5 days)
      gamma: 0.1         # 1/infectious_period (10 days)
      delta: 0.005       # infection fatality rate proxy
      I_0: 15            # initial infected
    - name: Cook County, IL
      N: 5150233
      beta_0: 0.38
    - name: Los Angeles County, CA
      N: 10039107
      beta_0: 0.32
    - name: Harris County, TX
      N: 4713325
      beta_0: 0.34
    - name: Maricopa County, AZ
      N: 4485414
      beta_0: 0.33
    - name: King County, WA
      N: 2252782
      beta_0: 0.30
    - name: Miami-Dade County, FL
      N: 2716940
      beta_0: 0.36
    - name: Wayne County, MI
      N: 1749343
      beta_0: 0.40
  # Shared parameters (initial guesses, refined per county)
  sigma_default: 0.2     # 1/day (5-day incubation)
  gamma_default: 0.1     # 1/day (10-day infectious period)
  delta_default: 0.005   # 1/day

## Boundary Conditions
# ODE system: no spatial boundaries
# Conservation: N = S + E + I + R + D (constant)
boundary: |
  conservation: S + E + I + R + D = N (enforced at each step)
  non_negativity: S, E, I, R, D >= 0

## Initial Conditions
# Per county from Johns Hopkins CSSE data (first reported case date)
initial: |
  S(0) = N - E_0 - I_0
  E(0) = 3 * I_0  (assume 3x latent for each observed case)
  I(0) = I_0  (from first week case data)
  R(0) = 0
  D(0) = 0

## Observables
# Validated against Johns Hopkins CSSE reported data (Dong et al., 2020)
# 1. Daily new cases: sigma * E(t) (compared to reported cases)
# 2. Peak infection timing: argmax_t I(t)
# 3. Cumulative cases at T_final: integral(sigma * E) dt
# 4. Cumulative deaths at T_final: D(T_final)
# 5. Effective reproduction number: R_eff(t) = beta(t) * S(t) / (gamma + delta) / N
# Data split: first 60% training, last 40% validation
observables:
  - daily_new_cases: dC/dt(t) for 180 days  # cases/day
  - peak_timing: t_peak  # day index
  - cumulative_cases: C(T_final)
  - cumulative_deaths: D(T_final)
  - R_effective: R_eff(t) for 180 days

## Tolerance
# Peak timing: relative error <= 15%
# Cumulative cases: relative error <= 20%
# These tolerances reflect inherent limitations of compartmental models
# without detailed intervention timing data
tolerance:
  peak_timing_relative: 0.15
  cumulative_cases_relative: 0.20
  metric: peak_timing_relative_error

## Primitives Required
primitives: [evolve, evaluate_nonlinear, couple, sample, constrain]
# E (ODE integration, RK4 or adaptive), N (nonlinear transmission term),
# K (coupling between compartments), S (parameter uncertainty quantification),
# B (conservation, non-negativity)

## Task Instance Variations
# Vary: county (8 base + 42 additional US counties), model structure
#        (SIR, SEIR, SEIR-D, SEIR-D with hospitalization), time window,
#        data source (JHU, NYT, USAFacts), intervention modeling
# Public: 8 counties with JHU data, parameter fits provided
# Dev: 20 additional counties, blind parameter estimation
# Hidden: out-of-sample prediction (fit on wave 1, predict wave 2),
#          model selection (which compartmental structure is best?)
