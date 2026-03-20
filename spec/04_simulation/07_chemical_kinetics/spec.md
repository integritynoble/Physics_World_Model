# Specification: GRI-Mech 3.0 Methane/Air Ignition Delay

> Source: `papers/universal_simulation/benchmark/07_chemical_kinetics/spec.md`

## Equations

```
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
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Ignition delay time tau_ign (defined as time of max dT/dt)
# 2. Temperature history T(t)
# 3. Peak temperature T_max
# 4. Major species histories: CH4(t), O2(t), CO(t), CO2(t), H2O(t), OH(t)
# 5. Heat release rate: dQ/dt(t)
# Validated against shock-tube ignition delay experiments (Smith et al., 1999)

**Tolerance**: # Ignition delay: relative error <= 10% compared to experimental shock-tube data # Temperature history: L2 relative error <= 5%

## Variations

Variations
# Vary: T_initial (800-2000 K), p_initial (1-50 atm), equivalence_ratio (0.5-2.0),
#        fuel (CH4, C2H6, C3H8), reactor type (CV, CP), dilution with Ar/He
# Public: standard conditions with experimental reference data
# Dev: lean/rich mixtures, elevated pressures

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/07_chemical_kinetics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/07_chemical_kinetics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/07_chemical_kinetics/spec.md`
