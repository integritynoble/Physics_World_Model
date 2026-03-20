# Specification: COVID-19 County-Level Epidemic Dynamics (SEIR-D)

> Source: `papers/universal_simulation/benchmark/08_epidemiology/spec.md`

## Equations

```
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
```

## Parameters

*See source spec.*

## Observables & Tolerance

# Validated against Johns Hopkins CSSE reported data (Dong et al., 2020)
# 1. Daily new cases: sigma * E(t) (compared to reported cases)
# 2. Peak infection timing: argmax_t I(t)
# 3. Cumulative cases at T_final: integral(sigma * E) dt
# 4. Cumulative deaths at T_final: D(T_final)
# 5. Effective reproduction number: R_eff(t) = beta(t) * S(t) / (gamma + delta) / N

**Tolerance**: # Peak timing: relative error <= 15% # Cumulative cases: relative error <= 20%

## Variations

Variations
# Vary: county (8 base + 42 additional US counties), model structure
#        (SIR, SEIR, SEIR-D, SEIR-D with hospitalization), time window,
#        data source (JHU, NYT, USAFacts), intervention modeling
# Public: 8 counties with JHU data, parameter fits provided

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/08_epidemiology/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/08_epidemiology/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/08_epidemiology/spec.md`
