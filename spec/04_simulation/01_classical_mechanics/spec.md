# Specification: Granular Flow with Frictional Contact (2D)

> Source: `papers/universal_simulation/benchmark/01_classical_mechanics/spec.md`

## Equations

```
# Discrete Element Method (DEM) with Hertz-Mindlin contact model
# For each particle i (i = 1, ..., N):
#   m_i * d^2 x_i / dt^2 = sum_j F_ij^n + sum_j F_ij^t + m_i * g
#
# Normal contact force (Hertz):
#   F_ij^n = k_n * delta_ij^(3/2) * n_ij - gamma_n * v_ij^n
#   delta_ij = max(0, R_i + R_j - |x_i - x_j|)
#
# Tangential contact force (Mindlin with Coulomb friction):
#   F_ij^t = min(|k_t * s_ij - gamma_t * v_ij^t|, mu * |F_ij^n|) * t_ij
#
# Parameters:
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Bulk stress tensor: sigma_ij = (1/V) * sum_contacts (f_c x l_c)
# 2. Volume fraction profile: phi(y) averaged in horizontal strips
# 3. Coordination number: Z = 2*N_contacts / N_particles
# 4. Shear band location (if applicable): y_shear from strain localization
observables:
  - bulk_stress_tensor: sigma_xx, sigma_yy, sigma_xy  # Pa

**Tolerance**: # Relative error in bulk stress tensor components: <= 1e-3 # Relative error in volume fraction profile: <= 1e-2

## Variations

Variations (100 instances x 3 tiers)
# Vary: N (3000-8000), mu (0.2-0.8), applied stress (1-50 kPa),
#        polydispersity (monodisperse to R_std=1mm), gravity angle (0-30 deg)
# Public tier: standard parameters, ground truth provided
# Dev tier: modified friction and particle count, blind evaluation

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/01_classical_mechanics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/01_classical_mechanics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/01_classical_mechanics/spec.md`
