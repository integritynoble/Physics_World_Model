# Specification: 2D Heat Conduction (Sanity Check)

> Source: `papers/universal_simulation/benchmark/05_thermodynamics/spec.md`

## Equations

```
# Heat equation:
#   du/dt = alpha * (d^2u/dx^2 + d^2u/dy^2)
#
# Steady-state Poisson variant:
#   -alpha * nabla^2 u = f(x, y)

equations: |
  du/dt = alpha * laplacian(u)
  steady_state: -alpha * laplacian(u) = f(x,y)

parameters:
  alpha: 0.01             # m^2/s (thermal diffusivity)
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Temperature field u(x, y, T_final) on 64x64 grid
# 2. Steady-state solution (for Poisson variant)
# 3. Heat flux at boundaries: q = -alpha * du/dn
# 4. Total thermal energy: E(t) = integral u(x,y,t) dA
observables:
  - temperature_field: u(x, y, T) on 64x64 grid

**Tolerance**: tolerance:   field_L2_relative: 1.0e-4

## Variations

Variations
# Vary: alpha (1e-4 to 1), boundary conditions (all Dirichlet, mixed,
#        time-dependent), source terms (Gaussian, sinusoidal), domain shape
# Public: unit square, analytical solution available
# Dev: L-shaped domain, anisotropic diffusivity

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/05_thermodynamics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/05_thermodynamics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/05_thermodynamics/spec.md`
