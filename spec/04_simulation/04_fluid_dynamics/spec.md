# Specification: Turbulent Backward-Facing Step (Re = 5100)

> Source: `papers/universal_simulation/benchmark/04_fluid_dynamics/spec.md`

## Equations

```
# Incompressible Navier-Stokes:
#   du/dt + (u . nabla) u = -1/rho * nabla p + nu * nabla^2 u
#   nabla . u = 0
#
# Reynolds number: Re = U_0 * h / nu = 5100
#   U_0 = centerline velocity at inlet
#   nu = kinematic viscosity

equations: |
  du/dt + (u . grad) u = -grad(p)/rho + nu * laplacian(u)
  div(u) = 0

```

## Parameters

*See source spec.*

## Observables & Tolerance

# Validated against JHU Turbulence Database DNS (Li et al., 2008)
# 1. Reattachment length: x_r / h (expected: ~6.1 for Re=5100)
# 2. Mean streamwise velocity: <U>(x, y) at x/h = 4, 6, 8, 10
# 3. Turbulent kinetic energy: TKE(x, y) = 0.5 * (<u'^2> + <v'^2>)
# 4. Reynolds shear stress: <u'v'>(x, y)
# 5. Skin friction coefficient: C_f(x) along bottom wall

**Tolerance**: # Reattachment length: relative error <= 5% # Mean velocity profiles: L2 relative error <= 5%

## Variations

Variations
# Vary: Re (1000-10000), expansion ratio (1.5-3.0), step geometry (BFS, double expansion,
#        ramp), inlet turbulence intensity, grid resolution
# Public: Re=5100 BFS with DNS reference data
# Dev: different Re, modified geometry

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/04_fluid_dynamics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/04_fluid_dynamics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/04_fluid_dynamics/spec.md`
