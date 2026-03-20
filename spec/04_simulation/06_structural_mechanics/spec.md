# Specification: Topology Optimization of Aerospace Bracket

> Source: `papers/universal_simulation/benchmark/06_structural_mechanics/spec.md`

## Equations

```
# Linear elasticity (state equation):
#   -div(sigma) = f   in Omega
#   sigma = C(rho) : epsilon(u)
#   epsilon(u) = 0.5 * (grad u + grad u^T)
#
# SIMP (Solid Isotropic Material with Penalization):
#   C(rho) = rho^p * C_0
#   p = 3 (penalization exponent)
#
# Topology optimization:
#   min_{rho} compliance = integral f . u dOmega
#   subject to: volume constraint integral rho dOmega <= V_max
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Optimal density field rho*(x, y) on design mesh
# 2. Final compliance value C*
# 3. Displacement field u*(x, y) under optimal topology
# 4. Von Mises stress field sigma_vm(x, y)
# 5. Volume fraction achieved (should be V_frac within tolerance)
observables:

**Tolerance**: # Compliance within 0.1% of reference (ANSYS/TopOpt benchmark) # Volume constraint satisfied to 0.1%

## Variations

Variations
# Vary: load position/direction, support configuration (cantilever, bridge, MBB beam),
#        volume fraction (0.2-0.6), multiple load cases, 3D extension
# Public: standard MBB beam and cantilever with TopOpt-88 reference solutions
# Dev: multi-load, asymmetric supports

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/06_structural_mechanics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/06_structural_mechanics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/06_structural_mechanics/spec.md`
