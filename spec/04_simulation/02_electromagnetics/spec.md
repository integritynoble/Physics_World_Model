# Specification: Rectangular Waveguide Mode Analysis

> Source: `papers/universal_simulation/benchmark/02_electromagnetics/spec.md`

## Equations

```
# Helmholtz equation for TE modes (H_z component):
#   nabla_t^2 H_z + k_c^2 H_z = 0
#   where k_c^2 = omega^2 * mu * epsilon - beta^2
#
# For TM modes (E_z component):
#   nabla_t^2 E_z + k_c^2 E_z = 0
#
# Cutoff frequencies:
#   f_c(m,n) = (1 / 2*pi*sqrt(mu*epsilon)) * sqrt((m*pi/a)^2 + (n*pi/b)^2)
#
# Analytical solution (for validation):
#   TE_mn: H_z = H_0 * cos(m*pi*x/a) * cos(n*pi*y/b)
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Cutoff frequencies f_c(m,n) for first 10 modes
# 2. Mode field patterns H_z(x,y) or E_z(x,y) on 64x64 grid
# 3. Propagation constants beta(f) for f in [8, 12] GHz
# 4. Attenuation constant alpha(f) for lossy walls (sigma_w = 5.8e7 S/m, copper)
observables:
  - cutoff_frequencies: f_c for (m,n) in first 10 modes  # Hz

**Tolerance**: tolerance:   cutoff_freq_relative: 1.0e-8

## Variations

Variations
# Vary: waveguide dimensions (WR-90 to WR-430), fill material (air, Teflon, alumina),
#        wall conductivity (PEC to lossy Cu/Al), frequency range, mode count
# Public: standard WR-90, analytical ground truth
# Dev: non-standard dimensions, dielectric-filled

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/02_electromagnetics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/02_electromagnetics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/02_electromagnetics/spec.md`
