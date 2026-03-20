# Specification: Fresnel Diffraction (Sanity Check)

> Source: `papers/universal_simulation/benchmark/09_optics/spec.md`

## Equations

```
# Fresnel diffraction integral:
#   U(x, y, z) = (exp(ikz) / (i*lambda*z)) *
#     integral integral U_0(x_a, y_a) *
#     exp(i*pi/(lambda*z) * ((x-x_a)^2 + (y-y_a)^2)) dx_a dy_a
#
# For circular aperture, reduces to Lommel functions (analytical)
# Fresnel number: N_F = R^2 / (lambda * z)

equations: |
  U(x,y,z) = (exp(ikz)/(i*lambda*z)) * FT{U_0 * exp(i*pi*(x_a^2+y_a^2)/(lambda*z))}
  I(x,y) = |U(x,y,z)|^2
  Fresnel_number: N_F = R^2 / (lambda * z)
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Intensity pattern I(x, y) on 512x512 observation grid
# 2. On-axis intensity I(0, 0, z) vs z for z in [0.01, 1.0] m
# 3. Radial intensity profile I(r) at z = 0.1 m
# 4. Encircled energy: E(r) = integral_0^r I(r') 2*pi*r' dr'
observables:
  - intensity_pattern: I(x, y) on 512x512 grid  # W/m^2 (normalized)

**Tolerance**: tolerance:   intensity_L2_relative: 1.0e-5

## Variations

Variations
# Vary: aperture shape (circle, rectangle, slit, annular), wavelength (400-800 nm),
#        propagation distance (near-field to far-field), coherence (partial),
#        aberrations (defocus, astigmatism)
# Public: circular aperture with Lommel function analytical reference

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/09_optics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/09_optics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/09_optics/spec.md`
