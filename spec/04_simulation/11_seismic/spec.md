# Specification: Seismic Full-Waveform Inversion (Marmousi-2)

> Source: `papers/universal_simulation/benchmark/11_seismic/spec.md`

## Equations

```
# Forward model (acoustic wave equation):
#   (1/c(x,z)^2) * d^2p/dt^2 = nabla^2 p + s(t) * delta(x - x_s)
#   c(x,z) = P-wave velocity field (unknown, to be inverted)
#   p(x, z, t) = pressure wavefield
#   s(t) = source wavelet
#
# Inverse problem (FWI):
#   min_{c} J(c) = 0.5 * sum_s sum_r ||p_obs(x_r, t; x_s) - p_syn(x_r, t; x_s; c)||_2^2
#                  + lambda * TV(c)
#
# Gradient via adjoint-state method:
#   dJ/dc = -2/c^3 * integral_0^T p_forward * p_adjoint dt
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Inverted velocity model c_inv(x, z) on 1700 x 350 grid
# 2. Velocity PSNR: 10*log10(max(c_true)^2 / MSE(c_inv, c_true))
# 3. Data fit: ||p_obs - p_syn(c_inv)||^2 / ||p_obs||^2
# 4. Structural similarity: SSIM(c_inv, c_true)
# 5. Depth profiles: c(z) at x = 5000, 8500, 12000 m
# Validated against SEG/EAGE Marmousi community benchmark

**Tolerance**: tolerance:   velocity_PSNR_minimum: 25.0  # dB

## Variations

Variations
# Vary: model (Marmousi, Overthrust, BP, synthetic layered), frequency range,
#        source/receiver geometry (marine, land, OBS), noise level,
#        starting model quality, regularization type
# Public: Marmousi-2 with reference velocity, full-aperture acquisition

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/11_seismic/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/11_seismic/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/11_seismic/spec.md`
