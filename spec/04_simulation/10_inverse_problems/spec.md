# Specification: CT Reconstruction (LoDoPaB)

> Source: `papers/universal_simulation/benchmark/10_inverse_problems/spec.md`

## Equations

```
# Forward model (Radon transform):
#   y = A * x + eta
#   A = Radon transform (parallel beam, n_angles projections)
#   eta ~ Poisson noise (realistic clinical noise model)
#
# Inverse problem (reconstruction):
#   min_x 0.5 * ||A*x - y||_2^2 + lambda_1 * ||x||_TV + lambda_2 * ||x||_2^2
#   subject to: x >= 0 (non-negativity)
#
# TV = total variation = sum |nabla x|
# Solved via ADMM or proximal gradient

```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Reconstructed image x_hat on 128x128 grid
# 2. PSNR: 10 * log10(max(x)^2 / MSE)  (dB)
# 3. SSIM: structural similarity index
# 4. Relative error: ||x_hat - x_true|| / ||x_true||
# Validated against LoDoPaB-CT dataset (Leuschner et al., 2021)
# 200 real clinical cases from Siemens scanner

**Tolerance**: tolerance:   PSNR_minimum: 30.0      # dB

## Variations

Variations
# Vary: n_angles (32-512), noise level (low to clinical), geometry
#        (parallel, fan-beam, cone-beam), regularization (TV, wavelet, learned),
#        image size (64x64 to 512x512)
# Public: LoDoPaB 128-angle with ground truth provided

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/10_inverse_problems/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/10_inverse_problems/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/10_inverse_problems/spec.md`
