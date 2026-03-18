# Entangled Photon Microscopy (`entangled_photon`)

Category: Quantum Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `qgi_dl` | QGI-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.entangled_photon import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.entangled_photon import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 32.8 | 27.2 | partial |
| PnP-ADMM [proxy] (PWM) | — | 32.8 | 27.2 | partial |
| QGI-DL [proxy] (PWM) | — | 32.8 | 27.2 | partial |
| precomputed_baseline (test) | — | 31.8 | 27.2 | partial |
| Compressed sensing QI | 2013 | 18.0 | 27.2 | done |
| Coincidence counting | 2002 | 15.0 | 27.2 | done |
