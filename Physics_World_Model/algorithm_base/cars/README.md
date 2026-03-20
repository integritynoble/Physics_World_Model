# Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `cars_dl` | CARS-DeepSpec [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.cars import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cars import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 27.9 | 32.4 | done |
| PnP-ADMM [proxy] (PWM) | — | 27.9 | 32.4 | done |
| CARS-DeepSpec [proxy] (PWM) | — | 27.9 | 32.4 | done |
| MEM (Maximum Entropy Method) | 2006 | 25.0 | 32.4 | done |
| DnCNN | 2023 | 23.0 | 32.4 | done |
| N2N (Noise2Noise) | 2023 | 20.6 | 32.4 | done |
| Median Filter | 2023 | 20.1 | 32.4 | done |
| precomputed_baseline (test) | — | 16.7 | 32.4 | done |
| Raw CARS (no correction) | 2000 | 15.0 | 32.4 | done |
