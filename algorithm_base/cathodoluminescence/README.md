# Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `cl_dl` | CL-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.cathodoluminescence import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cathodoluminescence import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 38.7 | 33.7 | partial |
| PnP-ADMM [proxy] (PWM) | — | 38.7 | 33.7 | partial |
| CL-Net [proxy] (PWM) | — | 38.7 | 33.7 | partial |
| precomputed_baseline (test) | — | 28.9 | 33.7 | done |
| PCA denoising | 2010 | 25.0 | 33.7 | done |
| Spectral unmixing | 2000 | 22.0 | 33.7 | done |
