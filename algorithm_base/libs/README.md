# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `libs_dl` | LIBS-CNN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.libs import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.libs import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 31.2 | 21.2 | partial |
| PnP-ADMM [proxy] (PWM) | — | 31.2 | 21.2 | partial |
| LIBS-CNN [proxy] (PWM) | — | 31.2 | 21.2 | partial |
| precomputed_baseline (test) | — | 26.5 | 21.2 | partial |
| PLS regression | 2005 | 25.0 | 21.2 | partial |
| Peak identification | 2000 | 22.0 | 21.2 | done |
