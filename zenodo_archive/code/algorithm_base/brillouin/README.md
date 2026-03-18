# Brillouin Microscopy (`brillouin`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `brillouin_dl` | Brillouin-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.brillouin import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.brillouin import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 40.4 | 25.6 | gap |
| PnP-ADMM [proxy] (PWM) | — | 40.4 | 25.6 | gap |
| Brillouin-Net [proxy] (PWM) | — | 40.4 | 25.6 | gap |
| precomputed_baseline (test) | — | 35.8 | 25.6 | gap |
| VIPA analysis | 2010 | 28.0 | 25.6 | done |
| Lorentzian fitting | 2000 | 25.0 | 25.6 | done |
