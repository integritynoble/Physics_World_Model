# Eddy Current Imaging (`eddy_current`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ec_dl` | ECT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.eddy_current import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.eddy_current import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Wavelet denoising | 2000 | 25.0 | 33.3 | done |
| Adjoint [proxy] (PWM) | — | 23.9 | 33.3 | done |
| PnP-ADMM [proxy] (PWM) | — | 23.9 | 33.3 | done |
| ECT-Net [proxy] (PWM) | — | 23.9 | 33.3 | done |
| precomputed_baseline (test) | — | 22.9 | 33.3 | done |
| Impedance plane analysis | 2000 | 22.0 | 33.3 | done |
