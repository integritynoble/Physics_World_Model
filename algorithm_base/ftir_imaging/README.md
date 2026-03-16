# FTIR Spectroscopic Imaging (`ftir_imaging`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ftir_dl` | FTIR-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ftir_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ftir_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 35.6 | 25.0 | gap |
| PnP-ADMM [proxy] (PWM) | — | 35.6 | 25.0 | gap |
| FTIR-UNet [proxy] (PWM) | — | 35.6 | 25.0 | gap |
| precomputed_baseline (test) | — | 34.6 | 25.0 | partial |
| U-Net SR FTIR | 2022 | 30.0 | 25.0 | partial |
| MCR-ALS | 2000 | 28.0 | 25.0 | done |
| ATR correction | 2000 | 24.0 | 25.0 | done |
