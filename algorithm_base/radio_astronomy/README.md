# Radio Aperture Synthesis (`radio_astronomy`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `clean_dl` | RadioAST-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.radio_astronomy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.radio_astronomy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| POLISH | 2022 | 55.9 | 20.5 | gap |
| Adjoint [proxy] (PWM) | — | 41.0 | 20.5 | gap |
| PnP-ADMM [proxy] (PWM) | — | 41.0 | 20.5 | gap |
| RadioAST-DL [proxy] (PWM) | — | 41.0 | 20.5 | gap |
| precomputed_baseline (test) | — | 37.3 | 20.5 | gap |
| U-Net denoising | 2021 | 35.0 | 20.5 | gap |
| CLEAN | 1974 | 25.0 | 20.5 | partial |
