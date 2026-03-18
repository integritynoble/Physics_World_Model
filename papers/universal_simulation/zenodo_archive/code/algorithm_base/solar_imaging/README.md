# Solar EUV/X-ray Imaging (`solar_imaging`)

Category: Astronomy & Space Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `solar_dl` | SolarNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.solar_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.solar_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2021 | 33.0 | 34.0 | done |
| Adjoint [proxy] (PWM) | — | 31.1 | 34.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 31.1 | 34.0 | done |
| SolarNet [proxy] (PWM) | — | 31.1 | 34.0 | done |
| Pixon | 1991 | 30.0 | 34.0 | done |
| precomputed_baseline (test) | — | 28.4 | 34.0 | done |
| Richardson-Lucy | 1972 | 25.0 | 34.0 | done |
