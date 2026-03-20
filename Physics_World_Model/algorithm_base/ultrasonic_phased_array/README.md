# Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `upa_dl` | TFM-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ultrasonic_phased_array import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ultrasonic_phased_array import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CycleSR | 2025 | 39.3 | 33.5 | partial |
| CinCGAN | 2025 | 36.4 | 33.5 | done |
| Adjoint [proxy] (PWM) | — | 35.2 | 33.5 | done |
| PnP-ADMM [proxy] (PWM) | — | 35.2 | 33.5 | done |
| precomputed_baseline (test) | — | 31.1 | 33.5 | done |
| TFM (Total Focusing Method) | 2004 | 28.0 | 33.5 | done |
