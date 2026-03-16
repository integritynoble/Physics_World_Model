# Pump-Probe Microscopy (`pump_probe`)

Category: Ultrafast Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `pp_dl` | PumpProbe-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.pump_probe import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.pump_probe import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| MCR-ALS | 2000 | 26.0 | 33.4 | done |
| Adjoint [proxy] (PWM) | — | 23.3 | 33.4 | done |
| PnP-ADMM [proxy] (PWM) | — | 23.3 | 33.4 | done |
| PumpProbe-Net [proxy] (PWM) | — | 23.3 | 33.4 | done |
| SVD analysis | 2000 | 22.0 | 33.4 | done |
| precomputed_baseline (test) | — | 18.6 | 33.4 | done |
| Simple averaging | 2000 | 18.0 | 33.4 | done |
