# Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

Category: Computational Photography

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `event_dl` | E2VID+ [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.event_camera import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.event_camera import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| HyperE2VID | 2024 | 14.8 | 35.9 | done |
| ET-Net | 2021 | 13.3 | 35.9 | done |
| E2VID+ | 2020 | 11.5 | 35.9 | done |
| SPADE-E2VID | 2021 | 10.4 | 35.9 | done |
| Adjoint [proxy] (PWM) | — | 9.7 | 35.9 | done |
| PnP-ADMM [proxy] (PWM) | — | 9.7 | 35.9 | done |
| precomputed_baseline (test) | — | 7.6 | 35.9 | done |
| E2VID | 2019 | 7.5 | 35.9 | done |
| Raw event accumulation | 2014 | 5.0 | 35.9 | done |
