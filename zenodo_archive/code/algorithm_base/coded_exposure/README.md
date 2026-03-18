# Coded Exposure / Flutter Shutter (`coded_exposure`)

Category: Computational Photography

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `coded_dl` | FlowNet-Coded [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.coded_exposure import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.coded_exposure import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 38.2 | 28.7 | partial |
| PnP-ADMM [proxy] (PWM) | — | 38.2 | 28.7 | partial |
| FlowNet-Coded [proxy] (PWM) | — | 38.2 | 28.7 | partial |
| Restormer | 2022 | 32.9 | 28.7 | partial |
| MPRNet | 2021 | 32.7 | 28.7 | partial |
| precomputed_baseline (test) | — | 32.1 | 28.7 | partial |
| Wiener (flutter shutter) | 2006 | 26.0 | 28.7 | done |
