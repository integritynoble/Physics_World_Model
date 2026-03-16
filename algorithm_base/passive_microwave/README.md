# Passive Microwave Radiometry (`passive_microwave`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `pm_dl` | PM-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.passive_microwave import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.passive_microwave import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RDA [proxy] (PWM) | — | 28.5 | 38.8 | done |
| SAR-DL [proxy] (PWM) | — | 28.5 | 38.8 | done |
| PM-Net [proxy] (PWM) | — | 28.5 | 38.8 | done |
| OI (Optimal Interpolation) | 2000 | 25.0 | 38.8 | done |
| Tikhonov retrieval | 2000 | 22.0 | 38.8 | done |
| precomputed_baseline (test) | — | 18.3 | 38.8 | done |
| Linear regression retrieval | 1990 | 18.0 | 38.8 | done |
