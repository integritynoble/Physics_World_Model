# Ocean Color Remote Sensing (`ocean_color`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `oc_dl` | OC-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ocean_color import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ocean_color import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RDA [proxy] (PWM) | — | 53.5 | 36.0 | gap |
| SAR-DL [proxy] (PWM) | — | 53.5 | 36.0 | gap |
| OC-Net [proxy] (PWM) | — | 53.5 | 36.0 | gap |
| precomputed_baseline (test) | — | 44.2 | 36.0 | partial |
| SRCNN | 2023 | 25.2 | 36.0 | done |
| MUMM | 2000 | 22.0 | 36.0 | done |
