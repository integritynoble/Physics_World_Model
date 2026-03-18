# TIRF Microscopy (`tirf`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (TIRF) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | TIRF-Net (CARE) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | TIRF-SRRF [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.tirf import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.tirf import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RED-fairSIM | 2021 | 33.2 | 40.8 | done |
| CARE | 2018 | 33.0 | 40.8 | done |
| TIRF-SRRF [proxy] (PWM) | — | 32.2 | 40.8 | done |
| precomputed_baseline (test) | — | 31.2 | 40.8 | done |
| Richardson-Lucy | 1972 | 28.0 | 40.8 | done |
