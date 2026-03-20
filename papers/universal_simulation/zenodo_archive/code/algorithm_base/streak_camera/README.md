# Streak Camera Imaging (`streak_camera`)

Category: Ultrafast Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `streak_dl` | StreakNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.streak_camera import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.streak_camera import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 36.7 | 26.8 | partial |
| StreakNet [proxy] (PWM) | — | 36.7 | 26.8 | partial |
| precomputed_baseline (test) | — | 30.8 | 26.8 | partial |
| PnP-BM3D (sim) | 2022 | 29.2 | 26.8 | done |
| PnP-FFDNet (sim) | 2022 | 28.4 | 26.8 | done |
| Temporal deconvolution | 2000 | 25.0 | 26.8 | done |
| Wiener deconvolution | 1949 | 22.0 | 26.8 | done |
