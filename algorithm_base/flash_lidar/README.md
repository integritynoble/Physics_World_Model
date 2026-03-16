# Flash LiDAR (`flash_lidar`)

Category: Depth Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `flash_dl` | FlashLiDAR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.flash_lidar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.flash_lidar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Joint depth+reflectivity DNN | 2025 | 29.1 | 31.7 | done |
| TCSPC histogram | 2000 | 22.0 | 31.7 | done |
| Matched filter SPAD | 2010 | 18.0 | 31.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 5.3 | 31.7 | done |
| FlashLiDAR-Net [proxy] (PWM) | — | 5.3 | 31.7 | done |
| precomputed_baseline (test) | — | 4.3 | 31.7 | done |
