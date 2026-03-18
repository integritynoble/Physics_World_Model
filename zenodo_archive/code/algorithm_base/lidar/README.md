# LiDAR Scanner (`lidar`)

Category: Depth Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (depth) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | PointNeXt [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | PointNet++ [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.lidar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lidar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PointNeXt [proxy] (PWM) | — | 52.0 | 36.9 | gap |
| PointNet++ [proxy] (PWM) | — | 52.0 | 36.9 | gap |
| BP-Net | 2022 | 36.0 | 36.9 | done |
| FISTA-L2 (depth) (PWM) | — | 35.8 | 36.9 | done |
| precomputed_baseline (test) | — | 35.8 | 36.9 | done |
| CompletionFormer | 2023 | 35.5 | 36.9 | done |
| NLSPN | 2020 | 35.0 | 36.9 | done |
| Bilateral Filter | 1998 | 25.0 | 36.9 | done |
