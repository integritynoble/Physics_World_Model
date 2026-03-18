# Time-of-Flight Depth Camera (`tof_camera`)

Category: Depth Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (depth) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | ToF-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | ToF-MPI Deconv [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.tof_camera import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.tof_camera import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Phase unwrapping | 2000 | 47.6 | 45.2 | done |
| ToF-Net [proxy] (PWM) | — | 47.6 | 45.2 | done |
| ToF-MPI Deconv [proxy] (PWM) | — | 47.6 | 45.2 | done |
| FISTA-L2 (depth) (PWM) | — | 42.2 | 45.2 | done |
| precomputed_baseline (test) | — | 42.2 | 45.2 | done |
| DeepToF | 2017 | 32.0 | 45.2 | done |
| Bilateral filter (depth) | 2014 | 29.5 | 45.2 | done |
