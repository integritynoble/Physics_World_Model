# US/MRI Fusion (`us_mri`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `us_mri_dl` | US-MRI-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.us_mri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.us_mri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| VoxelMorph | 2019 | 30.0 | 24.6 | partial |
| Adjoint [proxy] (PWM) | — | 28.3 | 24.6 | partial |
| PnP-ADMM [proxy] (PWM) | — | 28.3 | 24.6 | partial |
| US-MRI-Net [proxy] (PWM) | — | 28.3 | 24.6 | partial |
| precomputed_baseline (test) | — | 25.5 | 24.6 | done |
| B-spline FFD | 2003 | 25.0 | 24.6 | done |
| Demons registration | 1998 | 22.0 | 24.6 | done |
| Affine registration | 2000 | 21.0 | 24.6 | done |
