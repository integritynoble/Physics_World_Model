# CEST MRI (`cest_mri`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `cest_dl` | CEST-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.cest_mri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cest_mri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FBP [proxy] (PWM) | — | 44.3 | 30.1 | gap |
| DL-Recon [proxy] (PWM) | — | 44.3 | 30.1 | gap |
| CEST-Net [proxy] (PWM) | — | 44.3 | 30.1 | gap |
| ResUNet-NE | 2023 | 35.0 | 30.1 | partial |
| precomputed_baseline (test) | — | 32.1 | 30.1 | done |
| Z-spectrum fitting | 2003 | 25.0 | 30.1 | done |
