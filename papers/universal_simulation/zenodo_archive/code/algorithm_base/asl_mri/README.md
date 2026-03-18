# Arterial Spin Labeling (ASL) MRI (`asl_mri`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `asl_dl` | ASL-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.asl_mri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.asl_mri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| HUST (Transformer) 3D | 2025 | 45.1 | 22.3 | gap |
| HUST (Transformer) 2D | 2025 | 33.7 | 22.3 | gap |
| ASLRDB (Dilated+RDB) | 2025 | 25.0 | 22.3 | done |
| Control-label subtraction | 1998 | 22.0 | 22.3 | done |
| FBP [proxy] (PWM) | — | 12.9 | 22.3 | done |
| DL-Recon [proxy] (PWM) | — | 12.9 | 22.3 | done |
| ASL-Net [proxy] (PWM) | — | 12.9 | 22.3 | done |
| precomputed_baseline (test) | — | 10.9 | 22.3 | done |
