# Susceptibility-Weighted Imaging (SWI) (`swi`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `swi_dl` | SWI-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.swi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.swi import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DeepSWI (cGAN) | 2023 | 36.9 | 15.9 | gap |
| Homodyne filtering | 2004 | 28.0 | 15.9 | gap |
| FBP [proxy] (PWM) | — | 12.9 | 15.9 | done |
| DL-Recon [proxy] (PWM) | — | 12.9 | 15.9 | done |
| SWI-Net [proxy] (PWM) | — | 12.9 | 15.9 | done |
| precomputed_baseline (test) | — | 10.9 | 15.9 | done |
