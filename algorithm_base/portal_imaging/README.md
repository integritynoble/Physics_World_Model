# Portal Imaging (EPID) (`portal_imaging`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `portal_dl` | PortalDL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.portal_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.portal_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CycleGAN+Attention+Residual | 2024 | 34.0 | 36.2 | done |
| CycleGAN MVCT-to-kVCT | 2021 | 32.7 | 36.2 | done |
| Monte Carlo correction | 2005 | 28.0 | 36.2 | done |
| Flat-field correction | 2000 | 25.0 | 36.2 | done |
| FBP [proxy] (PWM) | — | 23.8 | 36.2 | done |
| DL-Recon [proxy] (PWM) | — | 23.8 | 36.2 | done |
| PortalDL [proxy] (PWM) | — | 23.8 | 36.2 | done |
| precomputed_baseline (test) | — | 17.3 | 36.2 | done |
| Raw EPID (uncorrected) | 2000 | 15.0 | 36.2 | done |
