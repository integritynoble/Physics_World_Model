# Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `us_dl_enhance` | US-DeepSight [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ceus import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ceus import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Real-time CNN | 2022 | 36.1 | 24.9 | gap |
| GAN-RW (Residual Dense) | 2022 | 33.9 | 24.9 | partial |
| FBP [proxy] (PWM) | — | 26.4 | 24.9 | done |
| DL-Recon [proxy] (PWM) | — | 26.4 | 24.9 | done |
| US-DeepSight [proxy] (PWM) | — | 26.4 | 24.9 | done |
| Singular value decomposition | 2015 | 25.0 | 24.9 | done |
| precomputed_baseline (test) | — | 24.5 | 24.9 | done |
| Temporal averaging | 2000 | 22.0 | 24.9 | done |
