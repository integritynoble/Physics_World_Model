# Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `cle_dl` | CLE-Net (CARE) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.confocal_endomicroscopy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.confocal_endomicroscopy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FBP [proxy] (PWM) | — | 41.5 | 55.2 | done |
| DL-Recon [proxy] (PWM) | — | 41.5 | 55.2 | done |
| CLE-Net (CARE) [proxy] (PWM) | — | 41.5 | 55.2 | done |
| Self-supervised denoising | 2024 | 36.1 | 55.2 | done |
| precomputed_baseline (test) | — | 34.0 | 55.2 | done |
| Richardson-Lucy | 1972 | 28.0 | 55.2 | done |
