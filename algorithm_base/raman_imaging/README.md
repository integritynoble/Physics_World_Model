# Raman Imaging / Microscopy (`raman_imaging`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `raman_dl` | RamanNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.raman_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.raman_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DeepeR (1D ResUNet) | 2022 | 46.2 | 38.0 | partial |
| PCA denoising | 2000 | 39.4 | 38.0 | done |
| Adjoint [proxy] (PWM) | — | 21.6 | 38.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 21.6 | 38.0 | done |
| RamanNet [proxy] (PWM) | — | 21.6 | 38.0 | done |
| Savitzky-Golay | 1964 | 20.0 | 38.0 | done |
| precomputed_baseline (test) | — | 19.7 | 38.0 | done |
