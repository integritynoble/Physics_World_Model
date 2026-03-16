# Electron Energy Loss Spectroscopy (EELS) (`eels`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (Fourier ratio) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | EELS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | MLLS-EELS [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `eels_dl` | EELS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.eels import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.eels import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Deep CNN Denoiser | 2021 | 42.9 | 27.0 | gap |
| FISTA-L2 (Fourier ratio) [proxy] (PWM) | — | 28.4 | 27.0 | done |
| EELS-Net [proxy] (PWM) | — | 28.4 | 27.0 | done |
| MLLS-EELS [proxy] (PWM) | — | 28.4 | 27.0 | done |
| PCA denoising | 2012 | 28.0 | 27.0 | done |
| NMF decomposition | 2015 | 26.0 | 27.0 | done |
| precomputed_baseline (test) | — | 25.2 | 27.0 | done |
