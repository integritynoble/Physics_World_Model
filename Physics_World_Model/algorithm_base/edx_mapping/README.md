# STEM-EDX Elemental Mapping (`edx_mapping`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | Richardson-Lucy (high quality) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `edx_dl` | Richardson-Lucy (DL baseline) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231 |

## Usage

```python
# Import and run
from algorithm_base.edx_mapping import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.edx_mapping import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| NMF denoising | 2015 | 26.0 | 32.1 | done |
| Richardson-Lucy (PWM) | — | 24.1 | 32.1 | done |
| Richardson-Lucy (high quality) (PWM) | — | 24.1 | 32.1 | done |
| Richardson-Lucy (DL baseline) (PWM) | — | 24.1 | 32.1 | done |
| precomputed_baseline (test) | — | 24.1 | 32.1 | done |
| PCA denoising | 2010 | 24.0 | 32.1 | done |
