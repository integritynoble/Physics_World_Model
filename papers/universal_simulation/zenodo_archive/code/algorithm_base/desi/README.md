# DESI Mass Spectrometry Imaging (`desi`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `desi_dl` | DESI-SegNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.desi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.desi import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| NMF denoising | 2015 | 25.0 | 27.0 | done |
| Peak fitting | 2000 | 22.0 | 27.0 | done |
| Adjoint [proxy] (PWM) | — | 16.1 | 27.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 16.1 | 27.0 | done |
| DESI-SegNet [proxy] (PWM) | — | 16.1 | 27.0 | done |
| Gaussian smoothing | 2000 | 16.0 | 27.0 | done |
| precomputed_baseline (test) | — | 15.1 | 27.0 | done |
