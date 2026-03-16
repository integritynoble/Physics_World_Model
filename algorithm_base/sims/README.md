# Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `sims_dl` | SIMS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.sims import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sims import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PCA denoising | 2010 | 24.0 | 26.0 | done |
| Adjoint [proxy] (PWM) | — | 22.6 | 26.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 22.6 | 26.0 | done |
| SIMS-Net [proxy] (PWM) | — | 22.6 | 26.0 | done |
| Dead-time correction | 2000 | 22.0 | 26.0 | done |
| precomputed_baseline (test) | — | 20.5 | 26.0 | done |
| De-MSI (DL) | 2025 | 18.9 | 26.0 | done |
