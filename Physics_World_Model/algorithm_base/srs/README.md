# Stimulated Raman Scattering (SRS) Microscopy (`srs`)

Category: Spectroscopy & Spectral Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `srs_dl` | SRS-DeepSpec [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.srs import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.srs import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 45.2 | 36.6 | partial |
| PnP-ADMM [proxy] (PWM) | — | 45.2 | 36.6 | partial |
| SRS-DeepSpec [proxy] (PWM) | — | 45.2 | 36.6 | partial |
| precomputed_baseline (test) | — | 30.6 | 36.6 | done |
| U-Net CNN | 2019 | 28.9 | 36.6 | done |
| SHRED | 2021 | 25.0 | 36.6 | done |
| Spectral unmixing | 2000 | 24.0 | 36.6 | done |
| UHRED (unsupervised) | 2021 | 22.0 | 36.6 | done |
| PURE-LET | 2019 | 13.5 | 36.6 | done |
