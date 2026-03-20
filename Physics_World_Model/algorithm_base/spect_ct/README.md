# SPECT/CT Fusion (`spect_ct`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `spectct_dl` | SPECT-CT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.spect_ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spect_ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| GAN projection-space denoising | 2022 | 42.5 | 28.4 | gap |
| U2-Net (bone SPECT/CT) | 2022 | 40.8 | 28.4 | gap |
| OSEM + CT AC | 2000 | 26.0 | 28.4 | done |
| MLEM | 1982 | 24.0 | 28.4 | done |
| MLEM (low-count, 2 iter) | 1982 | 15.0 | 28.4 | done |
| Adjoint [proxy] (PWM) | — | 14.6 | 28.4 | done |
| PnP-ADMM [proxy] (PWM) | — | 14.6 | 28.4 | done |
| SPECT-CT-Net [proxy] (PWM) | — | 14.6 | 28.4 | done |
| MLEM (1 iter, 1/20 counts) | 1982 | 13.0 | 28.4 | done |
| precomputed_baseline (test) | — | 11.4 | 28.4 | done |
