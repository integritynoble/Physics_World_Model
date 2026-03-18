# X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `xrf_dl` | XRF-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xrf_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xrf_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DnCNN (XFCT) | 2024 | 49.4 | 30.7 | gap |
| NLM (XFCT) | 2024 | 39.9 | 30.7 | partial |
| Adjoint [proxy] (PWM) | — | 29.8 | 30.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 29.8 | 30.7 | done |
| XRF-Net [proxy] (PWM) | — | 29.8 | 30.7 | done |
| precomputed_baseline (test) | — | 26.7 | 30.7 | done |
| PCA denoising | 2010 | 25.0 | 30.7 | done |
| Fundamental parameters | 2000 | 22.0 | 30.7 | done |
