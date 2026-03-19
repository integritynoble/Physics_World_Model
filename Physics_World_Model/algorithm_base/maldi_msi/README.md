# MALDI Mass Spectrometry Imaging (`maldi_msi`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `msi_dl` | MSI-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.maldi_msi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.maldi_msi import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 34.8 | 25.9 | partial |
| PnP-ADMM [proxy] (PWM) | — | 34.8 | 25.9 | partial |
| MSI-UNet [proxy] (PWM) | — | 34.8 | 25.9 | partial |
| precomputed_baseline (test) | — | 27.1 | 25.9 | done |
| NMF denoising | 2010 | 25.0 | 25.9 | done |
| Peak picking | 2000 | 22.0 | 25.9 | done |
