# X-ray Fluorescence Tomography (`xrf_tomo`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `xrft_dl` | XRFT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xrf_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xrf_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| 1D-CNN + U-Net | 2025 | 39.1 | 7.9 | gap |
| Optimized SCUNet | 2024 | 39.0 | 7.9 | gap |
| SIRT | 1972 | 26.0 | 7.9 | gap |
| FBP reconstruction | 2000 | 25.0 | 7.9 | gap |
| FBP | 1971 | 22.0 | 7.9 | gap |
| Adjoint [proxy] (PWM) | — | 16.6 | 7.9 | partial |
| PnP-ADMM [proxy] (PWM) | — | 16.6 | 7.9 | partial |
| XRFT-Net [proxy] (PWM) | — | 16.6 | 7.9 | partial |
| precomputed_baseline (test) | — | 15.6 | 7.9 | partial |
